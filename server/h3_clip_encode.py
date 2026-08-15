#!/usr/bin/env python3
"""
MiniMax H3 遠端 encode server（官方 MiniMaxH3ReferenceToVideo 路徑）

單卡：CLIP Qwen3VL-32B INT8（offload）+ video/audio VAE encode refs。
官方流程：tokenize(minimax_ref_items) → encode_from_tokens_scheduled
         → vae.encode refs 寫進 cond['minimax_refs'] + empty AV latent。

通訊：POST /encode，body = torch.save(bytes)，回傳 {positive, latent}。
"""
import argparse
import asyncio
import io
import itertools
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import comfy  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.sd import CLIPType, VAE, load_clip  # noqa: E402
from comfy_extras.nodes_minimax_h3 import (  # noqa: E402
    MiniMaxH3ReferenceToVideo,
    _empty_av_latent,
    _resize,
    adapt_canvas,
    CANVAS_MULTIPLE,
    REF_IMAGE_SHORT_EDGE,
    FPS,
)
import node_helpers  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import Response  # noqa: E402


def _serialize(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, comfy.nested_tensor.NestedTensor):
        return {"__h3_nt__": True, "tensors": [_serialize(t) for t in obj.tensors]}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def _deserialize(obj):
    if isinstance(obj, dict):
        if obj.get("__h3_nt__"):
            return comfy.nested_tensor.NestedTensor(
                [_deserialize(t) for t in obj["tensors"]])
        return {k: _deserialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize(v) for v in obj]
    return obj


def dump_bytes(obj):
    buf = io.BytesIO()
    torch.save(_serialize(obj), buf)
    return buf.getvalue()


def load_bytes(b):
    return _deserialize(torch.load(io.BytesIO(b), weights_only=False))


CLIP = None
VIDEO_VAE = None
AUDIO_VAE = None
PROGRESS = {"status": "idle", "elapsed": 0.0}


def load_models(clip_path, video_vae_path, audio_vae_path):
    """權重載進 RAM / patcher，不上卡。官方 execute 會自己 VAE 先、CLIP 後、不夠就 offload。"""
    global CLIP, VIDEO_VAE, AUDIO_VAE
    t0 = time.time()
    print(f"[h3-encode] 載入 CLIP（CPU/offload）: {clip_path}", flush=True)
    CLIP = load_clip([clip_path], clip_type=CLIPType.MINIMAX)
    print(f"[h3-encode] CLIP patcher 就緒 {time.time() - t0:.1f}s "
          f"vram={comfy.model_management.vram_state.name}", flush=True)

    t0 = time.time()
    sd, metadata = comfy.utils.load_torch_file(video_vae_path, return_metadata=True)
    VIDEO_VAE = VAE(sd=sd, metadata=metadata)
    # 官方 tile=256 在 3080 20GB 上，768 短邊影片 encode activation 會先把卡塞滿
    if getattr(VIDEO_VAE, "first_stage_model", None) is not None:
        VIDEO_VAE.first_stage_model.tile_size = 128
        VIDEO_VAE.first_stage_model.tile_overlap_min = 32
        VIDEO_VAE.first_stage_model.tiling = True
    print(f"[h3-encode] video VAE 就緒 {time.time() - t0:.1f}s tile=128 (RAM offload)", flush=True)

    t0 = time.time()
    sd, metadata = comfy.utils.load_torch_file(audio_vae_path, return_metadata=True)
    AUDIO_VAE = VAE(sd=sd, metadata=metadata)
    print(f"[h3-encode] audio VAE 就緒 {time.time() - t0:.1f}s", flush=True)


def _free_gpu(tag):
    """權重卸回 RAM（CPU），不下盤。下一輪 load_models_gpu 從 RAM 搬回來。"""
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated(0) / 1e9
        print(f"[h3-encode] {tag} vram={used:.2f}G (weights stay in RAM)", flush=True)


def _vae_encode_pixels(pixel_nhwc):
    """官方 inner encode：影片留 CPU，17 幀切 + spatial tile。權重不夠就 RAM offload。

    不走 VAE.encode 外層。那層「先整包再 fallback」OOM 時 activation 會釘在 exception 上。
    """
    fs = VIDEO_VAE.first_stage_model
    fs.tiling = True
    fs.tile_size = 96
    fs.tile_overlap_min = 32
    frames = pixel_nhwc[..., :3]
    if frames.ndim != 4:
        raise ValueError(f"expected NHWC frames, got {tuple(frames.shape)}")
    # [F,H,W,C] -> [1,C,F,H,W]
    x = frames.movedim(-1, 1).movedim(1, 0).unsqueeze(0)
    x = x.contiguous().to("cpu", dtype=torch.float16)
    comfy.model_management.load_models_gpu(
        [VIDEO_VAE.patcher], memory_required=1_300_000_000, force_full_load=False)
    print(f"[h3-encode] vae.encode in={list(x.shape)} tile={fs.tile_size} "
          f"vram={torch.cuda.memory_allocated(0)/1e9:.2f}G", flush=True)
    with torch.no_grad():
        z = fs.encode(x, device=comfy.model_management.get_torch_device())
    z = z.to("cpu")
    print(f"[h3-encode] vae.encode out={list(z.shape)} "
          f"vram={torch.cuda.memory_allocated(0)/1e9:.2f}G", flush=True)
    return z


def _encode_refs(data, width, height, length):
    """官方 execute 的 VAE 段：ref image/video/audio → latent blocks + CLIP 用的 ref_items。"""
    latent, frame_count = _empty_av_latent(width, height, length)
    ref_items = []
    ref_blocks = []
    ref_images = data.get("ref_images") or {}
    ref_videos = data.get("ref_videos") or {}
    ref_video_audios = data.get("ref_video_audios") or {}
    ref_audios = data.get("ref_audios") or {}
    ref_image_size = data.get("ref_image_size") or "match"

    for img in ref_images.values():
        if img is None:
            continue
        h, w = img.shape[1], img.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / (w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img[:1], tw, th, "disabled")
        print(f"[h3-encode] vae image {list(resized.shape)}", flush=True)
        z = _vae_encode_pixels(resized)
        ref_items.append({"type": "image", "data": resized.cpu()})
        ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})
        comfy.model_management.soft_empty_cache()

    for name, video_frames in ref_videos.items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        cw, ch = adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(video_frames, cw, ch, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = frames.shape[0]
        if n < 5:
            raise ValueError("reference videos need at least 5 frames")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        print(f"[h3-encode] vae video {list(frames.shape)}", flush=True)
        comfy.model_management.soft_empty_cache()
        z = _vae_encode_pixels(frames)
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None:
            audio_latent, ref_audio_t = MiniMaxH3ReferenceToVideo._encode_ref_audio(
                AUDIO_VAE, soundtrack)
            if hasattr(audio_latent, "cpu"):
                audio_latent = audio_latent.cpu()
            ref_items.append({"type": "audio"})
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        qwen_frames = frames[sample_idx].contiguous().cpu()
        print(f"[h3-encode] clip video tokens frames={list(qwen_frames.shape)}", flush=True)
        ref_items.append({"type": "video", "data": qwen_frames,
                          "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                           "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                           "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})
        del frames
        comfy.model_management.soft_empty_cache()

    for audio in ref_audios.values():
        if audio is None:
            continue
        audio_latent, ref_audio_t = MiniMaxH3ReferenceToVideo._encode_ref_audio(
            AUDIO_VAE, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                           "audio_latent": audio_latent})

    return latent, ref_items, ref_blocks


def run_encode(data):
    """兩段：VAE ref 先上卡 → 卸回 RAM → CLIP（大包 vision token）offload 上卡。"""
    t0 = time.time()
    PROGRESS.update(status="running", elapsed=0.0)
    try:
        _free_gpu("before")
        prompt = data.get("prompt") or ""
        width = int(data.get("width", 1344))
        height = int(data.get("height", 768))
        length = int(data.get("length", 124))
        print(f"[h3-encode] start {width}x{height} L={length} refs="
              f"img={bool(data.get('ref_images'))} vid={bool(data.get('ref_videos'))}",
              flush=True)

        latent, ref_items, ref_blocks = _encode_refs(data, width, height, length)
        _free_gpu("after-vae")

        tokens = CLIP.tokenize(prompt, minimax_ref_items=ref_items)
        print("[h3-encode] clip encode (reserve 12G for vision/dequant)", flush=True)
        cond = CLIP.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

        elapsed = time.time() - t0
        PROGRESS.update(status="done", elapsed=elapsed)
        print(f"[h3-encode] 完成 {elapsed:.1f}s", flush=True)
        return {"positive": cond, "latent": latent}
    except Exception:
        PROGRESS.update(status="error", elapsed=time.time() - t0)
        raise
    finally:
        _free_gpu("after-clip")


app = FastAPI(title="MiniMax H3 Remote Encode")
_POOL = ThreadPoolExecutor(max_workers=1)
_QDEPTH = 0
_QLOCK = threading.Lock()
_QID = itertools.count(1)


def _run_queued(fn, *args):
    global _QDEPTH
    with _QLOCK:
        _QDEPTH += 1
        jid = next(_QID)
        waiting = _QDEPTH - 1
    print(f"[h3-encode] queue count={jid} waiting={waiting}", flush=True)
    try:
        return fn(*args)
    finally:
        with _QLOCK:
            _QDEPTH -= 1


@app.get("/health")
def health():
    return {"status": "ok", "clip": CLIP is not None, "video_vae": VIDEO_VAE is not None,
            "queue": _QDEPTH}


@app.get("/progress")
def progress():
    return dict(PROGRESS)


@app.post("/encode")
async def encode_endpoint(request: Request):
    import traceback
    try:
        body = await request.body()
        print(f"[h3-encode] /encode {len(body)/1e6:.1f}MB", flush=True)
        data = load_bytes(body)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_POOL, _run_queued, run_encode, data)
        payload = dump_bytes(result)
        print(f"[h3-encode] packed {len(payload)/1e6:.1f}MB wait client", flush=True)
        return Response(content=payload, media_type="application/octet-stream")
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return Response(content=tb.encode(), status_code=500, media_type="text/plain")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--video-vae", required=True)
    parser.add_argument("--audio-vae", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8301)
    args = parser.parse_args()
    load_models(args.clip, args.video_vae, args.audio_vae)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
