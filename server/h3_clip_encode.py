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
import os
import sys
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
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo  # noqa: E402

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
    global CLIP, VIDEO_VAE, AUDIO_VAE
    t0 = time.time()
    print(f"[h3-encode] 載入 CLIP: {clip_path}", flush=True)
    CLIP = load_clip([clip_path], clip_type=CLIPType.MINIMAX)
    comfy.model_management.load_models_gpu([CLIP.patcher])
    print(f"[h3-encode] CLIP 就緒 {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    sd, metadata = comfy.utils.load_torch_file(video_vae_path, return_metadata=True)
    VIDEO_VAE = VAE(sd=sd, metadata=metadata)
    print(f"[h3-encode] video VAE 就緒 {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    sd, metadata = comfy.utils.load_torch_file(audio_vae_path, return_metadata=True)
    AUDIO_VAE = VAE(sd=sd, metadata=metadata)
    print(f"[h3-encode] audio VAE 就緒 {time.time() - t0:.1f}s", flush=True)


def run_encode(data):
    t0 = time.time()
    PROGRESS.update(status="running", elapsed=0.0)
    prompt = data.get("prompt") or ""
    width = int(data.get("width", 1344))
    height = int(data.get("height", 768))
    length = int(data.get("length", 124))
    ref_image_size = data.get("ref_image_size") or "match"
    out = MiniMaxH3ReferenceToVideo.execute(
        CLIP, VIDEO_VAE, AUDIO_VAE, prompt, width, height, length, ref_image_size,
        ref_images=data.get("ref_images"),
        ref_videos=data.get("ref_videos"),
        ref_video_audios=data.get("ref_video_audios"),
        ref_audios=data.get("ref_audios"),
    )
    positive, latent = out.args
    elapsed = time.time() - t0
    PROGRESS.update(status="done", elapsed=elapsed)
    print(f"[h3-encode] 完成 {elapsed:.1f}s", flush=True)
    return {"positive": positive, "latent": latent}


app = FastAPI(title="MiniMax H3 Remote Encode")
_POOL = ThreadPoolExecutor(max_workers=1)


@app.get("/health")
def health():
    return {"status": "ok", "clip": CLIP is not None, "video_vae": VIDEO_VAE is not None}


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
        result = await loop.run_in_executor(_POOL, run_encode, data)
        payload = dump_bytes(result)
        print(f"[h3-encode] 回傳 {len(payload)/1e6:.1f}MB", flush=True)
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
