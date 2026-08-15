#!/usr/bin/env python3
"""
MiniMax H3 遠端 AV decode server

接收 denoise 完成的 NestedTensor(video, audio)：
  video → frames [N,H,W,C] fp32 [0,1]
  audio → {waveform, sample_rate}（官方 VAEDecodeAudio）
"""
import argparse
import asyncio
import gc
import io
import itertools
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import comfy  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.sd import VAE  # noqa: E402
from h3_vae_dp_worker import main as _worker_main  # noqa: E402

# ---------------------------------------------------------------------------
# 序列化協定（與 h3_server.py / remote_sampler.py 對應）
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 模型載入與 decode
# ---------------------------------------------------------------------------

VIDEO_VAE = None   # comfy.sd.VAE wrapper（video，fp16）
AUDIO_VAE = None   # comfy.sd.VAE wrapper（audio，fp32）
_VIDEO_VAE_PATH = None
_WORLD = 1          # decode 並行卡數（--dp）
_DP_MODELS = {}     # unused (process 路徑不在父進程持有卡1模型)
_DP_WORKERS = {}    # rank -> {proc, in_q, out_q}
DEVICE = torch.device("cuda:0")
PROGRESS = {"status": "idle", "elapsed": 0.0}


# _dp_process_main 已拆到 h3_vae_dp_worker.py（避免 spawn 重 import 本檔時 torch 先初始化）


def load_vae(video_path, audio_path=None):
    """載入 video VAE。world>1 時 spawn 獨立進程持有額外卡。audio 可選。"""
    global VIDEO_VAE, AUDIO_VAE, _VIDEO_VAE_PATH
    t0 = time.time()
    print(f"[h3-decode] 載入 video VAE: {video_path}", flush=True)
    _VIDEO_VAE_PATH = video_path
    sd, metadata = comfy.utils.load_torch_file(video_path, return_metadata=True)
    VIDEO_VAE = VAE(sd=sd, metadata=metadata)
    VIDEO_VAE.first_stage_model.to(DEVICE, dtype=torch.float16).eval()
    print(f"[h3-decode] video VAE cuda:0 就緒 {time.time() - t0:.1f}s", flush=True)
    if audio_path:
        t1 = time.time()
        print(f"[h3-decode] 載入 audio VAE: {audio_path}", flush=True)
        sd, metadata = comfy.utils.load_torch_file(audio_path, return_metadata=True)
        AUDIO_VAE = VAE(sd=sd, metadata=metadata)
        AUDIO_VAE.first_stage_model.to(DEVICE).eval()
        print(f"[h3-decode] audio VAE 就緒 {time.time() - t1:.1f}s", flush=True)
    if _WORLD > 1:
        ctx = mp.get_context("spawn")
        for r in range(1, _WORLD):
            t1 = time.time()
            in_q = ctx.Queue()
            out_q = ctx.Queue()
            p = ctx.Process(target=_worker_main, args=(r, video_path, in_q, out_q),
                            daemon=True)
            p.start()
            msg, rk = out_q.get(timeout=120)
            if msg != "ready":
                raise RuntimeError(f"dp worker {r} failed to start: {msg}")
            _DP_WORKERS[r] = {"proc": p, "in_q": in_q, "out_q": out_q}
            print(f"[h3-decode] video VAE worker gpu{r} 就緒 {time.time() - t1:.1f}s",
                  flush=True)


def _decode_single(z):
    """單卡：走 0.33 原版 decode（含反正規化 + decode_temporal）。"""
    with torch.no_grad():
        out = VIDEO_VAE.first_stage_model.decode(z)  # [B, 3, F, H', W'] fp32
    return out


def _decode_clip(rank, clip_z_cpu):
    """單一時間 chunk 的空間 tiled_decode，結果搬回 CPU。rank0 本進程，其餘走 worker。"""
    if rank == 0:
        clip_z = clip_z_cpu.to(DEVICE, dtype=torch.float16)
        with torch.no_grad():
            return VIDEO_VAE.first_stage_model._adaptive_decode(clip_z).to("cpu")
    w = _DP_WORKERS[rank]
    w["in_q"].put((-1, clip_z_cpu))  # placeholder, caller uses batched API
    raise RuntimeError("_decode_clip rank>0 is unused; use _decode_dp queues")


def _decode_dp(z, world):
    """chunk 級 DP：各卡算自己的時間 chunk，main 用原版 decode_temporal 迴圈拼。

    關鍵：原版 decode() 先 z*std+mean 反正規化再進 decode_temporal。
    """
    fs = VIDEO_VAE.first_stage_model

    mean = fs.latents_mean.view(1, -1, 1, 1, 1).to(device=z.device, dtype=z.dtype)
    std = fs.latents_std.view(1, -1, 1, 1, 1).to(device=z.device, dtype=z.dtype)
    z = z * std + mean

    orig_shape = tuple(z.shape)
    pad_tokens, num_chunks = fs._decode_temporal_chunks(z.shape[2])
    if pad_tokens > 0:
        pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
        z = torch.cat([z, pad_z], dim=2)

    z_cpu = z.detach().to("cpu")
    clips = []
    for i in range(num_chunks):
        t_start = i * fs.tokens_chunk_size
        t_end = t_start + fs.tokens_chunk_size + fs.token_overlap
        clips.append(z_cpu[:, :, t_start:t_end, :, :].clone())

    clip_decs = [None] * num_chunks

    # 各 rank 分到的 chunk index
    by_rank = {r: list(range(r, num_chunks, world)) for r in range(world)}

    # 先把非 0 卡的工作丟進 worker queue
    pending = 0
    for r, idxs in by_rank.items():
        if r == 0:
            continue
        w = _DP_WORKERS.get(r)
        if w is None:
            raise RuntimeError(f"dp worker gpu{r} not started (load_vae with _WORLD>1)")
        for i in idxs:
            w["in_q"].put((i, clips[i]))
            pending += 1

    # rank0 本進程算自己的 chunk（與 worker 重疊）
    for i in by_rank.get(0, []):
        clip_decs[i] = _decode_clip(0, clips[i])

    # 收 worker 結果
    got = 0
    while got < pending:
        for r, idxs in by_rank.items():
            if r == 0:
                continue
            w = _DP_WORKERS[r]
            try:
                i, dec_i = w["out_q"].get(timeout=0.1)
            except Empty:
                if not w["proc"].is_alive():
                    raise RuntimeError(f"dp worker gpu{r} died")
                continue
            clip_decs[i] = dec_i
            got += 1

    if any(x is None for x in clip_decs):
        raise RuntimeError("dp missing chunk results")

    # 以下照搬 0.33 decode_temporal 的拼圖迴圈（clip_dec 已算完）
    chunk_dec = fs.tokens_chunk_size * fs.vae_ratio_t
    split_count = int(fs.token_drop > 0) + 1
    dec = torch.empty(fs.decode_output_shape(orig_shape), dtype=torch.float32,
                      device=comfy.model_management.intermediate_device())
    dec_overlap = None
    write_pos = 0

    def write_part(part):
        nonlocal write_pos
        part_frames = part.shape[2]
        if part_frames <= 0:
            return
        part = fs._finalize_pixels(part)
        copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
        if copy_frames > 0:
            dec[:, :, write_pos:write_pos + copy_frames, :, :].copy_(
                part[:, :, :copy_frames, :, :])
            write_pos += copy_frames

    for i in range(num_chunks):
        clip_dec = clip_decs[i]
        for j in range(split_count):
            f_start = j * chunk_dec
            f_end = min(f_start + chunk_dec, clip_dec.shape[2])
            part = clip_dec[:, :, f_start:f_end, :, :]
            part = part[:, :, fs.frame_pre_padding:, :, :]
            if j == 0:
                if dec_overlap is not None:
                    part = fs.blend(dec_overlap, part, fs.frame_overlap, dim=-3)
                    dec_overlap = None
                write_part(part)
            else:
                dec_overlap = part.contiguous()
        if i == num_chunks - 1 and dec_overlap is not None:
            write_part(dec_overlap)
            dec_overlap = None

    print(f"[h3-decode] dp world={world} chunks={num_chunks}", flush=True)
    return dec


def decode_video(z, world=1):
    """z: [B, 24, T, H, W] normalized latent -> [N, H', W', 3] fp32 [0,1]."""
    z = z.to(DEVICE, dtype=torch.float16)
    if world <= 1:
        out = _decode_single(z)
    else:
        out = _decode_dp(z, world)
    out = out.permute(0, 2, 3, 4, 1).contiguous()     # [B, F, H', W', 3]
    # 對齊 0.30 VAEDecode 節點：5D -> reshape(-1, H', W', C)
    return out.reshape(-1, out.shape[-3], out.shape[-2], out.shape[-1])


def decode_audio(audio_z):
    """官方數值：first_stage_model.decode → [B,2,L]，再 std 正規化。

    不走 VAE.decode wrapper：video 剛解完卡是滿的，wrapper 整包失敗會誤用 2D tiled。
    """
    fs = AUDIO_VAE.first_stage_model
    z = audio_z.to(device=DEVICE, dtype=next(fs.parameters()).dtype)
    with torch.no_grad():
        wave = fs.decode(z).float()
    std = torch.std(wave, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    wave = wave / std
    sr = getattr(AUDIO_VAE, "audio_sample_rate_output",
                 getattr(AUDIO_VAE, "audio_sample_rate", 32000))
    return {"waveform": wave.contiguous().cpu(), "sample_rate": int(sr)}


def _video_to_cpu():
    if VIDEO_VAE is not None and VIDEO_VAE.first_stage_model is not None:
        VIDEO_VAE.first_stage_model.to("cpu")
    torch.cuda.empty_cache()


def _video_to_gpu():
    if VIDEO_VAE is not None and VIDEO_VAE.first_stage_model is not None:
        VIDEO_VAE.first_stage_model.to(DEVICE, dtype=torch.float16).eval()


def run_decode(samples, world=1):
    """samples: NestedTensor(video, audio) / tensor / dict with samples."""
    t0 = time.time()
    PROGRESS.update(status="running", elapsed=0.0)

    if isinstance(samples, dict) and "samples" in samples:
        samples = samples["samples"]

    video_z = None
    audio_z = None
    if isinstance(samples, comfy.nested_tensor.NestedTensor):
        ts = samples.tensors
        if ts:
            video_z = ts[0]
        if ts is not None and len(ts) > 1:
            audio_z = ts[-1]
    elif isinstance(samples, torch.Tensor):
        video_z = samples

    if video_z is None:
        raise ValueError("no video latent in payload")

    tv = time.time()
    frames = decode_video(video_z, world=world)
    frames = frames.cpu()
    print(f"[h3-decode] video decode world={world} {time.time() - tv:.1f}s "
          f"frames={list(frames.shape)}", flush=True)

    audio = None
    if audio_z is not None and AUDIO_VAE is not None:
        _video_to_cpu()
        ta = time.time()
        try:
            audio = decode_audio(audio_z)
        finally:
            _video_to_gpu()
        print(f"[h3-decode] audio decode {time.time() - ta:.1f}s "
              f"wave={list(audio['waveform'].shape)} sr={audio['sample_rate']}",
              flush=True)

    elapsed = time.time() - t0
    PROGRESS.update(status="done", elapsed=elapsed)
    print(f"[h3-decode] 完成 {elapsed:.1f}s", flush=True)
    return {"frames": frames, "audio": audio}


# ---------------------------------------------------------------------------
# HTTP 服務（FastAPI）
# ---------------------------------------------------------------------------

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import Response  # noqa: E402

app = FastAPI(title="MiniMax H3 Remote AV Decode")
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
    print(f"[h3-decode] queue job={jid} waiting={waiting}", flush=True)
    try:
        return fn(*args)
    finally:
        with _QLOCK:
            _QDEPTH -= 1


@app.get("/health")
def health():
    return {"status": "ok", "video_vae": VIDEO_VAE is not None,
            "audio_vae": AUDIO_VAE is not None, "dp": _WORLD, "queue": _QDEPTH}


@app.get("/progress")
def progress():
    return dict(PROGRESS)


@app.post("/decode")
async def decode_endpoint(request: Request):
    import traceback
    try:
        body = await request.body()
        print(f"[h3-decode] /decode {len(body)/1e6:.1f}MB", flush=True)
        data = load_bytes(body)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _POOL, _run_queued, run_decode, data.get("samples"), _WORLD)
        payload = dump_bytes(result)
        print(f"[h3-decode] 回傳 {len(payload)/1e6:.1f}MB", flush=True)
        return Response(content=payload, media_type="application/octet-stream")
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return Response(content=tb.encode(), status_code=500, media_type="text/plain")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-vae", required=True,
                        help="minimax_h3_video_vae_fp16.safetensors 路徑")
    parser.add_argument("--audio-vae", default="",
                        help="minimax_h3_audio_vae_fp32.safetensors 路徑")
    parser.add_argument("--dp", type=int, default=1,
                        help="decode 並行卡數（1=單卡，2/4=chunk DP）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()

    global _WORLD
    _WORLD = args.dp
    load_vae(args.video_vae, args.audio_vae or None)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
