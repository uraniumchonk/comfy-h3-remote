#!/usr/bin/env python3
"""
MiniMax H3 遠端 video decode server（跑在 192.168.0.160，卡 0）

接收 denoise 完成的 latent（AV NestedTensor 標記 dict，只取 video 半邊），
在本機 GPU 跑 video VAE（ViT3D 36 層 transformer decoder）decode。
回傳：
  frames: [N, H, W, C] fp32 [0,1]（ComfyUI IMAGE 標準格式，直送 VHS_VideoCombine）

audio VAE 留在 10 號機（工作流 VAEDecodeAudio 節點不變）。

數值行為對齊 160 執行樹 ComfyUI 0.33.0：
  - MiniMaxH3VideoVAE.decode 直接輸出 [0,1] fp32（_finalize_pixels）
10 號機（0.30.0）VAEDecode 的 wrapper 會把 [-1,1] 轉回 [0,1]，最終格式一致。

通訊協定同 h3_server.py：POST /decode，body = torch.save(bytes)。
"""
import argparse
import asyncio
import gc
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
from comfy.sd import VAE  # noqa: E402

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
DEVICE = torch.device("cuda:0")
PROGRESS = {"status": "idle", "elapsed": 0.0}


def load_vae(video_path):
    """載入 video VAE（一次），權重常駐 DEVICE。"""
    global VIDEO_VAE
    t0 = time.time()
    print(f"[h3-decode] 載入 video VAE: {video_path}", flush=True)
    sd, metadata = comfy.utils.load_torch_file(video_path, return_metadata=True)
    VIDEO_VAE = VAE(sd=sd, metadata=metadata)
    VIDEO_VAE.first_stage_model.to(DEVICE, dtype=torch.float16).eval()
    print(f"[h3-decode] video VAE 就緒 {time.time() - t0:.1f}s", flush=True)


def decode_video(z):
    """z: [B, 24, T, H, W] normalized latent -> [N, H', W', 3] fp32 [0,1]."""
    z = z.to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        # 0.33: decode_temporal 自建 CPU output_buffer，逐 chunk 串流寫出
        out = VIDEO_VAE.first_stage_model.decode(z)  # [B, 3, F, H', W'] fp32
    out = out.permute(0, 2, 3, 4, 1).contiguous()     # [B, F, H', W', 3]
    return out.reshape(-1, out.shape[-2], out.shape[-1], 3)


def run_decode(samples):
    """samples: NestedTensor(video, audio) / tensor / {"samples": ...}。

    只取 video 半邊（NestedTensor unbind()[0]），audio 留給 10 號機。
    """
    t0 = time.time()
    PROGRESS.update(status="running", elapsed=0.0)

    if isinstance(samples, dict) and "samples" in samples:
        samples = samples["samples"]

    video_z = None
    if isinstance(samples, comfy.nested_tensor.NestedTensor):
        ts = samples.tensors
        if ts:
            video_z = ts[0]
    elif isinstance(samples, torch.Tensor):
        video_z = samples

    if video_z is None:
        raise ValueError("no video latent in payload")

    tv = time.time()
    frames = decode_video(video_z)
    print(f"[h3-decode] video decode {time.time() - tv:.1f}s "
          f"frames={list(frames.shape)}", flush=True)

    elapsed = time.time() - t0
    PROGRESS.update(status="done", elapsed=elapsed)
    print(f"[h3-decode] 完成 {elapsed:.1f}s", flush=True)
    return {"frames": frames}


# ---------------------------------------------------------------------------
# HTTP 服務（FastAPI）
# ---------------------------------------------------------------------------

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import Response  # noqa: E402

app = FastAPI(title="MiniMax H3 Remote Video Decode")
_POOL = ThreadPoolExecutor(max_workers=1)


@app.get("/health")
def health():
    return {"status": "ok", "video_vae": VIDEO_VAE is not None}


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
        result = await loop.run_in_executor(_POOL, run_decode, data.get("samples"))
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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()

    load_vae(args.video_vae)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
