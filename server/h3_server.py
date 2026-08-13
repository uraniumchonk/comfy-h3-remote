#!/usr/bin/env python3
"""
MiniMax H3 Ref2VA 遠端 denoise server（跑在 192.168.0.160）

接收 ComfyUI 端（192.168.0.10）序列化過來的
  latent_image（乾淨 AV latent）、positive/negative conditioning、採樣參數
在本機 GPU 上跑完整 denoise（comfy.sample.sample），回傳結果 latent。

通訊協定：HTTP POST /denoise，body = torch.save(bytes)
序列化時 NestedTensor 轉成 {"__h3_nt__": True, "tensors": [...]} 標記 dict，
避免跨 ComfyUI 版本 pickle class 引用問題。
"""
import argparse
import gc
import io
import os
import sys
import time

# Set device before importing torch to avoid CUDA init issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

import torch

# Same dir: h3_tp.py. COMFYUI_ROOT: ComfyUI checkout (0.30+ MiniMax H3 + kitchen).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_comfy = os.environ.get("COMFYUI_ROOT") or os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
if _comfy:
    sys.path.insert(0, _comfy)

import comfy  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.model_sampling  # noqa: E402
import comfy.model_management  # noqa: E402
from comfy.sd import load_diffusion_model  # noqa: E402

# ---------------------------------------------------------------------------
# 序列化協定（與 ComfyUI 端 custom node 的 serializer 對應）
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
# 模型載入與採樣
# ---------------------------------------------------------------------------

MODEL = None          # base ModelPatcher（未 clone）
MODEL_PATH = None
LOADED_SHIFT_V = 12.0
LOADED_SHIFT_A = 3.0


def load_model(path, tensor_parallel=False):
    """載入 Ref2VA DiT（一次）。"""
    global MODEL, MODEL_PATH
    t0 = time.time()
    print(f"[h3-server] 載入 diffusion model: {path}", flush=True)
    model_options = {}
    if tensor_parallel:
        # 整包 20GB 先待在 CPU，切完 shard 再把殘件搬去 cuda:0
        model_options = {
            "load_device": torch.device("cpu"),
            "offload_device": torch.device("cpu"),
        }
    MODEL = load_diffusion_model(path, model_options=model_options)
    MODEL_PATH = path
    print(f"[h3-server] 載入完成，耗時 {time.time() - t0:.1f}s", flush=True)
    if tensor_parallel:
        from h3_tp import apply_tp
        dit = MODEL.model.diffusion_model
        world = apply_tp(dit)
        dit.to("cuda:0")
        MODEL.load_device = torch.device("cuda:0")
        MODEL.offload_device = torch.device("cuda:0")
        print(f"[h3-server] TP{world} applied, blocks={len(dit.blocks)}", flush=True)
    # 預設 shifts 從 model config sampling_settings 讀
    global LOADED_SHIFT_V, LOADED_SHIFT_A
    try:
        ss = MODEL.model.model_config.sampling_settings or {}
        LOADED_SHIFT_V = float(ss.get("shift", 12.0))
        LOADED_SHIFT_A = float(ss.get("audio_shift", 3.0))
    except Exception:
        pass
    print(f"[h3-server] 預設 shifts: video={LOADED_SHIFT_V} audio={LOADED_SHIFT_A}", flush=True)


def patch_sampling(model_patcher, shift_v, shift_a):
    """複製 ComfyUI MiniMaxH3SigmaShift 節點的邏輯。

    model_sampling 換成 ModelSamplingAV + CONST，並把 shifts 寫進
    transformer_options 供 DiT forward 讀取。
    """
    m = model_patcher.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV,
                                comfy.model_sampling.CONST):
        pass

    original = m.get_model_object("model_sampling")
    ms = ModelSamplingAdvanced(m.model.model_config)
    ms.set_parameters(shift=shift_v, audio_shift=shift_a)
    if hasattr(original, "noise_scale"):
        ms.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", ms)

    to = m.model_options["transformer_options"] = \
        m.model_options.get("transformer_options", {}).copy()
    to["minimax_h3_sigma_shift_video"] = shift_v
    to["minimax_h3_sigma_shift_audio"] = shift_a
    return m


def run_denoise(data):
    """執行一次完整採樣，回傳結果 dict（含 samples）。"""
    latent_image = data["latent_image"]
    if isinstance(latent_image, dict):
        latent_image = latent_image["samples"]
    positive = data["positive"]
    negative = data.get("negative") or []
    steps = int(data["steps"])
    cfg = float(data["cfg"])
    sampler_name = data["sampler_name"]
    scheduler = data["scheduler"]
    denoise = float(data.get("denoise", 1.0))
    seed = int(data.get("seed", 0))
    shift_v = float(data.get("shift_video", LOADED_SHIFT_V))
    shift_a = float(data.get("shift_audio", LOADED_SHIFT_A))
    disable_noise = bool(data.get("disable_noise", False))

    model = patch_sampling(MODEL, shift_v, shift_a)

    t0 = time.time()
    if disable_noise:
        noise = comfy.sample.prepare_empty_noise(latent_image)
    else:
        noise = comfy.sample.prepare_noise(latent_image, seed)

    def _step_cb(*_a, **_k):
        torch.cuda.empty_cache()

    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, latent_image,
        denoise=denoise, seed=seed, callback=_step_cb,
    )
    elapsed = time.time() - t0
    print(f"[h3-server] denoise 完成: steps={steps} sampler={sampler_name} "
          f"scheduler={scheduler} cfg={cfg} 耗時 {elapsed:.1f}s "
          f"({elapsed / max(steps,1):.2f}s/step)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"samples": samples}


# ---------------------------------------------------------------------------
# HTTP 服務（FastAPI）
# ---------------------------------------------------------------------------

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import Response  # noqa: E402

app = FastAPI(title="MiniMax H3 Ref2VA Remote Denoise")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH,
            "shift_video": LOADED_SHIFT_V, "shift_audio": LOADED_SHIFT_A}


@app.post("/denoise")
async def denoise_endpoint(request: Request):
    import traceback
    try:
        body = await request.body()
        print(f"[h3-server] /denoise {len(body)/1e6:.1f}MB", flush=True)
        data = load_bytes(body)
        result = run_denoise(data)
        return Response(content=dump_bytes(result),
                        media_type="application/octet-stream")
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return Response(content=tb.encode(), status_code=500, media_type="text/plain")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Ref2VA GGUF 或 INT8 safetensors 路徑")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8299)
    parser.add_argument("--tp", action="store_true",
                        help="2-GPU tensor parallel for INT8+ConvRot DiT")
    args = parser.parse_args()

    load_model(args.model, tensor_parallel=args.tp)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
