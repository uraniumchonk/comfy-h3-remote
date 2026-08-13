"""
MiniMax H3 Remote Denoise - Client-side custom node for ComfyUI.

Sends denoise jobs to the remote server (192.168.0.160:8299) and receives
the denoised latent back.

Wired between:
  MiniMaxH3ReferenceToVideo (or FL2VA) -> RemoteDenoiseNode -> VAE Decode (3D)
"""
import io
import json
import struct
import time
import urllib.request
import urllib.error

import torch

try:
    import folder_paths
    _LORA_LIST = ["None"] + list(folder_paths.get_filename_list("loras"))
except Exception:
    _LORA_LIST = ["None"]

try:
    import comfy.samplers as _cs
    _SAMPLERS = list(_cs.KSampler.SAMPLERS)
    _SCHEDULERS = list(_cs.KSampler.SCHEDULERS)
except Exception:
    _SAMPLERS = [
        "euler", "res_multistep", "euler_ancestral", "dpmpp_2m",
        "dpmpp_2s_ancestral", "dpmpp_sde", "ddim", "lms", "heun", "uni_pc",
    ]
    _SCHEDULERS = ["simple", "normal", "karras", "exponential", "sgm_uniform", "beta"]

# ---------------------------------------------------------------------------
# Serialization helpers (matching h3_server.py)
# ---------------------------------------------------------------------------

def _serialize(obj):
    """Serialize torch.Tensor and NestedTensor to plain types."""
    if isinstance(obj, torch.Tensor):
        return obj
    # NestedTensor detection by attribute (avoid hard import)
    if hasattr(obj, "tensors") and hasattr(obj, "__class__") and obj.__class__.__name__ == "NestedTensor":
        return {"__h3_nt__": True, "tensors": [_serialize(t) for t in obj.tensors]}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def _deserialize(obj):
    """Deserialize back to torch.Tensor / NestedTensor."""
    if isinstance(obj, dict):
        if obj.get("__h3_nt__"):
            from comfy.nested_tensor import NestedTensor
            return NestedTensor([_deserialize(t) for t in obj["tensors"]])
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
# HTTP client
# ---------------------------------------------------------------------------

# llama-swap /upstream/<id>：第一次請求會卸載 vLLM、拉起 H3
DEFAULT_SERVER = "http://192.168.0.160:8090/upstream/minimax-h3-ref2va"


def send_denoise_request(data, server_url=DEFAULT_SERVER, timeout=7200):
    """Send denoise request. Frontend Cancel posts /interrupt so the GPU box stops."""
    import threading
    import comfy.model_management as mm

    base = server_url.rstrip("/")
    health = f"{base}/health"
    print(f"[h3-client] waking {health}", flush=True)
    with urllib.request.urlopen(health, timeout=timeout) as resp:
        resp.read()

    payload = dump_bytes(data)
    req = urllib.request.Request(
        f"{base}/denoise",
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    t0 = time.time()
    print(f"[h3-client] Sending {len(payload)/1e6:.1f}MB to {server_url}...", flush=True)

    box = {"data": None, "err": None}

    def _post():
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                box["data"] = resp.read()
        except Exception as e:
            box["err"] = e

    th = threading.Thread(target=_post, daemon=True)
    th.start()
    while th.is_alive():
        if mm.processing_interrupted():
            try:
                urllib.request.urlopen(
                    urllib.request.Request(f"{base}/interrupt", method="POST"),
                    timeout=5,
                )
            except Exception:
                pass
            mm.throw_exception_if_processing_interrupted()
        th.join(0.2)

    if box["err"] is not None:
        raise box["err"]
    response_data = box["data"]
    elapsed = time.time() - t0
    print(f"[h3-client] Received {len(response_data)/1e6:.1f}MB in {elapsed:.1f}s", flush=True)
    return load_bytes(response_data)


# ---------------------------------------------------------------------------
# ComfyUI Nodes
# ---------------------------------------------------------------------------

class RemoteDenoiseSampler:
    """
    Replaces SamplerCustomAdvanced. Takes the same inputs but sends
    the denoise job to the remote server.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE", {}),
                "guider": ("GUIDER", {}),
                "sampler": ("SAMPLER", {}),
                "sigmas": ("SIGMAS", {}),
                "latent_image": ("LATENT", {}),
                "server_url": ("STRING", {
                    "default": DEFAULT_SERVER,
                    "multiline": False,
                }),
            },
        }
    
    RETURN_TYPES = ("LATENT", "LATENT")
    FUNCTION = "sample"
    CATEGORY = "MiniMax H3"
    
    def sample(self, noise, guider, sampler, sigmas, latent_image, server_url):
        # Extract model from guider
        model = guider.model
        
        # Extract conditioning from guider
        if hasattr(guider, "positive"):
            positive = guider.positive
            negative = getattr(guider, "negative", [])
            cfg = getattr(guider, "cfg", 1.0)
        else:
            # BasicGuider only has positive
            positive = guider.conditioning
            negative = []
            cfg = 1.0
        
        # Get sampler name
        sampler_name = sampler.name if hasattr(sampler, "name") else str(sampler.__class__.__name__)
        
        # Calculate steps from sigmas
        steps = len(sigmas) - 1 if hasattr(sigmas, "__len__") else 20
        
        # Get seed
        seed = int(noise.get("seed", 0)) if isinstance(noise, dict) else 0
        
        # Build request
        data = {
            "latent_image": latent_image,
            "positive": positive,
            "negative": negative,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": "normal",
            "denoise": 1.0,
            "seed": seed,
            "shift_video": 12.0,
            "shift_audio": 3.0,
            "disable_noise": False,
        }
        
        # Send to remote
        result = send_denoise_request(data, server_url)
        
        return ({"samples": result["samples"]}, {"samples": result["samples"]})


class RemoteDenoiseNode:
    """
    10 號機只跑 CLIP / VAE / Ref2VA 編碼。
    Ref2VA DiT 在 160 經 llama-swap 載入，這裡不吃 MODEL。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_image": ("LATENT", {}),
                "positive": ("CONDITIONING", {}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (_SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "server_url": ("STRING", {
                    "default": DEFAULT_SERVER,
                    "multiline": False,
                }),
            },
            "optional": {
                "negative": ("CONDITIONING", {}),
                "lora_stack": ("H3_LORA_STACK",),
                "lora_name": (_LORA_LIST, {"default": "None"}),
                "lora_strength": ("FLOAT", {"default": 0.9, "min": -2.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "denoise"
    CATEGORY = "MiniMax H3"

    def denoise(self, latent_image, positive, steps, cfg, sampler_name, scheduler,
                seed, denoise, shift_video, shift_audio, server_url, negative=None,
                lora_stack=None, lora_name="None", lora_strength=0.9):
        data = {
            "latent_image": latent_image,
            "positive": positive,
            "negative": negative if negative is not None else [],
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
            "seed": seed,
            "shift_video": shift_video,
            "shift_audio": shift_audio,
            "disable_noise": False,
        }
        loras = list(lora_stack or [])
        if lora_name and lora_name != "None":
            loras.append({"name": lora_name, "strength": float(lora_strength)})
        if loras:
            data["loras"] = loras
        result = send_denoise_request(data, server_url)
        return ({"samples": result["samples"]},)


class H3LoraStack:
    """Chain like official LoraLoader: Stack -> Stack -> RemoteDenoise.

    Each node appends one (name, strength) if enabled. Output is a plain
    list; the denoise node sends it as data['loras'].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (_LORA_LIST, {"default": "None"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lora_stack": ("H3_LORA_STACK",),
            },
        }

    RETURN_TYPES = ("H3_LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "stack"
    CATEGORY = "MiniMax H3"

    def stack(self, lora_name, strength, enabled=True, lora_stack=None):
        out = list(lora_stack or [])
        if enabled and lora_name and lora_name != "None":
            out.append({"name": lora_name, "strength": float(strength)})
        return (out,)