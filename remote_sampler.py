"""
MiniMax H3 Remote Denoise - Client-side custom node for ComfyUI.

Sends denoise jobs to the remote server (192.168.0.160:8299) and receives
the denoised latent back.

Wired between:
  MiniMaxH3ReferenceToVideo (or FL2VA) -> RemoteDenoiseNode -> VAE Decode (3D)
"""
import io
import json
import os
import struct
import threading
import time
import urllib.error
import urllib.request

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

    ensure_upstream(server_url)
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

    steps = int(data.get("steps") or 20)
    pbar = None
    bar = None
    try:
        from comfy.utils import ProgressBar
        pbar = ProgressBar(steps)
        pbar.update_absolute(0, steps)
    except Exception:
        pbar = None
    try:
        from tqdm import tqdm
        bar = tqdm(total=steps, desc="H3", unit="it", dynamic_ncols=True)
    except Exception:
        bar = None

    last = -1
    while th.is_alive():
        if mm.processing_interrupted():
            try:
                urllib.request.urlopen(
                    urllib.request.Request(f"{base}/interrupt", method="POST"),
                    timeout=5,
                )
            except Exception:
                pass
            if bar is not None:
                bar.close()
            mm.throw_exception_if_processing_interrupted()
        try:
            with urllib.request.urlopen(f"{base}/progress", timeout=2) as resp:
                info = json.loads(resp.read().decode())
            i = int(info.get("step") or 0)
            total = int(info.get("total") or steps)
            if i != last and i >= 0:
                if pbar is not None:
                    pbar.update_absolute(i, total)
                if bar is not None:
                    bar.total = total
                    bar.n = i
                    step_s = info.get("step_s") or 0
                    if step_s:
                        bar.set_postfix_str(f"{float(step_s):.1f}s/it")
                    bar.refresh()
                last = i
        except Exception:
            pass
        th.join(0.5)

    if bar is not None:
        if box["err"] is None:
            bar.n = bar.total
            bar.refresh()
        bar.close()
    if pbar is not None and box["err"] is None:
        pbar.update_absolute(pbar.total, pbar.total)

    if box["err"] is not None:
        raise box["err"]
    response_data = box["data"]
    elapsed = time.time() - t0
    print(f"[h3-client] Received {len(response_data)/1e6:.1f}MB in {elapsed:.1f}s", flush=True)
    return load_bytes(response_data)


# ---------------------------------------------------------------------------
# ComfyUI Nodes
# ---------------------------------------------------------------------------

# llama-swap /upstream/<id>：第一次請求會拉起服務。ttl=0 時 DiT 不會自己卸，
# ensure_upstream 在目標沒起來時先 unload 再等到 /health。
DEFAULT_DECODE_SERVER = "http://192.168.0.160:8090/upstream/minimax-h3-vae-decode-1"

_ASYNC_SWAP = (
    "minimax-h3-vae-decode-1",
    "minimax-h3-clip-encode",
    "minimax-h3-vae-decode",
)


def _upstream_id(server_url):
    u = server_url.rstrip("/")
    if "/upstream/" not in u:
        return None, None
    base, rest = u.split("/upstream/", 1)
    return base, rest.split("/")[0]


def ensure_upstream(server_url, timeout=300):
    """目標 /health 沒好：必要時卸 DiT，然後同步等到服務起來。"""
    import comfy.model_management as mm

    health = server_url.rstrip("/") + "/health"

    def ok(t=4):
        try:
            with urllib.request.urlopen(health, timeout=t) as r:
                return 200 <= r.status < 300
        except Exception:
            return False

    if ok():
        return

    base, target = _upstream_id(server_url)
    if base:
        names = []
        try:
            with urllib.request.urlopen(base + "/running", timeout=3) as r:
                data = json.loads(r.read().decode())
            names = [m.get("model", "") for m in data.get("running", [])]
        except Exception:
            names = []
        hold = [n for n in names if n in _ASYNC_SWAP and n != target]
        # DiT 跟 async 服務互斥：拉 DiT 一定先卸。拉 decode/encode 時若同伴已在就不要卸。
        must_unload = (target not in _ASYNC_SWAP) or (not hold)
        if must_unload:
            try:
                req = urllib.request.Request(
                    base + "/api/models/unload", data=b"", method="POST")
                urllib.request.urlopen(req, timeout=30).read()
                print(f"[h3-client] swap unload, waiting {target}", flush=True)
            except Exception as e:
                print(f"[h3-client] swap unload: {e}", flush=True)

    t0 = time.time()
    while time.time() - t0 < timeout:
        if mm.processing_interrupted():
            mm.throw_exception_if_processing_interrupted()
        if ok(8):
            print(f"[h3-client] upstream ready {target or health} "
                  f"{time.time()-t0:.1f}s", flush=True)
            return
        time.sleep(1)
    raise RuntimeError(f"upstream not ready: {health}")


def _upstream_loaded(server_url):
    base, target = _upstream_id(server_url)
    if not target:
        return True
    try:
        with urllib.request.urlopen(base + "/running", timeout=3) as r:
            data = json.loads(r.read().decode())
        names = [m.get("model", "") for m in data.get("running", [])]
        return target in names
    except Exception:
        return False


def _backend_slot(server_url, timeout=8, start=False):
    if not start and not _upstream_loaded(server_url):
        return "idle"
    health = server_url.rstrip("/") + "/health"
    with urllib.request.urlopen(health, timeout=timeout) as r:
        info = json.loads(r.read().decode())
    return info.get("slot") or "idle"


def _urlopen_slot(req, timeout, tag):
    """HTTP that retries backend 409 instead of failing."""
    import comfy.model_management as mm

    t0 = time.time()
    last = None
    while True:
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            last = e
            if e.code != 409:
                raise
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{tag} backend slot timeout") from last
        if mm.processing_interrupted():
            mm.throw_exception_if_processing_interrupted()
        print(f"[h3-client] {tag} backend busy, waiting {time.time()-t0:.0f}s",
              flush=True)
        time.sleep(0.5)


_PULLED = {}
_PULLED_LOCK = threading.Lock()
_PULLED_EVT = {}


def _pulled_has(tag):
    with _PULLED_LOCK:
        return tag in _PULLED


def _pulled_inflight(tag):
    ev = _PULLED_EVT.get(tag)
    return ev is not None and not ev.is_set()


def _wait_collected(tag, timeout=7200):
    """Don't start a new submit while the previous hold is still uncollected."""
    import comfy.model_management as mm

    t0 = time.time()
    while _pulled_has(tag) or _pulled_inflight(tag):
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{tag} previous hold not collected")
        if mm.processing_interrupted():
            mm.throw_exception_if_processing_interrupted()
        time.sleep(0.4)


def _pulled_reset(tag):
    with _PULLED_LOCK:
        _PULLED.pop(tag, None)
        ev = threading.Event()
        _PULLED_EVT[tag] = ev
        return ev


def _pulled_put(tag, value):
    with _PULLED_LOCK:
        _PULLED[tag] = value
        ev = _PULLED_EVT.get(tag)
    if ev is not None:
        ev.set()


def _pulled_pop(tag):
    with _PULLED_LOCK:
        return _PULLED.pop(tag, None)


def _pulled_wait(tag, timeout=7200):
    import comfy.model_management as mm

    ev = _PULLED_EVT.get(tag)
    t0 = time.time()
    while ev is None or not ev.is_set():
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{tag} prefetch timeout")
        if mm.processing_interrupted():
            mm.throw_exception_if_processing_interrupted()
        time.sleep(0.2)
        ev = _PULLED_EVT.get(tag)
    return _pulled_pop(tag)


def _prefetch(server_url, tag):
    try:
        raw = _backend_take(server_url, tag=tag, empty="error", start=False)
        _pulled_put(tag, raw)
        print(f"[h3-client] {tag} ready", flush=True)
    except Exception as e:
        _pulled_put(tag, e)
        print(f"[h3-client] {tag} prefetch error: {e}", flush=True)


def _backend_kick(server_url, path, data, timeout=7200, tag=""):
    """POST work to backend. Server holds the result. Prefetch pops in background."""
    import comfy.model_management as mm

    ensure_upstream(server_url)
    _wait_collected(tag, timeout=timeout)
    slot = _backend_slot(server_url, start=True)
    t0 = time.time()
    while slot == "running":
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{tag} backend still running")
        if mm.processing_interrupted():
            mm.throw_exception_if_processing_interrupted()
        print(f"[h3-client] {tag} backend running, wait kick {time.time()-t0:.0f}s",
              flush=True)
        time.sleep(0.5)
        slot = _backend_slot(server_url, start=True)
    _pulled_reset(tag)
    if slot == "hold":
        print(f"[h3-client] {tag} backend already hold, prefetch", flush=True)
        threading.Thread(target=_prefetch, args=(server_url, tag), daemon=True).start()
        return
    payload = dump_bytes(data)
    req = urllib.request.Request(
        server_url.rstrip("/") + path,
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    print(f"[h3-client] {tag}: sending {len(payload)/1e6:.1f}MB to {server_url}...",
          flush=True)
    box = {"err": None}

    def _post():
        try:
            with _urlopen_slot(req, timeout, tag) as resp:
                resp.read()
        except Exception as e:
            box["err"] = e

    th = threading.Thread(target=_post, daemon=True)
    th.start()
    while th.is_alive():
        if mm.processing_interrupted():
            mm.throw_exception_if_processing_interrupted()
        th.join(0.5)
    if box["err"] is not None:
        _pulled_put(tag, box["err"])
        raise box["err"]
    threading.Thread(target=_prefetch, args=(server_url, tag), daemon=True).start()


def _backend_take(server_url, timeout=7200, tag="", empty="error", start=False):
    """Pop the held result from backend. empty='error'|'none'."""
    import comfy.model_management as mm

    if start:
        ensure_upstream(server_url)
    elif not _upstream_loaded(server_url):
        if empty == "none":
            return None
        raise RuntimeError(f"{tag} mailbox empty")
    url = server_url.rstrip("/") + "/result"
    t0 = time.time()
    last = None
    while True:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=min(timeout, 600)) as resp:
                raw = resp.read()
            return raw
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:
                if empty == "none":
                    return None
                raise RuntimeError(f"{tag} mailbox empty") from e
            if e.code != 409:
                raise
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{tag} backend take timeout") from last
        if mm.processing_interrupted():
            mm.throw_exception_if_processing_interrupted()
        print(f"[h3-client] {tag} backend running, waiting take {time.time()-t0:.0f}s",
              flush=True)
        time.sleep(0.4)


def send_decode_request(data, server_url=DEFAULT_DECODE_SERVER, timeout=7200):
    _backend_kick(server_url, "/decode", data, timeout=timeout, tag="decode")
    raw = _pulled_wait("decode", timeout=timeout)
    if isinstance(raw, Exception):
        raise raw
    if raw is None:
        raise RuntimeError("decode mailbox empty")
    return load_bytes(raw)


async def send_decode_request_async(data, server_url=DEFAULT_DECODE_SERVER, timeout=7200):
    """async 版：await 期間 ComfyUI 可跑其他不依賴此輸出的 node。"""
    import asyncio
    import comfy.model_management as mm

    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, send_decode_request, data, server_url, timeout)
    while not fut.done():
        if mm.processing_interrupted():
            mm.throw_exception_if_processing_interrupted()
        await asyncio.sleep(0.2)
    return await fut


def _audio_or_false(audio):
    if not isinstance(audio, dict):
        return False
    w = audio.get("waveform")
    if w is None or not hasattr(w, "numel") or int(w.numel()) <= 2:
        return False
    return audio


def _unpack_decode(result):
    frames = result.get("frames")
    if frames is None:
        raise RuntimeError("decode server returned no frames")
    return (frames, _audio_or_false(result.get("audio")))


class RemoteDecodeNode:
    """
    Denoise 完成的 latent 送到 160 卡 0 做 video VAE decode，
    回傳 ComfyUI IMAGE（[N,H,W,C] fp32 [0,1]）直送 VHS_VideoCombine。

    async：等待遠端期間 10 號機可跑其他不依賴 IMAGE/AUDIO 的 node。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {}),
                "server_url": ("STRING", {
                    "default": DEFAULT_DECODE_SERVER,
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "decode"
    CATEGORY = "Remote Pipe"

    async def decode(self, samples, server_url):
        result = await send_decode_request_async({"samples": samples}, server_url)
        return _unpack_decode(result)


class RemoteDecodeSubmit:
    """丟 160 做 AV decode。trigger 輸出是進站 latent 原樣通透，只當執行順序。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {}),
                "server_url": ("STRING", {
                    "default": DEFAULT_DECODE_SERVER,
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "submit"
    OUTPUT_NODE = True
    CATEGORY = "Remote Pipe"

    @classmethod
    def IS_CHANGED(cls, *values, **kwargs):
        return float("NaN")

    def submit(self, samples, server_url):
        def _work():
            try:
                _backend_kick(server_url, "/decode", {"samples": samples},
                               tag="decode")
            except Exception as e:
                print(f"[h3-client] decode kick error: {e}", flush=True)

        threading.Thread(target=_work, daemon=True).start()
        print("[h3-client] decode submit", flush=True)
        return (samples,)


class RemoteDecodeCollect:
    """拿目前跑完的那一包（上一輪 Submit）。後端還在跑就堵住；空才吐 dummy。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*", {}),
            },
            "optional": {
                "server_url": ("STRING", {
                    "default": DEFAULT_DECODE_SERVER,
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "collect"
    CATEGORY = "Remote Pipe"

    @classmethod
    def IS_CHANGED(cls, *values, **kwargs):
        return float("NaN")

    def collect(self, trigger, server_url=DEFAULT_DECODE_SERVER):
        if _pulled_inflight("decode"):
            raw = _pulled_wait("decode")
        else:
            raw = _pulled_pop("decode")
        if raw is None:
            try:
                slot = _backend_slot(server_url, start=False)
            except Exception:
                slot = "idle"
            if slot == "hold":
                raw = _backend_take(server_url, tag="decode", empty="none",
                                    start=False)
            else:
                print("[h3-client] decode not ready, skip collect", flush=True)
                return (torch.zeros(1, 8, 8, 3), False)
        if isinstance(raw, Exception):
            print(f"[h3-client] decode prefetch error, skip: {raw}", flush=True)
            return (torch.zeros(1, 8, 8, 3), False)
        if raw is None:
            print("[h3-client] decode not ready, skip collect", flush=True)
            return (torch.zeros(1, 8, 8, 3), False)
        packed = load_bytes(raw)
        if isinstance(packed, dict) and "frames" in packed:
            frames, audio = _unpack_decode(packed)
        else:
            frames, audio = packed, False
        audio = _audio_or_false(audio)
        print(f"[h3-client] decode collect {list(frames.shape)} "
              f"audio={'ok' if audio is not False else 'false'}", flush=True)
        return (frames, audio)


class RemoteDecodeGet(RemoteDecodeCollect):
    """舊名：跟 Collect 同一顆（信箱 pop），不再吃 RAM job handle。"""
    pass


DEFAULT_ENCODE_SERVER = "http://192.168.0.160:8090/upstream/minimax-h3-clip-encode"


def send_encode_request(data, server_url=DEFAULT_ENCODE_SERVER, timeout=7200):
    _backend_kick(server_url, "/encode", data, timeout=timeout, tag="encode")
    raw = _pulled_wait("encode", timeout=timeout)
    if isinstance(raw, Exception):
        raise raw
    if raw is None:
        raise RuntimeError("encode mailbox empty")
    return load_bytes(raw)


def _pack_encode_payload(prompt, width, height, length, ref_image_size,
                         ref_image=None, ref_video=None,
                         ref_video_audio=None, ref_audio=None):
    data = {
        "prompt": prompt,
        "width": int(width),
        "height": int(height),
        "length": int(length),
        "ref_image_size": ref_image_size,
    }
    if ref_image is not None:
        data["ref_images"] = {"ref_image_0": ref_image}
    if ref_video is not None:
        data["ref_videos"] = {"ref_video_0": ref_video}
    if ref_video_audio is not None:
        data["ref_video_audios"] = {"ref_video_audio_0": ref_video_audio}
    if ref_audio is not None:
        data["ref_audios"] = {"ref_audio_0": ref_audio}
    return data


class RemoteEncodeNode:
    """同步：官方 MiniMaxH3ReferenceToVideo 整段 encode（CLIP + ref VAE）丟 160。

    回傳 positive + latent，後面直接接 RemoteDenoise。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 4096, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 1}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "server_url": ("STRING", {
                    "default": DEFAULT_ENCODE_SERVER,
                    "multiline": False,
                }),
            },
            "optional": {
                "ref_image": ("IMAGE",),
                "ref_video": ("IMAGE",),
                "ref_video_audio": ("AUDIO",),
                "ref_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    CATEGORY = "Remote Pipe"

    async def encode(self, prompt, width, height, length, ref_image_size, server_url,
                     ref_image=None, ref_video=None, ref_video_audio=None, ref_audio=None):
        import asyncio
        import comfy.model_management as mm

        data = _pack_encode_payload(
            prompt, width, height, length, ref_image_size,
            ref_image, ref_video, ref_video_audio, ref_audio)
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, send_encode_request, data, server_url)
        while not fut.done():
            if mm.processing_interrupted():
                mm.throw_exception_if_processing_interrupted()
            await asyncio.sleep(0.2)
        result = await fut
        return (result["positive"], result["latent"])


class RemoteEncodeSubmit:
    """跨圖信箱：立刻丟 encode，不等結果。下一輪 Collect 再拿 cond。"""

    @classmethod
    def INPUT_TYPES(cls):
        types = RemoteEncodeNode.INPUT_TYPES()
        types.setdefault("optional", {})
        types["optional"]["trigger"] = ("LATENT",)
        types["optional"]["latent"] = ("LATENT",)
        return types

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "submit"
    OUTPUT_NODE = True
    CATEGORY = "Remote Pipe"

    @classmethod
    def IS_CHANGED(cls, *values, **kwargs):
        return float("NaN")

    def submit(self, prompt, width, height, length, ref_image_size, server_url,
               ref_image=None, ref_video=None, ref_video_audio=None, ref_audio=None,
               trigger=None, latent=None):
        packed = _pack_encode_payload(
            prompt, width, height, length, ref_image_size,
            ref_image, ref_video, ref_video_audio, ref_audio)

        def _work():
            try:
                _backend_kick(server_url, "/encode", packed, tag="encode")
            except Exception as e:
                print(f"[h3-client] encode kick error: {e}", flush=True)

        threading.Thread(target=_work, daemon=True).start()
        print("[h3-client] encode submit", flush=True)
        return (latent if latent is not None else trigger,)


class RemoteEncodeCollect:
    """拿目前跑完的那一包 encode（上一輪 Submit）。還在跑就堵；空立刻報錯。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "server_url": ("STRING", {
                    "default": DEFAULT_ENCODE_SERVER,
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "collect"
    CATEGORY = "Remote Pipe"

    @classmethod
    def IS_CHANGED(cls, *values, **kwargs):
        return float("NaN")

    def collect(self, server_url=DEFAULT_ENCODE_SERVER):
        raw = _pulled_pop("encode")
        if raw is None:
            ev = _PULLED_EVT.get("encode")
            if ev is not None and not ev.is_set():
                raw = _pulled_wait("encode")
            else:
                raw = _backend_take(server_url, tag="encode", empty="error",
                                    start=False)
        if isinstance(raw, Exception):
            raise raw
        if raw is None:
            raise RuntimeError("encode mailbox empty")
        result = load_bytes(raw)
        print("[h3-client] encode collect", flush=True)
        return (result["positive"], result["latent"])


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
    CATEGORY = "Remote Pipe"
    
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
                "positive": ("CONDITIONING", {}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Official H3 is 1.0 (flow-matching, no CFG). >1 is allowed; snow/black is on you.",
                }),
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
                "latent_image": ("LATENT", {}),
                "lora_stack": ("H3_LORA_STACK",),
                "lora_name": (_LORA_LIST, {"default": "None"}),
                "lora_strength": ("FLOAT", {"default": 0.9, "min": -2.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "denoise"
    CATEGORY = "Remote Pipe"

    def denoise(self, positive, steps, cfg, sampler_name, scheduler,
                seed, denoise, shift_video, shift_audio, server_url,
                negative=None, latent_image=None, lora_stack=None,
                lora_name="None", lora_strength=0.9):
        if latent_image is None:
            raise ValueError("latent_image is required")
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
    CATEGORY = "Remote Pipe"

    def stack(self, lora_name, strength, enabled=True, lora_stack=None):
        out = list(lora_stack or [])
        if enabled and lora_name and lora_name != "None":
            out.append({"name": lora_name, "strength": float(strength)})
        return (out,)