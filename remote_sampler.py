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
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

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

# video decode 服務（160 卡 0，llama-swap 管理）。audio decode 預留：目前留在 10 號機本機。
# llama-swap /upstream/<id>：第一次請求會卸載當前模型、拉起 decode 服務
DEFAULT_DECODE_SERVER = "http://192.168.0.160:8090/upstream/minimax-h3-vae-decode"

_MAILBOX_LOCK = threading.Lock()


def _mailbox_paths():
    try:
        import folder_paths
        root = folder_paths.get_output_directory()
    except Exception:
        root = os.path.join(os.path.expanduser("~"), "h3_decode_mailbox")
    jobs_dir = os.path.join(root, "h3_decode_jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    return os.path.join(root, "h3_decode_mailbox.json"), jobs_dir


def _mailbox_load():
    path, _ = _mailbox_paths()
    if not os.path.isfile(path):
        return {"jobs": []}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "jobs" not in data:
        return {"jobs": []}
    return data


def _mailbox_save(data):
    path, _ = _mailbox_paths()
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _mailbox_update(job_id, **fields):
    with _MAILBOX_LOCK:
        data = _mailbox_load()
        for job in data["jobs"]:
            if job.get("id") == job_id:
                job.update(fields)
                break
        _mailbox_save(data)


def send_decode_request(data, server_url=DEFAULT_DECODE_SERVER, timeout=7200):
    """Send decode request. No step progress (decode is single-shot)."""
    import threading
    import comfy.model_management as mm

    base = server_url.rstrip("/")
    payload = dump_bytes(data)
    req = urllib.request.Request(
        f"{base}/decode",
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    t0 = time.time()
    print(f"[h3-client] decode: sending {len(payload)/1e6:.1f}MB to {server_url}...",
          flush=True)

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
            mm.throw_exception_if_processing_interrupted()
        th.join(0.5)

    if box["err"] is not None:
        raise box["err"]
    elapsed = time.time() - t0
    print(f"[h3-client] decode: received {len(box['data'])/1e6:.1f}MB in {elapsed:.1f}s",
          flush=True)
    return load_bytes(box["data"])


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


def _unpack_decode(result):
    frames = result.get("frames")
    audio = result.get("audio")
    if frames is None:
        raise RuntimeError("decode server returned no frames")
    return (frames, audio)


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
    """DVB 信箱：denoise 完立刻丟 160。沒有輸出，避免跟這一輪 latent 搞混。"""

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

    RETURN_TYPES = ()
    FUNCTION = "submit"
    OUTPUT_NODE = True
    CATEGORY = "Remote Pipe"

    @classmethod
    def IS_CHANGED(cls, *values, **kwargs):
        return float("NaN")

    def submit(self, samples, server_url):
        job_id = uuid.uuid4().hex
        _, jobs_dir = _mailbox_paths()
        frames_path = os.path.join(jobs_dir, job_id + ".pt")
        rec = {
            "id": job_id,
            "status": "running",
            "server_url": server_url,
            "frames_path": frames_path,
            "error": None,
            "created": time.time(),
        }
        with _MAILBOX_LOCK:
            data = _mailbox_load()
            data["jobs"].append(rec)
            _mailbox_save(data)

        def _work():
            try:
                result = send_decode_request({"samples": samples}, server_url)
                if result.get("frames") is None:
                    raise RuntimeError("decode server returned no frames")
                torch.save(_serialize(result), frames_path)
                _mailbox_update(job_id, status="ready")
                print(f"[h3-client] mailbox ready {job_id}", flush=True)
            except Exception as e:
                _mailbox_update(job_id, status="error", error=str(e))
                print(f"[h3-client] mailbox error {job_id}: {e}", flush=True)

        threading.Thread(target=_work, daemon=True).start()
        print(f"[h3-client] mailbox submit {job_id}", flush=True)
        return {}


class RemoteDecodeCollect:
    """只 pop 已經 ready 的上一輪畫面。trigger 只決定執行順序，不是這一輪 latent。

    不要等 running：Submit 跟 Collect 都掛在 denoise 後，等 running 會變成收這一輪。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*", {}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "collect"
    CATEGORY = "Remote Pipe"

    @classmethod
    def IS_CHANGED(cls, *values, **kwargs):
        return float("NaN")

    def collect(self, trigger):
        with _MAILBOX_LOCK:
            data = _mailbox_load()
            job = next((j for j in data["jobs"] if j.get("status") == "ready"), None)

        silent = {"waveform": torch.zeros(1, 2, 1), "sample_rate": 32000}
        if job is None:
            print("[h3-client] decode mailbox empty, skip collect", flush=True)
            return (torch.zeros(1, 8, 8, 3), silent)

        path = job.get("frames_path")
        if not path or not os.path.isfile(path):
            _mailbox_update(job["id"], status="error", error="frames file missing")
            raise RuntimeError(f"mailbox job {job['id']} frames missing: {path}")

        packed = torch.load(path, weights_only=False)
        with _MAILBOX_LOCK:
            data = _mailbox_load()
            data["jobs"] = [j for j in data["jobs"] if j.get("id") != job["id"]]
            _mailbox_save(data)
        try:
            os.remove(path)
        except OSError:
            pass

        if isinstance(packed, dict) and "frames" in packed:
            result = _deserialize(packed)
            frames, audio = _unpack_decode(result)
        else:
            frames, audio = packed, silent
        if audio is None:
            audio = silent
        print(f"[h3-client] mailbox collect {job['id']} {list(frames.shape)}", flush=True)
        return (frames, audio)


class RemoteDecodeGet(RemoteDecodeCollect):
    """舊名：跟 Collect 同一顆（信箱 pop），不再吃 RAM job handle。"""
    pass


DEFAULT_ENCODE_SERVER = "http://192.168.0.160:8090/upstream/minimax-h3-clip-encode"
_ENCODE_MAILBOX = "h3_encode_mailbox.json"
_ENCODE_JOBS_DIR = "h3_encode_jobs"


def _encode_mailbox_paths():
    try:
        import folder_paths
        root = folder_paths.get_output_directory()
    except Exception:
        root = os.path.join(os.path.expanduser("~"), "h3_encode_mailbox")
    jobs_dir = os.path.join(root, _ENCODE_JOBS_DIR)
    os.makedirs(jobs_dir, exist_ok=True)
    return os.path.join(root, _ENCODE_MAILBOX), jobs_dir


def _encode_box_load():
    path, _ = _encode_mailbox_paths()
    if not os.path.isfile(path):
        return {"jobs": []}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "jobs" not in data:
        return {"jobs": []}
    return data


def _encode_box_save(data):
    path, _ = _encode_mailbox_paths()
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _encode_box_update(job_id, **fields):
    with _MAILBOX_LOCK:
        data = _encode_box_load()
        for job in data["jobs"]:
            if job.get("id") == job_id:
                job.update(fields)
                break
        _encode_box_save(data)


def send_encode_request(data, server_url=DEFAULT_ENCODE_SERVER, timeout=7200):
    import comfy.model_management as mm

    base = server_url.rstrip("/")
    payload = dump_bytes(data)
    req = urllib.request.Request(
        f"{base}/encode",
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    t0 = time.time()
    print(f"[h3-client] encode: sending {len(payload)/1e6:.1f}MB to {server_url}...",
          flush=True)
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
            mm.throw_exception_if_processing_interrupted()
        th.join(0.5)
    if box["err"] is not None:
        raise box["err"]
    print(f"[h3-client] encode: received {len(box['data'])/1e6:.1f}MB in {time.time()-t0:.1f}s",
          flush=True)
    return load_bytes(box["data"])


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
        return types

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("queued",)
    FUNCTION = "submit"
    OUTPUT_NODE = True
    CATEGORY = "Remote Pipe"

    def submit(self, prompt, width, height, length, ref_image_size, server_url,
               ref_image=None, ref_video=None, ref_video_audio=None, ref_audio=None,
               trigger=None):
        job_id = uuid.uuid4().hex
        _, jobs_dir = _encode_mailbox_paths()
        payload_path = os.path.join(jobs_dir, job_id + ".pt")
        rec = {
            "id": job_id,
            "status": "running",
            "server_url": server_url,
            "result_path": payload_path,
            "error": None,
            "created": time.time(),
        }
        with _MAILBOX_LOCK:
            data = _encode_box_load()
            data["jobs"].append(rec)
            _encode_box_save(data)

        packed = _pack_encode_payload(
            prompt, width, height, length, ref_image_size,
            ref_image, ref_video, ref_video_audio, ref_audio)

        def _work():
            try:
                result = send_encode_request(packed, server_url)
                torch.save(_serialize(result), payload_path)
                _encode_box_update(job_id, status="ready")
                print(f"[h3-client] encode mailbox ready {job_id}", flush=True)
            except Exception as e:
                _encode_box_update(job_id, status="error", error=str(e))
                print(f"[h3-client] encode mailbox error {job_id}: {e}", flush=True)

        threading.Thread(target=_work, daemon=True).start()
        print(f"[h3-client] encode mailbox submit {job_id}", flush=True)
        return (1,)


def _mailbox_wait(load_fn, kind, timeout=7200):
    """ready 就回傳；有 running 就堵住等；真的沒上一輪才回 None。"""
    import comfy.model_management as mm

    t0 = time.time()
    while True:
        with _MAILBOX_LOCK:
            data = load_fn()
            jobs = data.get("jobs") or []
            ready = next((j for j in jobs if j.get("status") == "ready"), None)
            running = any(j.get("status") == "running" for j in jobs)
            failed = next((j for j in jobs if j.get("status") == "error"), None)
        if ready is not None:
            return ready
        if failed is not None and not running:
            raise RuntimeError(f"{kind} mailbox error: {failed.get('error')}")
        if not running:
            return None
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{kind} mailbox wait timeout ({timeout}s)")
        if mm.processing_interrupted():
            mm.throw_exception_if_processing_interrupted()
        print(f"[h3-client] {kind} mailbox running, waiting {time.time()-t0:.0f}s",
              flush=True)
        time.sleep(0.4)


class RemoteEncodeCollect:
    """跨圖信箱：拿上一輪 encode。還在跑就等；信箱真的空才報錯。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "collect"
    CATEGORY = "Remote Pipe"

    @classmethod
    def IS_CHANGED(cls, *values, **kwargs):
        return float("NaN")

    def collect(self):
        job = _mailbox_wait(_encode_box_load, "encode")
        if job is None:
            raise RuntimeError(
                "encode mailbox 沒有上一輪。先跑 h3_async_prefill.json 再 Queue 這張。"
            )
        path = job.get("result_path")
        if not path or not os.path.isfile(path):
            _encode_box_update(job["id"], status="error", error="result missing")
            raise RuntimeError(f"encode job {job['id']} result missing")
        result = _deserialize(torch.load(path, weights_only=False))
        _encode_box_update(job["id"], status="collected")
        try:
            os.remove(path)
        except OSError:
            pass
        print(f"[h3-client] encode mailbox collect {job['id']}", flush=True)
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