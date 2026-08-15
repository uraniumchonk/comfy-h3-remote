# Remote Pipe (MiniMax H3)

[繁體中文](README.zh-TW.md)

ComfyUI nodes that offload MiniMax H3 heavy work to a GPU box. Two independent scenes.

The point of Scene B is that **encoder and decoder time go to ~0**. Denoise is the only long step; CLIP and VAE hide behind the next / previous queue. Measured: 0.4MP / 10s / 8-step LoRA — **91s denoise, 95.90s full queue** (~4s ffmpeg).

| | Scene A | Scene B |
|---|---|---|
| Who denoises | Remote DiT (TP2) | **Local** Sampler / UNET |
| Remote runs | Denoise only | **CLIP encode + AV VAE decode** |
| How you queue | One graph, wait | Cross-queue: sample locally, encode/decode remotely |
| Workflows | `h3_remote_ref2va.json` | `h3_async_prefill.json` + `h3_ref2va_async_pub.json` |

Menu category: `Remote Pipe`.

```text
cd ComfyUI/custom_nodes
git clone https://github.com/uraniumchonk/comfy-h3-remote RemotePipe
```

---

## Scene A — remote denoise

The client keeps official Ref2VA (CLIP + empty latent) and VAE decode. Drop the UNET / Sampler chain for one `Pipe Denoise` node.

```text
MiniMaxH3ReferenceToVideo
        │ positive          │ LATENT
        ▼                   ▼
        └────── Pipe Denoise ──────┐
                                   ▼
              VAEDecode + VAEDecodeAudio → VHS
```

Or decode remotely too: `Pipe Denoise` → `Pipe Decode (sync)` → VHS.

| Field | Notes |
|---|---|
| steps / sampler / scheduler / seed | Same as KSampler |
| cfg | **Must be 1.0** (H3 is flow-matching) |
| denoise | `0–1`, not 12 |
| shift_video / shift_audio | Defaults 12 / 3 |
| server_url | `http://<gpu>:8090/upstream/minimax-h3-ref2va` |

Example: `examples/workflows/h3_remote_ref2va.json`.

Server:

```bash
python server/h3_server.py \
  --model /path/to/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
  --tp --host 127.0.0.1 --port 8299
```

INT8+ConvRot + TP2 fits 2×3080 20GB. No vLLM-omni. Swap with LLMs via llama-swap. See `docs/efficiency.md`.

---

## Scene B — local denoise, remote CLIP + VAE

Keep the local card on the Sampler. Ship CLIP 32B and Video/Audio VAE to the GPU box. Mailboxes span Comfy queues so **encode/decode cost vanishes** and wall time is denoise:

- Start of a round: Collect last CLIP (to denoise) and last VAE (to save the video)
- End of a round: Submit this decode, Submit next encode

While you denoise, the remote box decodes the previous clip and encodes the next one.

### How to run

Step-by-step (EN/中): `docs/async-howto.md`.

1. **Prefill** (`h3_async_prefill.json`) — Queue once so the mailbox has a CLIP job. Dummy is fine. Then start the main loop.
2. **First real round** (`h3_ref2va_async_pub.json`) — Collect that CLIP → **local denoise** → Decode Submit (for the next start) → Encode Submit (next CLIP). VAE Collect is empty (`audio = False`); the graph Switch handles it.
3. **Later rounds** — at queue start, Collect last VAE and save; Collect CLIP and denoise the next pack.

Whatever you set on the left this round is what the *next* round denoises.

### Nodes

| Node | Role |
|---|---|
| Pipe Encode Submit / Collect | CLIP mailbox. Collect errors if empty; waits if still running. Submit `latent` can pipe Collect through |
| Pipe Decode Submit / Collect | AV VAE mailbox. Submit `trigger` is the same latent. Collect `trigger` is order only; no audio → `False` |
| Pipe Encode / Decode (sync) | Wait in the same graph (no mailbox) |

`server_url`:

| Service | URL |
|---|---|
| encode | `http://<gpu>:8090/upstream/minimax-h3-clip-encode` |
| decode | `http://<gpu>:8090/upstream/minimax-h3-vae-decode-1` (1 GPU, can sit next to encode) |

Server (both can stay up, split GPUs):

```bash
python server/h3_clip_encode.py \
  --clip /path/to/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --video-vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --host 127.0.0.1 --port 8301

python server/h3_vae_decode.py \
  --video-vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --dp 1 --host 127.0.0.1 --port 8300
```

Calls wait for `/health` before the real POST. Same service is FIFO. Kitchen RoPE only sees `cuda:0`; multi-GPU decode needs one process per card and `CUDA_VISIBLE_DEVICES` remap.

llama-swap snippet: `examples/llama-swap.yaml` (`h3-async`: encode on GPU 1, decode-1 on GPU 0).

---

## Hardware notes (2×3080 20GB)

- Scene A DiT TP2: idle ~5GB + 2.2GB. 0.3MP ≈ 38–65s/step, 0.6MP ≈ 240s/step (attention is O(n²)).
- Scene B, measured (local 8-step LoRA denoise, remote CLIP+VAE): **0.4MP / 10s / 8 steps — denoise 91s, full queue 95.90s**. The extra ~4s is ffmpeg mux. Collect / Submit / other nodes are effectively free. Denoise is fully async.
- Scene B decode 0.6MP 124 frames: ~44s on 1 GPU; 243×672×448 fp32 pack ≈ 880MB.
- Encode: VAE first, offload to RAM, then CLIP (keep ~12GB free for vision / dequant).
- No NVLink, BAR1=256MiB: NCCL is SHM/direct only.

## Docs

- `docs/async-howto.md` — Scene B, English + 中文
- `docs/environment.md` / `docs/efficiency.md`
- `plan.md` / `decode_plan.md`
