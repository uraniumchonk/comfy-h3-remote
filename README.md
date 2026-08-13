> **⚠️ Still under active development. Not ready for use.**
>
> This project is currently experimental. The architecture, node interface, and communication protocol may change significantly at any time.  
> It is not guaranteed to work, and is not recommended for any production or serious workflows.  

# ComfyUI Remote Denoise (MiniMax H3)

## Goal

Run MiniMax H3 Ref2VA denoise inside ComfyUI with multi-GPU tensor parallelism (TP2 / TP4). Anyone running local LLMs already owns a multi-GPU server — that same box can also run compute-heavy ComfyUI video generation and benefit from multi-GPU acceleration. The client only runs CLIP / VAE / Ref2VA encoding; the DiT runs sharded across GPUs on a remote box, and the denoised latent is sent back to the client for decoding.

## Problems solved

- **ComfyUI has no multi-GPU TP out of the box**: MiniMax H3 Ref2VA actually fits on a single GPU, but if you own a 2 / 4-GPU server, those idle cards are wasted potential — video generation is extremely compute-heavy, exactly where multi-GPU acceleration pays off the most. ComfyUI DiTs only run on one GPU by default; multi-GPU TP simply doesn't exist natively. `server/h3_tp.py` implements Megatron-style TP from scratch: qkv / fc1 are column-parallel (QKV / SwiGLU use index maps, not naive half-splits), out / fc2 are row-parallel with all-reduce, and it scales to 2 or 4 cards with all 50 DiT blocks evenly spread across them.
- **No vLLM-omni, no giant full BF16 models**: the alternative route is dumping the full BF16 model into vLLM-omni, which needs a huge number of big cards. This project uses INT8+ConvRot quantization + TP instead, so 2× RTX 3080 20GB is enough for short clips.
- **Simplified wiring (bonus)**: the official template requires a chain of nodes — `UNETLoader → BasicGuider`, `RandomNoise + KSamplerSelect + BasicScheduler → SamplerCustomAdvanced` — and the client has to load UNET. This node collapses the whole chain into one; the client never loads UNET or touches samplers.
- **GPU sharing (bonus)**: the GPU box can sit behind llama-swap and swap exclusively with LLMs. H3 is only loaded when needed; the rest of the time the cards serve vLLM / LLM. One machine, two jobs.

## Client

```text
cd ComfyUI/custom_nodes
git clone https://github.com/uraniumchonk/comfy-h3-remote ComfyUI-RemoteDenoiseH3
```

Restart ComfyUI. Menu: `MiniMax H3` → `Remote Denoise Node (H3)`.

## Wiring

The official template requires a chain of samplers:

```text
UNETLoader → BasicGuider
RandomNoise + KSamplerSelect + BasicScheduler → SamplerCustomAdvanced
MiniMaxH3ReferenceToVideo ──latent/cond──▶ SamplerCustomAdvanced ──▶ VAE Decode
```

This node absorbs that whole chain. Replace it with:

```text
MiniMaxH3ReferenceToVideo
        │ positive          │ LATENT
        ▼                   ▼
        └────── Remote Denoise Node (H3) ──────┐
                                               ▼
                          VAEDecode + VAEDecodeAudio → Video Combine
```

The client must not load UNET, and must not wire up KSampler / SamplerCustomAdvanced / Guider / Scheduler / RandomNoise.

| Field | Meaning |
|---|---|
| steps / cfg / sampler / scheduler / seed | Same as on KSampler |
| denoise | `0–1`. **NOT 12** |
| shift_video | Default `12` |
| shift_audio | Default `3` |
| server_url | See below |

There is a `fixed` / `randomize` toggle after seed (added automatically by Comfy). Don't leave it out when saving the workflow, or 12 will leak into denoise.

`server_url`:

- llama-swap: `http://192.168.0.160:8090/upstream/minimax-h3-ref2va`
- Direct: `http://<gpu-box>:8299`

## Server

Requires ComfyUI 0.30+ (MiniMax H3 + comfy_kitchen) and INT8+ConvRot weights. The weights are not in this repo.

```bash
export COMFYUI_ROOT=/path/to/ComfyUI
python server/h3_server.py \
  --model /path/to/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
  --tp --host 0.0.0.0 --port 8299
```

`--tp`: tensor parallelism. Currently runs TP2 (`h3_server.py` calls `apply_tp` with the default 2 devices); the index-map sharding in `h3_tp.py` itself supports 2 / 4 cards (heads 56, FFN 14336, inner 7168 all divide evenly by 4) — to go TP4, just wire `apply_tp`'s devices parameter to 4 cards. QKV / SwiGLU are cut with index maps, not naive half-splits; out / fc2 use row-parallel + all-reduce.

llama-swap template: `examples/llama-swap.yaml`. Load via `/upstream/minimax-h3-ref2va/health`, unload via `POST /api/models/unload`.

## VRAM

2× RTX 3080 20GB TP2. Short clips (~5 frames, 0.3MP) are fine. A 10-second 0.3MP clip will saturate both cards. Lower duration / megapixels, or go TP4 to spread across more cards.
