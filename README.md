# ComfyUI Remote Denoise (MiniMax H3)

## Goal

Run MiniMax H3 Ref2VA denoise inside ComfyUI with multi-GPU tensor parallelism (TP2 / TP4). Anyone running local LLMs already owns a multi-GPU server — that same box can also run compute-heavy ComfyUI video generation and benefit from multi-GPU acceleration. The client only runs CLIP / VAE / Ref2VA encoding; the DiT runs sharded across GPUs on a remote box, and the denoised latent is sent back to the client for decoding.

## Problems solved

- **ComfyUI has no multi-GPU TP out of the box**: MiniMax H3 Ref2VA actually fits on a single GPU, but if you own a 2 / 4-GPU server, those idle cards are wasted potential — video generation is extremely compute-heavy, exactly where multi-GPU acceleration pays off the most. ComfyUI DiTs only run on one GPU by default; multi-GPU TP simply doesn't exist natively. `server/h3_tp.py` implements Megatron-style TP from scratch: qkv / fc1 are column-parallel (QKV / SwiGLU use index maps, not naive half-splits), out / fc2 are row-parallel with all-reduce, and it scales to 2 or 4 cards with all 50 DiT blocks evenly spread across them.
- **No vLLM-omni, no giant full BF16 models**: the alternative route is dumping the full BF16 model into vLLM-omni, which needs a huge number of big cards. This project uses INT8+ConvRot quantization + TP instead, so 2× RTX 3080 20GB is enough for short clips.
- **NCCL without P2P**: these cards have no NVLink and BAR1 is only 256MiB (`can_device_access_peer` = False), so P2P is impossible. `torch.cuda.nccl` falls back to SHM / direct host staging — measured 0.76ms for a 2M-element all-reduce and broadcast is ~2× faster than `.to()`. The server disables P2P / IB at startup and warms up one communicator at load time.
- **Block-wise CPU offload (Comfy lowvram / Wan2GP / Omni DLO style)**: 50 blocks stay in pinned RAM; while layer *i* computes, layer *i+1* is prefetched onto the cards. The first `resident` layers (auto-sized from free VRAM) stay resident to cut the per-layer PCIe sync. AdaLN projections (0.5–1GB each) stay on CPU — only the few-KB gates cross.
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
| steps / sampler / scheduler / seed | Same as on KSampler |
| cfg | **Always 1.0 for H3.** H3 is flow-matching; the official template uses BasicGuider (cfg=1.0, no CFG). Any cfg > 1 amplifies positive/negative error and the output comes out as snow/black. |
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

## VRAM & performance (reference: 2× RTX 3080 20GB TP2)

- Idle after load: `cuda:0 ≈ 5GB / cuda:1 ≈ 2.2GB` (leftovers on 0, AdaLN on CPU, prefix auto-sized).
- 0.3MP: ~65s/step. 0.6MP: ~240s/step — the 3.7× jump for 2× pixels is attention `O(n²)`, not the TP layer.
- A 10-second 0.3MP clip will saturate both cards. Lower duration / megapixels, or go TP4 to spread across more cards.
- One line per step in the server log: `[h3-server] step 3/20  65.1s  elapsed 195.3s`.

## Next up: sequence parallel (plan.md)

TP is at its efficiency ceiling for this hardware (0.75 × 2 cards, no P2P). The next win is Ulysses-style sequence parallel for attention — see `plan.md` (240s → ~150–170s expected for 0.6MP).

## Docs

- `docs/environment.md` — 兩台機器部署環境快照（GPU 伺服器 192.168.0.160 / Windows 客戶端 192.168.0.10）
- `docs/efficiency.md` — 雙 3080 TP2 vs 4070TiS 的算力與 step 時間拆帳（0.75 效率為 vLLM 實測）
- `examples/workflows/h3_remote_ref2va.json` — 正式工作流（含 RemoteDenoiseNode，可直接匯入 ComfyUI）
- `plan.md` — sequence parallel 實作計劃（Ulysses-style attention）
