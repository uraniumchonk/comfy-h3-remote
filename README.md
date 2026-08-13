# ComfyUI Remote Denoise (MiniMax H3)

客戶端只跑 CLIP / VAE / Ref2VA 編碼。Ref2VA DiT 在遠端 GPU（可掛 llama-swap 跟 LLM 互斥換卡）。

## 客戶端

```text
cd ComfyUI/custom_nodes
git clone https://github.com/uraniumchonk/comfy-h3-remote ComfyUI-RemoteDenoiseH3
```

重啟 ComfyUI。選單：`MiniMax H3` → `Remote Denoise Node (H3)`。

## 接線

官方模板要接一串採樣器：

```text
UNETLoader → BasicGuider
RandomNoise + KSamplerSelect + BasicScheduler → SamplerCustomAdvanced
MiniMaxH3ReferenceToVideo ──latent/cond──▶ SamplerCustomAdvanced ──▶ VAE Decode
```

這個節點把上面整段收進去。改成：

```text
MiniMaxH3ReferenceToVideo
        │ positive          │ LATENT
        ▼                   ▼
        └────── Remote Denoise Node (H3) ──────┐
                                               ▼
                          VAEDecode + VAEDecodeAudio → Video Combine
```

客戶端不要載 UNET，不要接 KSampler / SamplerCustomAdvanced / Guider / Scheduler / RandomNoise。

| 欄位 | 意思 |
|---|---|
| steps / cfg / sampler / scheduler / seed | 原本在 KSampler 上的 |
| denoise | `0–1`。**不是 12** |
| shift_video | 預設 `12` |
| shift_audio | 預設 `3` |
| server_url | 見下 |

seed 後面有一格 `fixed` / `randomize`（Comfy 自動加）。存工作流別漏，否則 12 會擠進 denoise。

`server_url`：

- llama-swap：`http://192.168.0.160:8090/upstream/minimax-h3-ref2va`
- 直連：`http://<gpu-box>:8299`

## 服務端

需要 ComfyUI 0.30+（MiniMax H3 + comfy_kitchen）跟 INT8+ConvRot 權重。權重不在這個 repo。

```bash
export COMFYUI_ROOT=/path/to/ComfyUI
python server/h3_server.py \
  --model /path/to/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
  --tp --host 0.0.0.0 --port 8299
```

`--tp`：兩張卡切 50 個 DiT block（QKV/SwiGLU 用 index map，不是對半切）。

llama-swap 範本：`examples/llama-swap.yaml`。載入 `/upstream/minimax-h3-ref2va/health`，卸載 `POST /api/models/unload`。

## VRAM

2× RTX 3080 20GB TP2。短片（約 5 frame、0.3MP）OK。10 秒 0.3MP 會把兩張卡灌滿。先降 duration / megapixels。
