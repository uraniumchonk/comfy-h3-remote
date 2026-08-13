> **⚠️ Still under active development. Not ready for use.**
>
> This project is currently experimental. The architecture, node interface, and communication protocol may change significantly at any time.  
> It is not guaranteed to work, and is not recommended for any production or serious workflows.  

# ComfyUI Remote Denoise (MiniMax H3)

## 目標

讓 ComfyUI 能跑 MiniMax H3 Ref2VA 的 denoise，但 DiT 用自寫的多卡 tensor parallel（TP2 / TP4）執行，不走 vLLM-omni 整包載入完整 BF16 模型的路線。客戶端只跑 CLIP / VAE / Ref2VA 編碼，DiT 在遠端 GPU box 上切卡跑，denoise 完的 latent 送回客戶端解碼。

## 解決的痛點

- **ComfyUI 原生沒有多卡 TP**：ComfyUI 的 DiT 預設只能單卡跑，MiniMax H3 Ref2VA 的 DiT 單卡裝不下。`server/h3_tp.py` 自寫 Megatron-style TP：qkv / fc1 走 column-parallel（QKV / SwiGLU 用 index map，不是對半切），out / fc2 走 row-parallel + all-reduce，2 卡或 4 卡都能切，50 個 DiT block 均勻散到各卡。
- **不用 vLLM-omni 跑超大完整 BF16**：整包 BF16 模型要靠 vLLM-omni 這類方案，需要極多張大卡。這裡用 INT8+ConvRot 量化 + TP，2× RTX 3080 20GB 就能跑短片。
- **接線簡化（附帶）**：官方模板要接 `UNETLoader → BasicGuider`、`RandomNoise + KSamplerSelect + BasicScheduler → SamplerCustomAdvanced` 一串節點，客戶端還得載 UNET。這個節點把整段收成一個，客戶端不載 UNET、不碰採樣器。
- **GPU 共享（附帶）**：GPU box 可掛 llama-swap，跟 LLM 互斥換卡。H3 要跑時才載入，平時卡留給 vLLM / LLM，不用為 H3 留一台專用機。

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

`--tp`：tensor parallel。目前走 TP2（`h3_server.py` 呼叫 `apply_tp` 用預設 2 卡）；`h3_tp.py` 的 index map 切法本身支援 2 / 4 卡整除（heads 56、FFN 14336、inner 7168 都除得盡 4），要上 TP4 把 `apply_tp` 的 devices 參數接成 4 卡即可。QKV / SwiGLU 用 index map 切，不是對半切；out / fc2 用 row-parallel + all-reduce。

llama-swap 範本：`examples/llama-swap.yaml`。載入 `/upstream/minimax-h3-ref2va/health`，卸載 `POST /api/models/unload`。

## VRAM

2× RTX 3080 20GB TP2。短片（約 5 frame、0.3MP）OK。10 秒 0.3MP 會把兩張卡灌滿。先降 duration / megapixels，或上 TP4 攤更多卡。
