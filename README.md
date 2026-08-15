# Remote Pipe（MiniMax H3）

把 MiniMax H3 的重活（CLIP encode、DiT denoise、Video/Audio VAE decode）丟到 GPU 伺服器。ComfyUI 客戶端只跑圖、Queue、存影片。

兩種用法：

- **同步**：這一輪等遠端跑完再往下。接線最少。
- **跨圖非同步**：Submit 丟出去立刻結束，下一輪 Collect 再取結果。denoise 的幾十分鐘裡，160 可以同時解上一支、編下一支。

## 為什麼要拆機

H3 的 DiT 吃多卡、CLIP 32B、Video VAE 也大。客戶端（例如單張 4070）不該載這些。GPU 箱本來就在跑 LLM，用 llama-swap 跟 H3 互斥換卡：要出片時載 H3，平時把卡還給 vLLM。

DiT 走本專案的 INT8+ConvRot + TP2（`server/h3_tp.py`），2×3080 20GB 就能跑。不走 vLLM-omni / 整包 BF16。

## 安裝（客戶端）

```text
cd ComfyUI/custom_nodes
git clone https://github.com/uraniumchonk/comfy-h3-remote RemotePipe
```

重啟 ComfyUI。選單分類：`Remote Pipe`。

| 節點 | 作用 |
|---|---|
| Pipe Encode (sync) | 官方 Ref2VA 整段（CLIP + ref VAE）丟 160，等結果 |
| Pipe Encode Submit / Collect | 跨圖信箱：Submit 立刻丟，下一輪 Collect 拿 positive + latent |
| Pipe Denoise | 遠端 DiT。取代 UNET + Guider + Sampler 那串 |
| Pipe Decode (sync) | 遠端 video + audio VAE，等畫面和聲音 |
| Pipe Decode Submit / Collect | 跨圖信箱：Submit 丟 AV latent；Collect 的 `trigger` 只決定執行順序，輸出 `images` + `audio` |
| Pipe LoRA Stack | 多顆 LoRA 串給 Denoise |

## 同步（一張圖跑完）

```text
素材 / prompt
    → Pipe Encode (sync) → positive + latent
    → Pipe Denoise
    → Pipe Decode (sync) → images + audio → VHS
```

`server_url` 預設：

| 服務 | URL |
|---|---|
| encode | `http://<gpu>:8090/upstream/minimax-h3-clip-encode` |
| denoise | `http://<gpu>:8090/upstream/minimax-h3-ref2va` |
| decode | `http://<gpu>:8090/upstream/minimax-h3-vae-decode`（DP2）或 `.../minimax-h3-vae-decode-1`（單卡） |

Denoise：`cfg` 必須 1.0（H3 是 flow-matching）。turbo LoRA 用 `euler` 4～8 step。

## 跨圖非同步（串行加速）

同一張 pipeline 反覆 Queue。**這一輪左邊填的素材，是下一輪才 denoise 的。**

先跑一次 Prefill（只 Submit encode），再進 pipeline：

```text
Prefill     Encode Submit A
Queue 1     Encode Collect A → Denoise A → Decode Submit A
            denoise 完再 Encode Submit B
Queue 2     Encode Collect B → Denoise B → Decode Submit B
            Decode Collect.trigger → VHS 存出 A（images + audio）
```

接線重點：

- Decode Submit 接 **這一輪** denoise 的 LATENT，沒有輸出。
- Decode Collect 的 `trigger` 也接 denoise（只為了等它跑完），**不要**當成「解這一輪」。輸出是上一輪已經 ready 的畫面和聲音。
- 第一輪 Collect 信箱空：8×8 黑圖 + 靜音，正常。
- Encode Collect：上一輪 encode 還在跑就堵住等；真的沒上一輪才報錯。

信箱在客戶端 `ComfyUI/output/h3_*_mailbox.json`，本體在 `h3_*_jobs/`。

## 伺服器

需要 ComfyUI 0.30+（MiniMax H3 + comfy_kitchen）和 INT8+ConvRot 權重（不在本 repo）。

```bash
# DiT TP2
python server/h3_server.py \
  --model /path/to/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
  --tp --host 127.0.0.1 --port 8299

# CLIP encode（單卡，權重 offload 到 RAM）
python server/h3_clip_encode.py \
  --clip /path/to/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --video-vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --host 127.0.0.1 --port 8301

# VAE decode（video + audio；--dp 1 單卡 / 2 雙卡 chunk）
python server/h3_vae_decode.py \
  --video-vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --dp 1 --host 127.0.0.1 --port 8300
```

llama-swap 範本：`examples/llama-swap.yaml`。

建議分兩組：

- **同步**：`minimax-h3-ref2va` 跟 `minimax-h3-vae-decode`（DP2）互斥換卡
- **非同步**：`minimax-h3-clip-encode`（卡 1）跟 `minimax-h3-vae-decode-1`（卡 0）可同時掛著

Kitchen RoPE/dlpack 只認 `cuda:0`。雙卡 decode 必須獨立 process + `CUDA_VISIBLE_DEVICES` remap，不能同進程雙卡。

## 硬體參考（2× RTX 3080 20GB）

- DiT TP2 閒置：`cuda:0 ≈ 5GB / cuda:1 ≈ 2.2GB`。0.3MP ≈ 38–65s/step，0.6MP ≈ 240s/step（attention O(n²)，TP 不減總 FLOPs）。
- Decode 0.6MP 124 幀：單卡 ≈ 44s，DP2 ≈ 27s。
- Encode：VAE 先、卸回 RAM、再 CLIP（預留 12GB 給 vision / INT8 dequant）。圖+短影片大約數十秒；長參考影片會更久，但不应 OOM。
- 無 NVLink、BAR1=256MiB：NCCL 只能 SHM/direct。

## 文件

- `docs/environment.md` — 部署環境
- `docs/efficiency.md` — TP2 效率帳
- `examples/workflows/` — 工作流（async 範例另補）
- `plan.md` — sequence parallel
- `decode_plan.md` — decode 架構筆記
