# Remote Pipe（MiniMax H3）

[English](README.md)

ComfyUI 節點：把 MiniMax H3 的重活丟到 GPU 伺服器。兩個互不綁死的場景。

場景 B 最好的地方：**encoder / decoder 時間無限接近 0**。大頭全在 denoise，也是最久的那步；CLIP 跟 VAE 疊在下一輪 / 上一輪後面。實測 0.4MP / 10 秒 / 8 步 LoRA — **denoise 91 秒，整張 Queue 95.90 秒**（多的約 4 秒是 ffmpeg）。

| | 場景 A | 場景 B |
|---|---|---|
| 誰 denoise | 遠端 DiT（TP2） | **本地** Sampler / UNET |
| 遠端跑什麼 | 只有 denoise | **CLIP encode + AV VAE decode** |
| 圖怎麼跑 | 一張等完 | 跨圖 Queue，本地狂採樣、遠端狂編解 |
| 工作流 | `h3_remote_ref2va.json` | `h3_async_prefill.json` + `h3_ref2va_async_pub.json` |

選單分類：`Remote Pipe`。

```text
cd ComfyUI/custom_nodes
git clone https://github.com/uraniumchonk/comfy-h3-remote RemotePipe
```

---

## 場景 A — denoise 丟遠端

客戶端跑官方 Ref2VA（CLIP + 空 latent）和 VAE decode。UNET / Sampler 整串拿掉，換成一顆 `Pipe Denoise`。

```text
MiniMaxH3ReferenceToVideo
        │ positive          │ LATENT
        ▼                   ▼
        └────── Pipe Denoise ──────┐
                                   ▼
              VAEDecode + VAEDecodeAudio → VHS
```

或遠端順便解：`Pipe Denoise` → `Pipe Decode (sync)` → VHS。

| 欄位 | 說明 |
|---|---|
| steps / sampler / scheduler / seed | 跟 KSampler 一樣 |
| cfg | **必須 1.0**（H3 是 flow-matching） |
| denoise | `0–1`，不是 12 |
| shift_video / shift_audio | 預設 12 / 3 |
| server_url | `http://<gpu>:8090/upstream/minimax-h3-ref2va` |

範例：`examples/workflows/h3_remote_ref2va.json`。

遠端：

```bash
python server/h3_server.py \
  --model /path/to/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
  --tp --host 127.0.0.1 --port 8299
```

INT8+ConvRot + TP2，2×3080 20GB 可跑。不走 vLLM-omni。llama-swap 跟 LLM 互斥換卡。詳見 `docs/efficiency.md`。

---

## 場景 B — 本地 denoise，遠端 CLIP + VAE（跨圖並行）

本地卡狂跑 Sampler。CLIP 32B 和 Video/Audio VAE 丟遠端，信箱跨 Queue，**編解時間被藏掉**，牆上時鐘只剩 denoise：

- 這一輪開頭：Collect 上一輪已經好的 CLIP（開 denoise）和 VAE（存片）
- 這一輪結尾：Submit 這一輪 decode、Submit 下一輪 encode

denoise 那幾分鐘，遠端同時在解上一支、編下一支。

### 怎麼跑

中英逐步：`docs/async-howto.md`。

1. **Prefill**（`h3_async_prefill.json`）Queue 一次，信箱先有一筆 CLIP。可以是墊檔。送出就能開主循環。
2. **第一輪正式**（`h3_ref2va_async_pub.json`）Collect 那筆 CLIP → **本地 denoise** → Decode Submit（留給下一輪開頭）→ Encode Submit（下一輪 CLIP）。VAE Collect 此時空（`audio = False`），用圖裡的 Switch。
3. **之後每一輪開頭** 抓上一輪 VAE 存片，同時 Collect CLIP 開下一包 denoise。

這一輪左邊填的素材，是下一輪才 denoise 的。

### 節點

| 節點 | 作用 |
|---|---|
| Pipe Encode Submit / Collect | CLIP 信箱。Collect 空立刻報錯；還在跑會等。Submit 的 `latent` 可接 Collect 通透，輸出同包 latent |
| Pipe Decode Submit / Collect | AV VAE 信箱。Submit 的 `trigger` 是 latent 通透。Collect 的 `trigger` 只決定順序；沒聲音吐 `False` |
| Pipe Encode / Decode (sync) | 同一張圖裡等遠端跑完（不用信箱） |

`server_url`：

| 服務 | URL |
|---|---|
| encode | `http://<gpu>:8090/upstream/minimax-h3-clip-encode` |
| decode | `http://<gpu>:8090/upstream/minimax-h3-vae-decode-1`（單卡，可跟 encode 並掛） |

遠端（可同時掛，分卡）：

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

請求會先等 `/health` 再送本體；同服務 FIFO 排隊。Kitchen RoPE 只認 `cuda:0`，雙卡 decode 要獨立 process + `CUDA_VISIBLE_DEVICES` remap。

llama-swap 範本：`examples/llama-swap.yaml`（`h3-async` 組：encode 卡 1 + decode-1 卡 0）。

---

## 硬體參考（2×3080 20GB）

- 場景 A DiT TP2：閒置約 5GB + 2.2GB。0.3MP ≈ 38–65s/step，0.6MP ≈ 240s/step（attention O(n²)）。
- 場景 B 實測（本地 8 步 LoRA denoise，遠端 CLIP+VAE）：**0.4MP / 10 秒 / 8 step — denoise 91 秒，整張 Queue 95.90 秒**。多出來的約 4 秒是 ffmpeg 轉檔。Collect / Submit 等節點等於瞬間。denoise 已完全非同步。
- 場景 B decode 0.6MP 124 幀：單卡 ≈ 44s；243 幀 672×448 fp32 回條約 880MB。
- Encode：VAE 先、卸回 RAM、再 CLIP（預留 12GB 給 vision / dequant）。
- 無 NVLink、BAR1=256MiB：NCCL 只能 SHM/direct。

## 文件

- `docs/async-howto.md` — 場景 B 中英步驟
- `docs/environment.md` / `docs/efficiency.md`
- `plan.md` / `decode_plan.md`
