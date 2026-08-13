# MiniMax H3 Remote Denoise

ComfyUI 客戶端把 Ref2VA DiT 的 denoise 丟給遠端 GPU 主機跑：客戶端只做
CLIP / VAE / 編碼，DiT 活在另一台機器（llama-swap `/upstream` 熱載入）。

## 接線

原本（全在本機）：

```
UNETLoader → BasicGuider → RandomNoise → KSamplerSelect
                                              ↓
EmptyLatent → BasicScheduler → SamplerCustomAdvanced → VAEDecode
```

現在：

```
MiniMaxH3ReferenceToVideo.positive ─┐
                                    ├→ Remote Denoise Node (H3) → VAEDecode
MiniMaxH3ReferenceToVideo.LATENT ───┘                          → VAEDecodeAudio
```

客戶端**不要載 UNET、不要接 KSampler**。`steps` / `cfg` / `sampler` / `seed` /
`denoise` / `shift` 全寫在 Remote 節點上：

- `denoise`：0–1
- `shift_video`：預設 12
- `seed`：後面有 Comfy 自動加的 fixed/randomize
- `server_url` 預設 `http://192.168.0.160:8090/upstream/minimax-h3-ref2va`

## 目錄

```
custom_node/   ComfyUI custom node（拷到 ComfyUI/custom_nodes/）
server/        FastAPI denoise server + TP2 shard（跑在 GPU 主機）
examples/      llama-swap 設定範例
```

## Server（GPU 主機）

```bash
export COMFYUI_ROOT=/path/to/ComfyUI        # comfy/ 本體，不在本 repo
export PYTHONPATH=$PWD/server
python server/h3_server.py \
  --model /path/to/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
  --tp --host 0.0.0.0 --port 8299
```

VRAM：2× RTX 3080 20GB TP2。短片 OK；長片會灌滿兩張卡。

## Client（ComfyUI）

把 `custom_node/` 拷到 `ComfyUI/custom_nodes/RemoteDenoise/`，重啟 ComfyUI，
節點出現在 **MiniMax H3** 類別。

## License

MIT © 2026 uraniumchonk
