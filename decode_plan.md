# 三站分工架構計劃（10號 denoise / 160 encode+decode）

> 主人拍板架構：10 號機專職 denoise；160 雙卡跑 encode（CLIP）+ decode（VAE）。
> 動機：10 號機 RAM 只有 47.9GB，同時裝 CLIP 25.28GB + DiT 20GB + VAE 5.2GB 超載，
> 所以 denoise 被迫遠端。把 CLIP+VAE 挪到 160（RAM 94GB）後，10 號機專心 denoise。

## 目標架構

```
10號機 (4070TiS 16GB / RAM 47.9GB, ComfyUI 0.30 主體)
  LoadImage/LoadVideo -> RemoteEncodeNode -> HTTP -> 160 卡1
160 卡1 (3080 20GB): CLIP Qwen3VL-32B INT8(25.28GB, offload) + video/audio VAE encode ref
  <- cond (Qwen hidden states ~0.5GB) + ref latents
10號機: 本地 denoise (DiT INT8 20GB offload, 4070TiS)
  -> latent (7MB) -> HTTP -> 160 卡0
160 卡0 (3080 20GB): video VAE decode (ViT3D 36層, 5.2GB) + audio decode (BigVGAN 605MB)
  <- frames [B,H,W,C] fp32 [0,1] + audio waveform
10號機: VHS_VideoCombine
```

- 10 號機 RAM 負載：DiT 20GB + 系統 → 47.9GB 綽綽有餘（現況 50GB 超載解除）
- 160 卡1 CLIP 25.28GB > 20GB VRAM → offload（或將來 2TP 拆兩卡全駐 VRAM）
- 160 卡0 VAE 5.2GB 直接駐 VRAM
- 跨單 pipeline：10 號機 denoise(i) 期間，160 可 encode(i+1) / decode(i-1)

## 並行選項（world 參數）

- encode：TP 1/2（Qwen3VL-32B 2TP 拆 12.6GB/卡全駐 VRAM，免 offload）
- decode：tile DP 1/2/4（chunk×tile 單元平分，零 NCCL、完美線性）
- denoise：10 號機單卡 offload（無 TP；4070TiS fp16 160T 比 3080 119T 強 34%）

## 關鍵結構事實

- H3 video VAE decoder = ViT3D：36 層、dim 2048、32 heads、full attention（與 DiT 同構）
- decode 路徑：decode_temporal（7 chunks, overlap 2 tokens）-> tiled_decode（21 tiles,
  256 像素 + overlap 64）-> 每 tile ViT3D 全跑 → 獨立單元 ~147 個，完美 DP
- 160 執行樹 = ComfyUI 0.33.0（10 號機 0.30.0）：0.33 decode 輸出 [0,1] fp32、
  有 operations 注入、chunked I/O；回傳節點側用 [0,1]（ComfyUI IMAGE 標準）
- decode 實測 ~20s（10 號機 4070TiS），並行化收益小；搬遷價值在管線分工

## 實作里程碑

- M0: VAE 權重 scp 到 160（models/vae/）                                    [進行中]
- M1: h3_encode_server.py：160 卡1 遠端 encode 服務
  - CLIP Qwen3VL-32B INT8 convrot 載入（offload）+ ref VAE encode
  - payload 進：prompt + ref 影像/影片/音訊；cond + ref latents 出
- M2: h3_vae_decode.py：160 卡0 遠端 decode 服務
  - MiniMaxH3VideoVAE + audio VAE（0.33 行為、[0,1] fp32）
  - tile DP world=1/2/4（單元平分、main 線程 blend + canvas）
- M3: 10 號機本地 denoise 落地
  - 先驗證 ComfyUI 0.30 原生 H3 採樣（KSampler/官方採樣節點）在 4070TiS offload 的速度
  - 太慢再移植 h3_server 單卡 resident/offload 優化版到 Windows
- M4: custom nodes
  - RemoteEncodeNode（10 號機發 encode 請求、收 cond）
  - RemoteDecodeNode（10 號機發 decode 請求、收 IMAGE/AUDIO）
  - 工作流更新：拔掉 VAELoader/CLIPLoader/RemoteDenoiseNode，換新三件套
- M5: 部署 + 端到端驗證（0.3MP 快測 → 0.6MP 完整單）

## 收益與風險

| 項目 | 量級 |
|---|---|
| 10 號機 RAM 超載解除 | 50GB -> 20GB（主因） |
| 10 號機只 denoise | 4070TiS 全算力給 DiT |
| 160 專職前後處理 | encode/decode 與 denoise 跨單重疊 |
| decode 並行 | 20s -> ~10s（2DP），4 卡 ~5s（小但免費） |

| 風險 | 對策 |
|---|---|
| 10 號機 denoise 單卡速度未知 | M3 先實測原生採樣；慢於 240s/step 太多就移植 h3_server 優化版 |
| CLIP 25.28GB 卡1 offload 慢 | 先 1 卡 offload 跑通；2TP 全駐 VRAM 是升級路徑 |
| cond 傳輸 0.5GB+ | 千兆 ~4-5s/單，可接受；之後可快照壓縮 |
| 0.30/0.33 decode 輸出域不同 | 回傳 [0,1]（ComfyUI IMAGE 標準） |
