# 效率計算：雙 3080 TP2 vs 單卡 4070TiS

> 目的：解釋「為什麼雙卡 3080 跑 0.3MP 是 65s/step，而單卡 4070TiS 是 50s/step」，並推估 0.6MP 該怎麼看。

## 1. 理論算力（dense tensor core）

| 配置 | FP16 | INT8 |
|---|---|---|
| RTX 3080 單卡 | 59.54 TF | 119.1 TF |
| RTX 3080 ×2 × 效率 | 89.3 TF | 178.7 TF |
| RTX 4070 Ti Super | 88.0 TF | 176.0 TF |

> 0.75 擴展效率：來自 **vLLM 在本機 2×3080（無 NVLink、BAR1 256MiB、NCCL SHM）的實際測試計算**，不是理論值。它代表「兩卡合計有效算力 = 單卡 × 2 × 0.75」。

雙卡 3080 的有效算力與 4070TiS **幾乎打平**（差 ~1.5%），FP16 與 INT8 皆然。

## 2. 實測對照（0.3MP）

| 配置 | step 時間 | 說明 |
|---|---|---|
| 雙 3080 TP2（NCCL SHM + offload） | ~65s | 含每層權重 H2D + comm |
| 4070TiS 單卡 | ~50s | 無 TP 開銷 |

算力相同，為何 4070TiS 快 1.3×？拆帳：

```
65s ≈ 50s（等同算力的純計算）
    + 15s（TP 固定開銷：100 次權重 H2D/step + NCCL SHM comm + thread 同步）
```

- 權重 offload 的位元組數與解析度**無關**（50 層 × CFG 正負 = 100 次 H2D/step，固定）。
- comm（broadcast/all-reduce）隨 seq 成長，但對 0.3MP 占比不大。
- 結論：**解析度越大，這 30% 固定開銷占比越小**：0.3MP 輸 30%，0.6MP 只輸 ~15%。

## 3. 0.6MP 推估

- 0.3MP → 0.6MP 是 2× 像素，實測時間 65s → 240s = **3.69×**，接近 attention `O(n²)` 的 4×，證明 0.6MP 是 attention 主導。
- 4070TiS 若同樣縮放：50 × 3.69 ≈ **185s/step**。但 4070TiS 只有 16GB，20GB 權重裝不下，必須全 offload，實際會更慢（估計 400s+），且未經驗證。
- 結論：**0.6MP 只有雙卡 3080 這條路**（240s/step），下一步想壓時間靠 sequence parallel（`plan.md`，預期 150–170s）。

## 4. 場景 B 管線實測（本地 denoise + 遠端 CLIP/VAE）

| 項目 | 值 |
|---|---|
| 解析度 | 0.4MP |
| 長度 | 10s（約 124 幀 @ 24fps） |
| 採樣 | 8 step + LoRA（本地） |
| Denoise | **91s** |
| 整張 Queue | **95.90s** |
| 差額 | ~4s ffmpeg 轉檔 |

Collect / Submit / 其他節點可忽略。遠端 CLIP 與 VAE 完全疊在下一輪 / 上一輪，不擋本地 Sampler。

## 5. 驗證數字

| 項目 | 數值 |
|---|---|
| 2M 元素 NCCL all-reduce | 0.76ms |
| 2M 元素 NCCL broadcast | 0.24ms（比 `.to()` 快 ~2.3×） |
| NCCL comm 建立（warmup） | 317ms |
| 0.3MP step | ~65s（雙 3080） |
| 0.6MP step | ~240s（雙 3080） |
