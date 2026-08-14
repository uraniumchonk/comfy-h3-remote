# Sequence Parallel 計劃（Ulysses-style attention）

> 目標：0.6MP 正式版 240s/step → 150–170s/step
> 前提：TP 已到效率天花板（0.75 × 2 卡、無 P2P、BAR1 256MiB、NCCL SHM host staging）。接下來砍的不是算力，是**每層的 comm 體積與同步點**，以及 attention 的 activation 峰值。

---

## 1. 現況拆帳（0.6MP，240s/step）

| 項目 | 說明 | 佔比粗估 |
|---|---|---|
| attention 計算 | packed seq S 很長，QK^T 是 O(S²)。TP 只切 head，每卡仍看整條 S | 最大宗 |
| 每層 comm | attn + mlp 各 1 次 `_nccl_broadcast(x)` + 1 次 all-reduce hidden = 4 次 S×H×2B / 層 × 50 層 | 次大 |
| 權重 offload | H2D 100 次/step（CFG 正負），位元組固定、與解析度無關 | 固定成本 |
| attention activation | [S, S] 級 workspace 把 cuda:0 吃乾 → resident 被迫壓低 → offload 變多 | 連鎖 |

關鍵事實：**TP 切 head 沒有減少總 attention FLOPs**。兩卡加起來算的 QK^T 跟單卡一樣多，只是拆開平行。要真正把 S² 砍半，只能切 sequence。

---

## 2. 設計：Ulysses 單機版（2 卡，Megatron 語義）

```
輸入 x: [S, H]   → 切成 x0 [S/2, H] / x1 [S/2, H] 兩卡各持一份（residual 不再廣播）
      │
qkv 各卡算自己的 seq 半條：q/k/v = [S/2, H/P]（column-parallel 繼續切 head）
      │
all-to-all（head × seq 交換）：
      卡 0 拿到 [S, H/2] 的 q/k/v（整條 seq、一半 head）
      卡 1 拿到 [S, H/2] 的 q/k/v
      │
attention：每卡算 [S, S] × (H/2 heads)，但 FLOPs 總量仍是 S²×H
      │
all-to-all 回來 → 每卡回 [S/2, H] 的 attn out
      │
out_proj（row-parallel local GEMM + all-reduce，維持現狀）
      │
mlp：fc1 column + fc2 row + all-reduce，維持現狀
      │
residual 全程只在各自卡上，不再整條廣播
```

### 2.1 換到什麼

| | 現在（head TP） | 之後（+seq parallel） |
|---|---|---|
| 每層 NCCL 次數 | 4（bcast ×2 + allreduce ×2） | 3（all-to-all ×2 + fc2 reduce ×1） |
| 每層 comm 體積 | 4 × S×H×2B | ~3 × S×H×2B（all-to-all 傳 qkv 半條 → 淨省約 1.5×） |
| residual | 每層整條廣播 | 完全本地，零廣播 |
| attention workspace | 每卡 [S, S]×H/P | 每卡 [S, S]×H/P（同，但 QK^T 以 S/2 tile 產生，峰值可降） |
| 同步點 | 層內 4 個 NCCL 屏障 | 層內 3 個，且 attn 區塊單一 all-to-all 取代 bcast+reduce 對 |

淨效果：comm 體積約 -1.5×，同步點 -25%，activation 峰值下降 → resident 可以再拉高 → offload 次數下降。240s 的期望拆解：comm 省 ~25–35s、offload 省 ~10–20s、attention tile 化省 ~5–10s，落在 **150–180s**。

---

## 3. 實作里程碑（每步可驗證）

### M0：profile comm（半天，不改邏輯）
在現行 forward 裡加 `torch.cuda.Event` 計時，拆出每層 `_nccl_broadcast` / all-reduce / offload H2D 各花多少。
- 驗證：印出 0.6MP 一層的 comm 秒數 → 確認 1.5× 省下來的量級
- 若 comm 只佔 <20s，目標下修（誠實面對，不做白工）

### M1：seq shard 切進 attn forward（1–2 天）
改 `_replace_attn` 的 forward：
1. 入參 x 若在 cuda:0 是 [S, H]，切成兩半 `x.chunk(2, dim=0)`，各卡持有自己的
2. qkv 照算（column shard 不變）
3. 用 `torch.cuda.nccl` 實作 2 卡 all-to-all：`reduce_scatter` + `all_gather` 組合（`torch.cuda.nccl.all_gather` / `reduce_scatter` 現成）
   - q/k/v 各自：gather seq → 每卡有 [S, H/P]，再按 head 對調
   - 回來時反向
4. RoPE：**`rms_rope_split_half_` 吃整條 seq 的 position**，切 seq 後每卡只看到自己那半條 → RoPE 必須在切之前做，或把完整 `rope_freqs` 也 all-gather 回來對齊（實作上選前者：qkv 前先對全 seq 算 rope，再切）
5. `mod_segments`（text/video/audio/ref 的 adaln row）目前是對整條 seq 的絕對 index → 切 seq 後要各自重投影成 local index。**這是最大坑**：`PackedLayout` 的 segments 是全域座標，卡 0/1 各自要算「哪些 segment 落在我的 seq 半條內」
- 驗證：單層 amax=0（對齊單卡）+ `python h3_tp.py` block-level 全過

### M2：mlp + residual 切 seq（1–2 天）
1. `_replace_mlp`：fc1 column 照舊（每卡算自己半條 x 的 gate/up），fc2 row + all-reduce 照舊 → 其實 mlp 不需要改，**只要 x 不再整條廣播**：進 attn 前切、出 attn 後各自 residual
2. DiT block 的 `norm1/norm2/adaln`：都對 local seq 半條做，segments 用 M1 的 local 投影
3. 移除 attn/mlp forward 開頭的 `_nccl_broadcast(x, devices)`
- 驗證：50-block forward finite + amax 對單卡 < 1e-6；0.3MP step 時間對照（應該比現在 65s 略快或持平，但不該變慢）

### M3：峰值優化 + resident 重算（提前做了一半）
1. ~~attention 的 QK^T 用 tile~~ **已上**：`seq>=20000` 時 Q 切 2 段，峰值 `[S/2,S]`（1.32G→0.66G）。Ulysses all-to-all 不會縮小這塊。
2. `_pick_resident`：`seq>=20000` 先 resident=0；tile 穩定後再把 24k 拉回 4～8
3. 0.6MP 實測
- 驗證：0.6MP step 時間落在 150–180s；兩卡 VRAM 接近對半

### M4：打包 + 文件
- 同步 `h3_tp.py` / `h3_server.py` 進 repo、更新 README 效能表、plan.md 標完成
- 留 `--no-seq-parallel` fallback flag（預設關，驗證完再開）

---

## 4. 風險與決策

| 風險 | 影響 | 決策 |
|---|---|---|
| 2 卡 all-to-all 在 SHM 上不比 bcast 快 | 白忙 | M0 先量；若無 1.3× 以上收益，停 M1，改做 M3 的 tile 化（也值 10–20s） |
| `mod_segments` / RoPE 切 seq 的 index 錯位 | 數值全錯、難查 | M1 單層 amax=0 當閘門；RoPE 在切前做 |
| 4 卡 scaling 才是 Ulysses 主場 | 2 卡收益打折 | 計劃先以 2 卡為準；架構留 `world` 參數，之後插 4 卡直接吃 |
| 換 3090 + ReBAR 後 P2P 開 | 現有 SHM 路徑變多餘 | NCCL 自動選 P2P；all-to-all 實作不綁 transport，無痛升級 |

---

## 5. 不做的事（明確排除）

- 不做 vLLM-Omni 官方路線（BF16 66.3GB / FP8 SM89 / 384GB RAM，3080 走不了）
- 不做 CUDA Graph（Kitchen dlpack stream=-1，疊不上）
- 不改量化格式（INT8+ConvRot 是這包權重的命根，row 1% 誤差維持可接受）
- 不做 4 卡擴充的完整優化（計劃以 2 卡出貨，4 卡留 world 參數）

---

## 6. 成功標準

- [ ] 0.6MP step 時間 ≤ 180s（目標 150–170s）
- [ ] 兩卡 VRAM 差距 < 20%
- [ ] 每 step 的 offload H2D 次數下降（resident 提升）
- [ ] 數值對齊：column amax=0、row rel ≤ 1%（維持現狀）
- [ ] 連續 2 單 0.6MP 不 OOM、不 NaN
