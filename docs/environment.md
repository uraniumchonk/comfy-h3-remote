# 環境簡易報告

> 快照日期：2026-08-14。這是「GPU 伺服器」與「Windows 客戶端」兩邊的部署環境。

## 1. GPU 伺服器（192.168.0.160，meowhome）

| 項目 | 數值 |
|---|---|
| CPU | 12 執行緒（GPU0/GPU1 共用 CPU affinity 0-11） |
| 記憶體 | 主機 RAM 充足（模型 20GB 權重常駐 RAM 待 offload） |
| GPU | 2× RTX 3080 20GB（Ampere SM86，無 NVLink） |
| GPU 互連 | PXB（PEX88024 交換器），BAR1 僅 256MiB，`can_device_access_peer=False` → **無 P2P** |
| NCCL 路徑 | `SHM/direct/direct`（host staging）；comm 建立 317ms、2M 元素 all-reduce 0.76ms |
| CUDA / torch | cu130 / torch 2.13.0+cu130（Comfy venv） |
| ComfyUI | 0.30.0 同源（含 comfy_kitchen、comfy-aimdo） |
| 模型 | `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（20GB，INT8+ConvRot，`models/diffusion_models/`） |
| 服務 | `h3_server.py --tp` 跑在 5203；llama-swap `/upstream/minimax-h3-ref2va`（8090） |
| 常駐 VRAM | 載入後 cuda:0 ≈ 5GB / cuda:1 ≈ 2.2GB（AdaLN 在 CPU、prefix 自動挑選） |

## 2. Windows 客戶端（192.168.0.10）

| 項目 | 數值 |
|---|---|
| OS / RAM | win32，51.4GB（free ~12GB） |
| GPU | RTX 4070 Ti Super（16GB，Ada SM89，FP8 可用） |
| ComfyUI | 0.30.0（frontend 1.47.12），comfy-kitchen 0.2.26、comfy-aimdo 0.4.11 |
| Python / torch | 3.13.11（Anaconda）/ torch 2.9.1+cu130 |
| 分工 | 只跑 CLIP / VAE / Ref2VA 編碼；**不載 UNET、不跑 KSampler** |
| Custom node | `M:\ComfyUI-master\custom_nodes\RemoteDenoise\`（remote_sampler.py + __init__.py） |
| 工作流 | `user/default/workflows/h3_remote_ref2va.json`（見 `examples/workflows/`） |

## 3. 通訊

- 客戶端 → llama-swap：`http://192.168.0.160:8090/upstream/minimax-h3-ref2va`
- 直連備援：`http://192.168.0.160:8299`（5203 為 llama-swap 內部 proxy 埠）
- 取消：客戶端 Cancel → `POST /interrupt` → server thread pool 的 callback 拋 `InterruptProcessingException` → HTTP 499
- payload：`torch.save` bytes（latent/cond/參數），單次約 96–187MB

## 4. 已知限制

- **無 P2P**：`x.to(對面)` 都是 GPU→RAM→GPU。換 3090 開滿 ReBAR 後 NCCL 自動升級為 P2P，現有程式不用改。
- **0.6MP 是 240s/step**：attention `O(n²)` 主導，TP 切 head 救不了；下一步 sequence parallel 見 `plan.md`。
- **權重 20GB > 客戶端 16GB**：0.6MP 只有雙卡 3080 這條路。
- 層間「使用率一起掉」是 offload H2D 同步點，屬結構性，非卡間輪流。
