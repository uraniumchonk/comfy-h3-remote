# Scenario B — local denoise, remote CLIP + VAE

Local GPU keeps sampling. Remote box encodes CLIP and decodes video/audio. Two graphs.

場景 B：本地狂 denoise，遠端狂跑 CLIP 與 VAE。兩張圖。

## 1. Prefill — `h3_async_prefill.json`

Queue once. Sends one CLIP encode job into the mailbox. The payload can be dummy.

Queue 一次。把一筆 CLIP encode 丟進信箱。內容可以是墊檔。

After this returns, start the main loop immediately. Do not wait for encode to finish; Collect will block if the job is still running.

送出後就可以立刻開主循環。不必等 encode 做完；Collect 若還在跑會自己等。

## 2. First official round — `h3_ref2va_async_pub.json`

```
Encode Collect (prefill) → Denoise → Decode Submit (this latent)
                         → VHS uses local / switch path if mailbox audio is False
                         → Encode Submit (next CLIP)
```

First VAE Collect is empty (`audio = False`). Your graph already switches that case. Decode Submit stores this round for the **next** start.

第一輪 VAE Collect 是空的（`audio = False`）。工作流裡的 Switch 會接住。Decode Submit 把這一輪留給**下一輪開頭**。

## 3. Every later round

At the **start** of the queue:

```
Encode Collect → next denoise
VAE Collect    → previous video + audio → VHS
```

At the **end**:

```
Decode Submit  → this denoise, for next start
Encode Submit  → next CLIP, for next start
```

每一輪**開始**抓上一輪 VAE 存片，**結束**再丟這一輪 decode / 下一輪 encode。denoise 那段時間 160 可以同時解上一支、編下一支。

Delete the dummy first clip if Prefill was junk.

Prefill 若是墊檔，第一支廢片刪掉即可。

## Wiring notes / 接線

| Node | Do / 做什麼 |
|---|---|
| Encode Submit | Fire-and-forget CLIP. `trigger` after denoise so it does not fight DiT. |
| Encode Collect | Wait if mailbox still running. Error only if truly empty. |
| Decode Submit | Send AV latent. `trigger` out is the same latent (passthrough). |
| Decode Collect | `trigger` is order only. `audio` is `False` when empty. Grab at round start. |

`cfg` must be **1.0**. H3 is flow-matching.
