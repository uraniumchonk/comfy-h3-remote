#!/usr/bin/env python3
"""M0 profiler: measure where one denoise step's time actually goes.

Runs a synthetic 50-block forward with a long packed sequence (mimics 0.6MP
video-ref) and times, per block:
  - H2D weight upload (place_block)
  - x broadcast (NCCL) + attn reduce (NCCL all-reduce)
  - compute (attn + mlp on both cards)

Usage (GPU box, after unloading H3):
  PYTHONPATH=/home/thomas2018/comfy_h3_server \
    /home/thomas2018/comfy_h3_server/.venv/bin/python3 profile_comm.py [seq_len]
"""
import os
import sys
import time

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h3_tp import (  # noqa: E402
    apply_tp,
    _nccl_broadcast,
    _nccl_allreduce,
    place_block,
    report_vram,
)

SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
HIDDEN = 5376
N_BLOCKS = 50


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    from comfy.sd import load_diffusion_model

    WEIGHT = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    )
    devices = ("cuda:0", "cuda:1")
    print(f"seq={SEQ} hidden={HIDDEN} blocks={N_BLOCKS}", flush=True)
    print("free", [round(torch.cuda.mem_get_info(i)[0] / 1e9, 2) for i in range(2)], flush=True)

    t0 = time.perf_counter()
    patcher = load_diffusion_model(
        WEIGHT,
        model_options={"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")},
    )
    dit = patcher.model.diffusion_model
    print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    apply_tp(dit, devices)
    print(f"apply_tp: {time.perf_counter() - t1:.1f}s", flush=True)
    print(report_vram(), flush=True)

    x = torch.randn(SEQ, HIDDEN, device=devices[0], dtype=torch.bfloat16)
    adaln = dit.blocks[0].adaln_proj
    t_dim = adaln.linear.in_features
    t_emb = torch.randn(2, t_dim, device=devices[0], dtype=torch.bfloat16)
    segs = [(0, SEQ, 0)]
    rope = None

    # Warmup NCCL + first block
    y = x
    for i in range(2):
        y = dit.blocks[i](y, t_emb, segs, rope, {})

    # --- per-block breakdown ---
    acc = {"h2d": 0.0, "bcast": 0.0, "reduce": 0.0, "compute": 0.0, "total": 0.0}
    for i, block in enumerate(dit.blocks):
        # H2D: place current block (pinned RAM -> VRAM)
        place_block(block, "cpu")
        torch.cuda.synchronize()
        _, h2d = timed(place_block, block, devices)
        # bcast x
        _, bc = timed(_nccl_broadcast, x, devices)
        # compute: block forward on a copied x
        _, comp = timed(lambda: block(x.to(devices[0]), t_emb, segs, rope, {}))
        # reduce: all-reduce a hidden-sized tensor
        parts = [torch.randn(SEQ, HIDDEN, device=d, dtype=torch.bfloat16) for d in devices]
        _, rd = timed(_nccl_allreduce, parts)
        acc["h2d"] += h2d
        acc["bcast"] += bc
        acc["compute"] += comp
        acc["reduce"] += rd
        acc["total"] += h2d + bc + comp + rd
        if (i + 1) % 10 == 0:
            print(f"  block {i+1:2d}/50  h2d={h2d*1000:.1f}ms bcast={bc*1000:.1f}ms "
                  f"compute={comp*1000:.1f}ms reduce={rd*1000:.1f}ms", flush=True)

    print("\n=== per-step extrapolation (CFG=2: two forwards/step) ===", flush=True)
    for k in ("h2d", "bcast", "reduce", "compute", "total"):
        per_step = acc[k] * 2  # positive + negative
        print(f"  {k:8s}: {acc[k]:7.2f}s/forward  {per_step:7.2f}s/step "
              f"({per_step / max(1, acc['total'] * 2) * 100:4.1f}%)", flush=True)
    print(report_vram(), flush=True)


if __name__ == "__main__":
    main()
