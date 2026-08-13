"""Bypass LoRA for H3 TP: INT8 GEMM stays intact, add s*(x @ A.T) @ B.T.

Native Comfy LoraLoader patches Linear.weight. After apply_tp those
weights are emptied, so patches would no-op. Turbo LoRAs on disk are
PEFT-style rank-64 A/B on qkv/out/fc1/fc2 (50 DiT blocks + 2-layer
token_refiner). Column shards slice B on out; row shards slice A on in.
"""
from __future__ import annotations

import os

import torch

from h3_tp import qkv_col_index, row_index, swiglu_col_index

_LORA_CACHE = {}


def _load_file(path):
    path = os.path.abspath(path)
    hit = _LORA_CACHE.get(path)
    if hit is not None:
        return hit
    from safetensors.torch import load_file
    sd = load_file(path, device="cpu")
    _LORA_CACHE[path] = sd
    return sd


def _pairs(sd):
    """{module_path: (A[rank,in], B[out,rank])} from PEFT keys."""
    out = {}
    for k, t in sd.items():
        if not k.endswith(".lora_A.weight"):
            continue
        base = k[: -len(".lora_A.weight")]
        bk = base + ".lora_B.weight"
        if bk not in sd:
            continue
        name = base[len("diffusion_model.") :] if base.startswith("diffusion_model.") else base
        out[name] = (t.contiguous(), sd[bk].contiguous())
    return out


def _add_lora(y, x, A, B, strength):
    if strength == 0:
        return y
    A = A.to(device=y.device, dtype=y.dtype, non_blocking=True)
    B = B.to(device=y.device, dtype=y.dtype, non_blocking=True)
    return y + strength * (x.to(dtype=y.dtype) @ A.t()) @ B.t()


def _wrap_linear(linear, A, B, strength):
    if getattr(linear, "_h3_lora_orig", None) is None:
        orig = linear.forward
        linear._h3_lora_orig = orig
        linear._h3_lora = []

        def fwd(x, *args, **kwargs):
            y = orig(x, *args, **kwargs)
            for Aa, Bb, s in linear._h3_lora:
                y = _add_lora(y, x, Aa, Bb, s)
            return y

        linear.forward = fwd
    linear._h3_lora.append((A, B, float(strength)))


def _attach_shards(shards, A, B, strength, kind, world):
    """kind: qkv | out | fc1 | fc2."""
    for rank, shard in enumerate(shards):
        if kind == "qkv":
            idx = qkv_col_index(rank, world)
            Aa, Bb = A, B.index_select(0, idx)
        elif kind == "fc1":
            idx = swiglu_col_index(rank, world)
            Aa, Bb = A, B.index_select(0, idx)
        elif kind == "out":
            idx = row_index(rank, world, A.shape[1])
            Aa, Bb = A.index_select(1, idx), B
        elif kind == "fc2":
            idx = row_index(rank, world, A.shape[1])
            Aa, Bb = A.index_select(1, idx), B
        else:
            raise ValueError(kind)
        if getattr(shard, "_h3_lora", None) is None:
            shard._h3_lora = []
        shard._h3_lora.append((Aa.contiguous(), Bb.contiguous(), float(strength)))


def clear_loras(dit):
    for block in dit.blocks:
        for shards in (
            getattr(block.attn, "_h3_qkv_shards", ()),
            getattr(block.attn, "_h3_out_shards", ()),
            getattr(block.mlp, "_h3_fc1_shards", ()),
            getattr(block.mlp, "_h3_fc2_shards", ()),
        ):
            for s in shards:
                s._h3_lora = []
    for mod in dit.modules():
        if getattr(mod, "_h3_lora_orig", None) is not None:
            mod.forward = mod._h3_lora_orig
            mod._h3_lora_orig = None
            mod._h3_lora = []


def apply_loras(dit, specs, devices=("cuda:0", "cuda:1")):
    """specs: list of {path, strength}. Replaces any previous LoRA set."""
    clear_loras(dit)
    if not specs:
        return 0
    world = len(devices)
    n = 0
    for spec in specs:
        path = spec.get("path") or spec.get("name")
        strength = float(spec.get("strength", 1.0))
        if not path or strength == 0:
            continue
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        pairs = _pairs(_load_file(path))
        for name, (A, B) in pairs.items():
            if name.startswith("blocks."):
                _, idx, kind, proj = name.split(".", 3)  # blocks.N.attn.qkv_proj
                block = dit.blocks[int(idx)]
                if kind == "attn" and proj == "qkv_proj":
                    _attach_shards(block.attn._h3_qkv_shards, A, B, strength, "qkv", world)
                elif kind == "attn" and proj == "out_proj":
                    _attach_shards(block.attn._h3_out_shards, A, B, strength, "out", world)
                elif kind == "mlp" and proj == "fc1":
                    _attach_shards(block.mlp._h3_fc1_shards, A, B, strength, "fc1", world)
                elif kind == "mlp" and proj == "fc2":
                    _attach_shards(block.mlp._h3_fc2_shards, A, B, strength, "fc2", world)
                n += 1
            else:
                # token_refiner.blocks.N.attn.qkv_proj — unsharded Linear
                parts = name.split(".")
                mod = dit
                for p in parts:
                    mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
                _wrap_linear(mod, A, B, strength)
                n += 1
    print(f"[h3-lora] applied {n} modules from {len(specs)} file(s)", flush=True)
    return n
