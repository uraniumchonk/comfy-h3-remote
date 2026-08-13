#!/usr/bin/env python3
"""Megatron-style TP for MiniMax H3 INT8+ConvRot DiT.

qkv_proj / fc1 : column-parallel (QKV and SwiGLU index maps, not naive N/2)
out_proj / fc2 : row-parallel + all_reduce
ConvRot group=256 stays intact: row splits land on group boundaries.
"""
from __future__ import annotations

import os
import sys
import threading
import time

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")

import torch
import torch.nn as nn

WEIGHT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
)

HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM          # 7168
HIDDEN = 5376
FFN = 14336
CONVROT_GS = 256


def qkv_col_index(rank: int, world: int) -> torch.Tensor:
    local_inner = INNER // world
    parts = []
    for slot in range(3):
        start = slot * INNER + rank * local_inner
        parts.append(torch.arange(start, start + local_inner))
    return torch.cat(parts)


def swiglu_col_index(rank: int, world: int) -> torch.Tensor:
    local = FFN // world
    gate = torch.arange(rank * local, (rank + 1) * local)
    up = torch.arange(FFN + rank * local, FFN + (rank + 1) * local)
    return torch.cat([gate, up])


def row_index(rank: int, world: int, k: int) -> torch.Tensor:
    local = k // world
    if local % CONVROT_GS != 0:
        raise ValueError(f"row split {local} is not a multiple of convrot {CONVROT_GS}")
    return torch.arange(rank * local, (rank + 1) * local)


def _as_scale_vec(scale: torch.Tensor, n: int) -> torch.Tensor:
    s = scale.reshape(-1)
    if s.numel() == 1:
        return s.expand(n).contiguous()
    if s.numel() != n:
        raise ValueError(f"scale numel {s.numel()} != N {n}")
    return s


def shard_column(qdata: torch.Tensor, scale: torch.Tensor, index: torch.Tensor):
    idx = index.to(device=qdata.device)
    q = qdata.index_select(0, idx).contiguous()
    s = _as_scale_vec(scale, qdata.shape[0]).index_select(0, idx).contiguous()
    return q, s.view(-1, 1)


def shard_row(qdata: torch.Tensor, scale: torch.Tensor, index: torch.Tensor):
    idx = index.to(device=qdata.device)
    q = qdata.index_select(1, idx).contiguous()
    s = _as_scale_vec(scale, qdata.shape[0]).contiguous().view(-1, 1)
    return q, s


def int8_linear(x, qdata, scale, input_act=None):
    import comfy.quant_ops as qo
    dev = x.device
    if dev.type == "cuda":
        with torch.cuda.device(dev):
            return qo.ck.int8_linear(
                x, qdata, scale, None, x.dtype, True, CONVROT_GS, input_act,
            )
    return qo.ck.int8_linear(
        x, qdata, scale, None, x.dtype, True, CONVROT_GS, input_act,
    )


def _move_shard(shard, device):
    # Int8Shard._apply pins against dit.to(); move buffers directly.
    dest = torch.device(device)
    if shard.qdata.device == dest:
        return
    if dest.type == "cpu":
        shard._buffers["qdata"] = shard.qdata.to("cpu").pin_memory()
        shard._buffers["scale"] = shard.scale.to("cpu").pin_memory()
        return
    shard._buffers["qdata"] = shard.qdata.to(dest, non_blocking=True)
    shard._buffers["scale"] = shard.scale.to(dest, non_blocking=True)


def _block_shards(block):
    return (
        list(block.attn._h3_qkv_shards)
        + list(block.attn._h3_out_shards)
        + list(block.mlp._h3_fc1_shards)
        + list(block.mlp._h3_fc2_shards)
    )


def place_block(block, devices, streams=None):
    """devices: sequence of 2 device strings, or 'cpu'."""
    groups = (
        block.attn._h3_qkv_shards,
        block.attn._h3_out_shards,
        block.mlp._h3_fc1_shards,
        block.mlp._h3_fc2_shards,
    )
    for shards in groups:
        for rank, shard in enumerate(shards):
            dest = "cpu" if devices == "cpu" else devices[rank]
            if streams is not None and devices != "cpu":
                with torch.cuda.device(dest), torch.cuda.stream(streams[rank]):
                    _move_shard(shard, dest)
            else:
                _move_shard(shard, dest)
    extra_dev = "cpu" if devices == "cpu" else devices[0]
    if streams is not None and devices != "cpu":
        with torch.cuda.device(extra_dev), torch.cuda.stream(streams[0]):
            for m in (block.norm1, block.norm2, block.adaln_proj, block.attn.q_norm, block.attn.k_norm):
                m.to(extra_dev, non_blocking=True)
    else:
        for m in (block.norm1, block.norm2, block.adaln_proj, block.attn.q_norm, block.attn.k_norm):
            m.to(extra_dev, non_blocking=True)


def _make_streams(devices):
    streams = []
    for dev in devices:
        with torch.cuda.device(dev):
            streams.append(torch.cuda.Stream())
    return streams


def _sync_streams(streams):
    for st in streams:
        st.synchronize()


def _run_ranks(devices, fn):
    """One thread per GPU. Kitchen dlpack uses stream=-1 (default stream sync),
    so CUDA streams do not overlap; vLLM avoids this with one process per GPU."""
    world = len(devices)
    outs = [None] * world
    err = [None] * world

    def worker(rank, dev):
        try:
            with torch.cuda.device(dev):
                outs[rank] = fn(rank, dev)
                torch.cuda.synchronize(dev)
        except BaseException as e:
            err[rank] = e

    threads = [threading.Thread(target=worker, args=(r, d)) for r, d in enumerate(devices)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in err:
        if e is not None:
            raise e
    return outs


def _nccl_broadcast(x, devices):
    """Replicate x onto every rank via NCCL SHM (no P2P)."""
    x = x.contiguous()
    xs = []
    root = 0
    for i, dev in enumerate(devices):
        if x.device == torch.device(dev):
            xs.append(x)
            root = i
        else:
            xs.append(torch.empty(x.shape, dtype=x.dtype, device=dev))
    torch.cuda.nccl.broadcast(xs, root=root)
    return xs


def _nccl_allreduce(parts):
    parts = [p.contiguous() for p in parts]
    torch.cuda.nccl.all_reduce(parts)
    return parts


def warmup_nccl(devices):
    ts = [torch.zeros(8, device=d, dtype=torch.bfloat16) for d in devices]
    torch.cuda.nccl.all_reduce(ts)
    for d in devices:
        torch.cuda.synchronize(d)


def reduce_row_linear(xs, shards, out_device, streams=None):
    """vLLM row-parallel: local INT8 GEMM + NCCL all-reduce (SHM, no P2P)."""
    devices = [str(s.home) for s in shards]

    def _one(rank, dev):
        x = xs[rank]
        if x.device != torch.device(dev):
            x = x.to(dev, non_blocking=True)
        return shards[rank](x)

    parts = _run_ranks(devices, _one)
    parts = _nccl_allreduce(parts)
    for p in parts:
        if p.device == torch.device(out_device):
            return p
    return parts[0].to(out_device)


class Int8Shard(nn.Module):
    def __init__(self, qdata, scale, device, input_act=None):
        super().__init__()
        self.register_buffer("qdata", qdata.to(device=device, non_blocking=True))
        self.register_buffer("scale", scale.to(device=device, non_blocking=True))
        self.input_act = input_act
        self.home = torch.device(device)

    def _apply(self, fn, recurse=True):
        # Pin to the shard device. Comfy model_load would otherwise
        # drag every buffer onto cuda:0 and collapse TP.
        return self

    def forward(self, x):
        return int8_linear(x, self.qdata, self.scale, self.input_act)


def _qt_parts(linear):
    w = linear.weight
    return w._qdata, w._params.scale


def _replace_attn(attn, devices, streams=None):
    world = len(devices)
    if attn.heads % world != 0:
        raise ValueError(f"heads {attn.heads} not divisible by tp {world}")
    if streams is None:
        streams = _make_streams(devices)
    local_heads = attn.heads // world
    qdata, scale = _qt_parts(attn.qkv_proj)
    o_q, o_s = _qt_parts(attn.out_proj)
    qkv_shards = []
    out_shards = []
    for rank, dev in enumerate(devices):
        q, s = shard_column(qdata, scale, qkv_col_index(rank, world))
        qkv_shards.append(Int8Shard(q, s, dev))
        rq, rs = shard_row(o_q, o_s, row_index(rank, world, o_q.shape[1]))
        out_shards.append(Int8Shard(rq, rs, dev))

    orig_q_norm = attn.q_norm
    orig_k_norm = attn.k_norm
    head_dim = attn.head_dim

    def forward(x, rope_freqs=None, transformer_options={}):
        from comfy.ldm.modules.attention import (
            AttentionTensorContainer,
            optimized_attention,
        )
        import comfy.model_management
        import comfy.quant_ops

        seq = x.shape[0]
        x = x.to(dtype=torch.bfloat16)
        out_device = x.device
        xs = _nccl_broadcast(x, devices)
        rfs = None
        if rope_freqs is not None:
            if rope_freqs.device != torch.device(devices[0]):
                rope_freqs = rope_freqs.to(devices[0], non_blocking=True)
            rfs = _nccl_broadcast(rope_freqs, devices)

        def _one(rank, dev):
            xr = xs[rank]
            qkv = qkv_shards[rank](xr)
            q, k, v = qkv.split(local_heads * head_dim, dim=-1)
            v = v.view(seq, local_heads, head_dim)
            if rfs is not None:
                rf = rfs[rank]
                q = q.view(1, seq, local_heads, head_dim)
                k = k.view(1, seq, local_heads, head_dim)
                qw = comfy.model_management.cast_to(orig_q_norm.weight, device=dev)
                kw = comfy.model_management.cast_to(orig_k_norm.weight, device=dev)
                rot = rf.shape[-3] * 2
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rf, qw, kw, epsilon=orig_q_norm.eps, rot_dim=rot)
                q = q[0]
                k = k[0]
            else:
                qw = comfy.model_management.cast_to(orig_q_norm.weight, device=dev)
                kw = comfy.model_management.cast_to(orig_k_norm.weight, device=dev)
                q = torch.nn.functional.rms_norm(
                    q.view(seq, local_heads, head_dim), (head_dim,), qw, orig_q_norm.eps)
                k = torch.nn.functional.rms_norm(
                    k.view(seq, local_heads, head_dim), (head_dim,), kw, orig_k_norm.eps)
            v = v.clone()
            q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
            k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
            v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
            return optimized_attention(
                q, k, v, local_heads, mask=None, skip_reshape=True,
                transformer_options=transformer_options,
            ).squeeze(0)

        local_outs = _run_ranks(devices, _one)
        return reduce_row_linear(local_outs, out_shards, out_device, streams)

    attn.forward = forward
    attn._h3_tp = True
    attn._h3_qkv_shards = qkv_shards
    attn._h3_out_shards = out_shards
    attn.qkv_proj.weight = nn.Parameter(torch.empty(0), requires_grad=False)
    attn.out_proj.weight = nn.Parameter(torch.empty(0), requires_grad=False)


def _replace_mlp(mlp, devices, streams=None):
    world = len(devices)
    if streams is None:
        streams = _make_streams(devices)
    q1, s1 = _qt_parts(mlp.fc1)
    q2, s2 = _qt_parts(mlp.fc2)
    fc1_shards = []
    fc2_shards = []
    for rank, dev in enumerate(devices):
        q, s = shard_column(q1, s1, swiglu_col_index(rank, world))
        fc1_shards.append(Int8Shard(q, s, dev))
        rq, rs = shard_row(q2, s2, row_index(rank, world, q2.shape[1]))
        fc2_shards.append(Int8Shard(rq, rs, dev, input_act="swiglu"))

    def forward(x):
        x = x.to(dtype=torch.bfloat16)
        out_device = x.device
        xs = _nccl_broadcast(x, devices)

        def _one(rank, dev):
            return fc1_shards[rank](xs[rank])

        hs = _run_ranks(devices, _one)
        return reduce_row_linear(hs, fc2_shards, out_device, streams)

    mlp.forward = forward
    mlp._h3_tp = True
    mlp._h3_fc1_shards = fc1_shards
    mlp._h3_fc2_shards = fc2_shards
    mlp.fc1.weight = nn.Parameter(torch.empty(0), requires_grad=False)
    mlp.fc2.weight = nn.Parameter(torch.empty(0), requires_grad=False)


def apply_tp(diffusion_model, devices=("cuda:0", "cuda:1"), offload=True):
    world = len(devices)
    if HEADS % world != 0 or FFN % world != 0 or INNER % world != 0:
        raise ValueError(f"tp={world} does not divide heads/ffn")
    streams = _make_streams(devices)
    diffusion_model._h3_streams = streams
    warmup_nccl(devices)
    print("[h3-tp] NCCL SHM ready (P2P disabled, BAR1=256M)", flush=True)
    blocks = list(diffusion_model.blocks)
    for block in blocks:
        _replace_attn(block.attn, devices, streams)
        _replace_mlp(block.mlp, devices, streams)
    if not offload:
        return world

    for block in blocks:
        place_block(block, "cpu")
    place_block(blocks[0], devices)

    n = len(blocks)
    copy_streams = _make_streams(devices)
    diffusion_model._h3_copy_streams = copy_streams
    for i, block in enumerate(blocks):
        orig = block.forward

        def _make(idx, blk, orig_fwd):
            def wrapped(*args, **kwargs):
                _sync_streams(copy_streams)
                place_block(blk, devices)
                nxt = blocks[idx + 1] if idx + 1 < n else blocks[0]
                if nxt is not blk:
                    place_block(nxt, devices, copy_streams)
                try:
                    return orig_fwd(*args, **kwargs)
                finally:
                    _sync_streams(streams)
                    place_block(blk, "cpu")
            return wrapped

        block.forward = _make(i, block, orig)
    return world


def _max_err(a, b):
    d = (a.detach().float() - b.detach().float()).abs()
    denom = b.detach().float().abs().amax().clamp_min(1e-6)
    return float(d.amax()), float(d.amax() / denom)


def _load_layer(f, prefix):
    w = f.get_tensor(f"{prefix}.weight")
    s = f.get_tensor(f"{prefix}.weight_scale")
    return w, s


def verify_layer(name, qdata, scale, kind, devices, seq=64):
    """kind: 'qkv' | 'swiglu' | 'row'."""
    world = len(devices)
    n, k = qdata.shape
    x = torch.randn(seq, k, device=devices[0], dtype=torch.bfloat16)
    full_w = qdata.to(devices[0], non_blocking=True)
    full_s = scale.to(devices[0], non_blocking=True)
    ref = int8_linear(x, full_w, full_s)

    if kind == "qkv":
        parts = []
        for rank, dev in enumerate(devices):
            idx = qkv_col_index(rank, world)
            qw, qs = shard_column(qdata, scale, idx)
            y = int8_linear(x.to(dev), qw.to(dev), qs.to(dev))
            parts.append(y.to(devices[0]))
        local = INNER // world
        qs, ks, vs = [], [], []
        for p in parts:
            q, k_, v = p.split(local, dim=-1)
            qs.append(q)
            ks.append(k_)
            vs.append(v)
        got = torch.cat([torch.cat(qs, -1), torch.cat(ks, -1), torch.cat(vs, -1)], -1)
        sliced = torch.cat(
            [ref.index_select(-1, qkv_col_index(r, world).to(ref.device)) for r in range(world)],
            -1,
        )
        return {
            "vs_full": _max_err(got, ref),
            "vs_sliced_cat": _max_err(torch.cat(parts, -1), sliced),
        }

    if kind == "swiglu":
        parts = []
        for rank, dev in enumerate(devices):
            idx = swiglu_col_index(rank, world)
            qw, qs = shard_column(qdata, scale, idx)
            y = int8_linear(x.to(dev), qw.to(dev), qs.to(dev))
            parts.append(y.to(devices[0]))
        local = FFN // world
        gates, ups = [], []
        for p in parts:
            g, u = p.split(local, dim=-1)
            gates.append(g)
            ups.append(u)
        got = torch.cat([torch.cat(gates, -1), torch.cat(ups, -1)], -1)
        return {"vs_full": _max_err(got, ref)}

    xs = []
    shards = []
    for rank, dev in enumerate(devices):
        idx = row_index(rank, world, k)
        qw, qs_ = shard_row(qdata, scale, idx)
        xs.append(x.index_select(-1, idx.to(x.device)).to(dev))
        shards.append(Int8Shard(qw, qs_, dev))
    got = reduce_row_linear(xs, shards, devices[0])
    return {"vs_full": _max_err(got, ref)}


def _make_qt(qdata, scale):
    from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout
    params = TensorWiseINT8Layout.Params(
        scale=scale.float().contiguous(),
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(qdata.shape),
        is_weight=True,
        convrot=True,
        convrot_groupsize=CONVROT_GS,
    )
    return QuantizedTensor(qdata.contiguous(), "TensorWiseINT8Layout", params)


def verify_block(f, devices, seq=32):
    from comfy.ldm.minimax.model import Attention, MLP
    import comfy.ops

    ops = comfy.ops.disable_weight_init
    attn = Attention(HIDDEN, HEADS, HEAD_DIM, 1e-6, dtype=torch.bfloat16, device="cpu", operations=ops)
    mlp = MLP(HIDDEN, FFN, dtype=torch.bfloat16, device="cpu", operations=ops)
    pairs = (
        (attn.qkv_proj, "blocks.0.attn.qkv_proj"),
        (attn.out_proj, "blocks.0.attn.out_proj"),
        (mlp.fc1, "blocks.0.mlp.fc1"),
        (mlp.fc2, "blocks.0.mlp.fc2"),
    )
    for linear, prefix in pairs:
        w, s = _load_layer(f, prefix)
        linear.weight = torch.nn.Parameter(_make_qt(w, s), requires_grad=False)
        linear.bias = None

    x = torch.randn(seq, HIDDEN, device=devices[0], dtype=torch.bfloat16)
    attn.to(devices[0])
    with torch.cuda.device(devices[0]):
        y_attn = attn(x)
    _replace_attn(attn, devices)
    y_attn_tp = attn(x)
    print("attn block", _max_err(y_attn_tp, y_attn), flush=True)

    mlp.to(devices[0])
    with torch.cuda.device(devices[0]):
        y_mlp = mlp(x)
    _replace_mlp(mlp, devices)
    y_mlp_tp = mlp(x)
    print("mlp block", _max_err(y_mlp_tp, y_mlp), flush=True)


def report_vram():
    lines = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        free_b, total_b = torch.cuda.mem_get_info(i)
        lines.append(
            f"cuda:{i} alloc={alloc:.2f}G reserved={reserved:.2f}G "
            f"free={(free_b/1e9):.2f}/{total_b/1e9:.2f}G"
        )
    return "\n".join(lines)


def verify_full(devices=("cuda:0", "cuda:1"), seq=64):
    from comfy.sd import load_diffusion_model

    print("load full model on CPU", WEIGHT, flush=True)
    t0 = time.time()
    patcher = load_diffusion_model(
        WEIGHT,
        model_options={"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")},
    )
    dit = patcher.model.diffusion_model
    print(f"loaded in {time.time()-t0:.1f}s blocks={len(dit.blocks)}", flush=True)

    t1 = time.time()
    world = apply_tp(dit, devices)
    dit.to(devices[0])
    torch.cuda.synchronize()
    print(f"TP{world} applied in {time.time()-t1:.1f}s", flush=True)
    print(report_vram(), flush=True)

    adaln = dit.blocks[0].adaln_proj
    t_dim = adaln.linear.in_features
    x = torch.randn(seq, HIDDEN, device=devices[0], dtype=torch.bfloat16)
    t_emb = torch.randn(2, t_dim, device=devices[0], dtype=torch.bfloat16)
    segs = [(0, seq, 0)]

    t2 = time.time()
    y = x
    for i, block in enumerate(dit.blocks):
        y = block(y, t_emb, segs, None, {})
        if i == 0 or i + 1 == len(dit.blocks) or (i + 1) % 10 == 0:
            print(
                f"block {i} shape={tuple(y.shape)} finite={bool(torch.isfinite(y).all())} "
                f"amax={float(y.detach().float().abs().amax()):.5g}",
                flush=True,
            )
    torch.cuda.synchronize()
    elapsed = time.time() - t2
    print(f"50-block forward {elapsed:.2f}s ({elapsed/len(dit.blocks)*1000:.1f} ms/block)", flush=True)
    print(report_vram(), flush=True)
    if not torch.isfinite(y).all():
        raise SystemExit("NaN/Inf in final hidden")
    print("FULL OK", flush=True)


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    if torch.cuda.device_count() < 2:
        raise SystemExit("need 2 GPUs")

    devices = ("cuda:0", "cuda:1")
    print("free", [torch.cuda.mem_get_info(i)[0] / 1e9 for i in range(2)], flush=True)

    if "--full" in sys.argv:
        verify_full(devices)
        return

    from safetensors import safe_open

    layers = {
        "qkv": ("blocks.0.attn.qkv_proj", "qkv"),
        "fc1": ("blocks.0.mlp.fc1", "swiglu"),
        "out": ("blocks.0.attn.out_proj", "row"),
        "fc2": ("blocks.0.mlp.fc2", "row"),
    }

    t0 = time.time()
    with safe_open(WEIGHT, framework="pt", device="cpu") as f:
        for tag, (prefix, kind) in layers.items():
            w, s = _load_layer(f, prefix)
            print(f"{tag} {tuple(w.shape)} scale {tuple(s.shape)} {w.dtype}", flush=True)
            stats = verify_layer(tag, w, s, kind, devices)
            for k, (amax, rel) in stats.items():
                print(f"  {k}: amax={amax:.5g} rel={rel:.5g}", flush=True)
            del w, s
            torch.cuda.empty_cache()
        print("block-level", flush=True)
        verify_block(f, devices)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
