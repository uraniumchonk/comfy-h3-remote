"""Decode DP worker. No torch import at module level.

Parent spawn-imports this file; CUDA_VISIBLE_DEVICES must be set
before torch is imported so Kitchen RoPE always sees cuda:0.
"""
import os
import sys


def main(rank, video_path, in_q, out_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import comfy.utils
    from comfy.sd import VAE

    sd, metadata = comfy.utils.load_torch_file(video_path, return_metadata=True)
    model = VAE(sd=sd, metadata=metadata).first_stage_model
    model = model.to("cuda:0", dtype=torch.float16).eval()
    out_q.put(("ready", rank))
    while True:
        job = in_q.get()
        if job is None:
            break
        i, clip_z = job
        clip_z = clip_z.to("cuda:0", dtype=torch.float16)
        with torch.no_grad():
            dec = model._adaptive_decode(clip_z).to("cpu")
        out_q.put((i, dec))
