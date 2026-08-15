# MiniMax H3 Remote Denoise - ComfyUI Custom Node
# Client side: sends denoise job to remote server (192.168.0.160)
# Decode: sends denoised latent to 160 video VAE decode service (llama-swap)

from .remote_sampler import (
    RemoteDenoiseSampler,
    RemoteDenoiseNode,
    H3LoraStack,
    RemoteDecodeNode,
    RemoteDecodeSubmit,
    RemoteDecodeGet,
)

NODE_CLASS_MAPPINGS = {
    "RemoteDenoiseSampler": RemoteDenoiseSampler,
    "RemoteDenoiseNode": RemoteDenoiseNode,
    "RemoteDecodeNode": RemoteDecodeNode,
    "RemoteDecodeSubmit": RemoteDecodeSubmit,
    "RemoteDecodeGet": RemoteDecodeGet,
    "H3LoraStack": H3LoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteDenoiseSampler": "Remote Denoise Sampler (H3)",
    "RemoteDenoiseNode": "Remote Denoise Node (H3)",
    "RemoteDecodeNode": "Remote Decode Node (H3)",
    "RemoteDecodeSubmit": "Remote Decode Submit (H3)",
    "RemoteDecodeGet": "Remote Decode Get (H3)",
    "H3LoraStack": "H3 LoRA Stack",
}
