# MiniMax H3 Remote Denoise - ComfyUI Custom Node
# Client side: sends denoise job to remote server (192.168.0.160)
# Decode: sends denoised latent to 160 video VAE decode service (port 8300)

from .remote_sampler import RemoteDenoiseSampler, RemoteDenoiseNode, H3LoraStack, RemoteDecodeNode

NODE_CLASS_MAPPINGS = {
    "RemoteDenoiseSampler": RemoteDenoiseSampler,
    "RemoteDenoiseNode": RemoteDenoiseNode,
    "RemoteDecodeNode": RemoteDecodeNode,
    "H3LoraStack": H3LoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteDenoiseSampler": "Remote Denoise Sampler (H3)",
    "RemoteDenoiseNode": "Remote Denoise Node (H3)",
    "RemoteDecodeNode": "Remote Decode Node (H3)",
    "H3LoraStack": "H3 LoRA Stack",
}
