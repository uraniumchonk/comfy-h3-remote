# MiniMax H3 Remote Denoise - ComfyUI Custom Node
# Client side: sends denoise job to remote server (192.168.0.160)
# Decode: llama-swap minimax-h3-vae-decode; mailbox Submit/Collect for cross-queue

from .remote_sampler import (
    RemoteDenoiseSampler,
    RemoteDenoiseNode,
    H3LoraStack,
    RemoteDecodeNode,
    RemoteDecodeSubmit,
    RemoteDecodeCollect,
    RemoteDecodeGet,
)

NODE_CLASS_MAPPINGS = {
    "RemoteDenoiseSampler": RemoteDenoiseSampler,
    "RemoteDenoiseNode": RemoteDenoiseNode,
    "RemoteDecodeNode": RemoteDecodeNode,
    "RemoteDecodeSubmit": RemoteDecodeSubmit,
    "RemoteDecodeCollect": RemoteDecodeCollect,
    "RemoteDecodeGet": RemoteDecodeGet,
    "H3LoraStack": H3LoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteDenoiseSampler": "Remote Denoise Sampler (H3)",
    "RemoteDenoiseNode": "Remote Denoise Node (H3)",
    "RemoteDecodeNode": "Remote Decode Node (H3)",
    "RemoteDecodeSubmit": "Remote Decode Submit (H3)",
    "RemoteDecodeCollect": "Remote Decode Collect (H3)",
    "RemoteDecodeGet": "Remote Decode Collect (H3)",
    "H3LoraStack": "H3 LoRA Stack",
}
