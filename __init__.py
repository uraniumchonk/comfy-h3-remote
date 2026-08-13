# MiniMax H3 Remote Denoise - ComfyUI Custom Node
# Client side: sends denoise job to remote server (192.168.0.160)

from .remote_sampler import RemoteDenoiseSampler, RemoteDenoiseNode

NODE_CLASS_MAPPINGS = {
    "RemoteDenoiseSampler": RemoteDenoiseSampler,
    "RemoteDenoiseNode": RemoteDenoiseNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteDenoiseSampler": "Remote Denoise Sampler (H3)",
    "RemoteDenoiseNode": "Remote Denoise Node (H3)",
}