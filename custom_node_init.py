# MiniMax H3 Remote — denoise / decode / encode client nodes

from .remote_sampler import (
    RemoteDenoiseSampler,
    RemoteDenoiseNode,
    H3LoraStack,
    RemoteDecodeNode,
    RemoteDecodeSubmit,
    RemoteDecodeCollect,
    RemoteDecodeGet,
    RemoteEncodeNode,
    RemoteEncodeSubmit,
    RemoteEncodeCollect,
)

NODE_CLASS_MAPPINGS = {
    "RemoteDenoiseSampler": RemoteDenoiseSampler,
    "RemoteDenoiseNode": RemoteDenoiseNode,
    "RemoteDecodeNode": RemoteDecodeNode,
    "RemoteDecodeSubmit": RemoteDecodeSubmit,
    "RemoteDecodeCollect": RemoteDecodeCollect,
    "RemoteDecodeGet": RemoteDecodeGet,
    "RemoteEncodeNode": RemoteEncodeNode,
    "RemoteEncodeSubmit": RemoteEncodeSubmit,
    "RemoteEncodeCollect": RemoteEncodeCollect,
    "H3LoraStack": H3LoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteDenoiseSampler": "Remote Denoise Sampler (H3)",
    "RemoteDenoiseNode": "Remote Denoise Node (H3)",
    "RemoteDecodeNode": "Remote Decode Node (H3)",
    "RemoteDecodeSubmit": "Remote Decode Submit (H3)",
    "RemoteDecodeCollect": "Remote Decode Collect (H3)",
    "RemoteDecodeGet": "Remote Decode Collect (H3)",
    "RemoteEncodeNode": "Remote Encode Node (H3)",
    "RemoteEncodeSubmit": "Remote Encode Submit (H3)",
    "RemoteEncodeCollect": "Remote Encode Collect (H3)",
    "H3LoraStack": "H3 LoRA Stack",
}
