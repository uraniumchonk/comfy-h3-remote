# Remote Pipe — 跨機 encode / denoise / decode + 跨圖信箱
#
# 客戶端只跑圖；GPU 箱跑 CLIP encode、DiT denoise、VAE decode。
# 選單分類：Remote Pipe

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
    "RemoteDenoiseSampler": "Pipe Denoise Sampler",
    "RemoteDenoiseNode": "Pipe Denoise",
    "RemoteDecodeNode": "Pipe Decode (sync)",
    "RemoteDecodeSubmit": "Pipe Decode Submit",
    "RemoteDecodeCollect": "Pipe Decode Collect",
    "RemoteDecodeGet": "Pipe Decode Collect",
    "RemoteEncodeNode": "Pipe Encode (sync)",
    "RemoteEncodeSubmit": "Pipe Encode Submit",
    "RemoteEncodeCollect": "Pipe Encode Collect",
    "H3LoraStack": "Pipe LoRA Stack",
}
