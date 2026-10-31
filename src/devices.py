import os

import torch

DEVICE_HELP = "cpu, cuda, or cuda:N; bare cuda uses LOCAL_RANK when set (torchrun)"


def resolve_device(name: str) -> torch.device:
    """Resolve a CLI device name to a concrete device and make a CUDA choice the current device.

    Bare "cuda" selects the LOCAL_RANK GPU when launched by torchrun and GPU 0 otherwise;
    an explicit "cuda:N" always wins. CUDA is probed here, never at import time.
    """
    device = torch.device(name)
    if device.type == "cpu":
        return device
    if device.type != "cuda":
        raise ValueError(f"Unsupported device {name!r}; use cpu, cuda, or cuda:N")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this command on the GPU machine or pass --device cpu.")
    index = device.index if device.index is not None else int(os.environ.get("LOCAL_RANK", "0"))
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise ValueError(f"CUDA device {index} requested, but only {count} visible")
    torch.cuda.set_device(index)
    return torch.device("cuda", index)
