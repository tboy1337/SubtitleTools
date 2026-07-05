"""PyInstaller hook: package torch for CPU inference only."""

from PyInstaller.utils.hooks import collect_dynamic_libs, excludedimports

# Distributed training, TensorBoard, CUDA, and TorchInductor are not used at runtime.
excludedimports = [
    "tensorboard",
    "torch._inductor",
    "torch.cuda",
    "torch.distributed",
    "torch.distributed.algorithms",
    "torch.distributed.autograd",
    "torch.distributed.checkpoint",
    "torch.distributed.elastic",
    "torch.distributed.fsdp",
    "torch.distributed.launch",
    "torch.distributed.nn",
    "torch.distributed.optim",
    "torch.distributed.pipelining",
    "torch.distributed.rpc",
    "torch.distributed.tensor",
    "torch.utils.tensorboard",
]

binaries = collect_dynamic_libs("torch")
