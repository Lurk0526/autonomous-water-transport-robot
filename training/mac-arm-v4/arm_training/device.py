"""Explicit policy devices; native MuJoCo physics always remains on the CPU."""
import os
import platform


def resolve_device(requested='cpu'):
    import torch
    if requested not in ('cpu','mps'):
        raise ValueError('Policy device must be cpu or mps; auto selection is intentionally disabled.')
    if requested=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':
        raise RuntimeError('MPS 对照与训练禁止静默 CPU fallback。请先 unset PYTORCH_ENABLE_MPS_FALLBACK，再重新启动命令。')
    if requested=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('已指定 MPS，但当前 PyTorch/macOS 的 MPS 不可用。请在 Mac 检查原生 arm64 环境和 torch.backends.mps.is_available()；不会静默切回 CPU。')
    return torch.device(requested)


def device_record(requested='cpu',actual=None):
    import torch
    return {'requested_policy_device':requested,
        'actual_policy_device':str(actual) if actual is not None else None,
        'physics_device':'cpu','mps_built':bool(torch.backends.mps.is_built()),
        'mps_available':bool(torch.backends.mps.is_available()),
        'torch_version':torch.__version__,'platform':platform.platform(),'machine':platform.machine(),
        'PYTORCH_ENABLE_MPS_FALLBACK':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        'mps_fallback':'not enabled by this package; an explicit MPS failure is reported'}


def synchronize(device):
    """Use at wall-clock measurement boundaries, never at every physics substep."""
    if str(device).split(':',1)[0]=='mps':
        import torch
        torch.mps.synchronize()
