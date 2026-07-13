"""Shared GPU-availability check used by every ML-backed engine
(EasyOCR, Stanza, PaddleOCR) so they all use the GPU automatically when
one is present, without any engine having to duplicate this check or
crash if torch/CUDA isn't set up correctly.
"""

_gpu_available_cache = None


def gpu_available() -> bool:
    """Return True if a CUDA GPU is available to PyTorch.

    Cached after the first call since this check is cheap but not free,
    and the answer can't change during the life of the process.
    """
    global _gpu_available_cache

    if _gpu_available_cache is not None:
        return _gpu_available_cache

    try:
        import torch
        _gpu_available_cache = bool(torch.cuda.is_available())
    except Exception:
        _gpu_available_cache = False

    return _gpu_available_cache
