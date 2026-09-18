"""Optional PyTorch adapter for one linear writer; no full-model loader.

Targets must be verified as the exact output of a linear writer. A transformer
block output or normalized/gated output is NOT an interchangeable location.
"""
from __future__ import annotations
from contextlib import contextmanager
from .linear import DirectionPatch


@contextmanager
def output_patch(module, patch: DirectionPatch):
    import torch
    cache = {}

    def hook(_module, _args, out):
        if not isinstance(out, torch.Tensor) or out.shape[-1] != patch.output_dim:
            raise ValueError("hook expected a tensor with the writer output dimension")
        key = (out.device, out.dtype)
        if key not in cache:
            cache[key] = (torch.tensor(patch.directions.copy(), device=out.device, dtype=torch.float32),
                          torch.tensor(patch.strengths.copy(), device=out.device, dtype=torch.float32))
        d, a = cache[key]
        # This patches the bias contribution too, when bias is present.
        y = out.float()
        return (y - ((y @ d) * a) @ d.T).to(out.dtype)
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
        cache.clear()
