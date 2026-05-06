import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.compile import compile_with_eager_fallback


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()
        self._compiled = False

    def enable_compile(self):
        if self._compiled: return
        self.forward = compile_with_eager_fallback(self.forward, "SiluAndMul.forward")
        self._compiled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
