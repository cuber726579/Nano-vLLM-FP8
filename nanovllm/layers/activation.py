import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()
        self._compiled = False

    def enable_compile(self):
        if self._compiled: return
        self.forward = torch.compile(self.forward)
        self._compiled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
