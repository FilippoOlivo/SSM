import torch
from .block.mamba_block import MambaBlock


class Mamba(torch.nn.Module):
    def __init__(self, n_layers, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        mamba_blocks = torch.nn.ModuleList(
            [MambaBlock(**kwargs) for _ in range(n_layers)]
        )
        self.mamba_blocks = torch.nn.Sequential(*mamba_blocks)

    def forward(self, x):
        return self.mamba_blocks(x)
