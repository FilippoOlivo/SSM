import warnings
import torch
from torch.nn import SiLU
from .s4_base_block import S4BaseBlock
from .s4_diagonal_block import S4DBlock
from .s4_low_rank_block import S4LowRankBlock
from .s6_block import S6Block


class MambaBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        expansion_factor: int,
        kernel_size: int,
        normalization: bool = False,
        ssm_type: str = "S4",
        **kwargs,
    ):
        """
        MambaBlock is a neural network block that combines S4 and S6 architectures.

        This implementation is based on the paper on the original
        Mamba architecture, proposed in https://github.com/state-spaces/mamba

        :param input_dim: The input dimension.
        :param hidden_dim: The hidden dimension.
        """

        super().__init__()

        if "input_dim" in kwargs:
            warnings.warn(
                "input_dim is determined by the input_net, ignoring kwargs['input_dim']"
            )
            kwargs.pop("input_dim")
        kwargs["input_dim"] = expansion_factor * input_dim
        if "hid_dim" in kwargs:
            warnings.warn(
                "hidden_dim is determined by the input_net, ignoring kwargs['hidden_dim']"
            )
            kwargs.pop("hidden_dim")
        kwargs["hid_dim"] = expansion_factor * input_dim

        self.input_net = torch.nn.Linear(
            input_dim, expansion_factor * input_dim
        )
        self.input_net_res = torch.nn.Linear(
            input_dim, expansion_factor * input_dim
        )
        self.output_net = torch.nn.Linear(
            expansion_factor * input_dim, input_dim
        )
        self.ssm = self._initialize_ssm_block(ssm_type, **kwargs)
        self.silu = SiLU()
        self.conv1d = torch.nn.Conv1d(
            in_channels=expansion_factor * input_dim,
            out_channels=expansion_factor * input_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
        )
        if normalization:
            self.norm = torch.nn.LayerNorm(
                expansion_factor * input_dim, elementwise_affine=False
            )
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        l = x.shape[1]
        x, x_res = self.input_net(x), self.silu(self.input_net_res(x))
        x = self.conv1d(x.transpose(1, 2))[:, :, :l].transpose(1, 2)
        x = self.silu(x)
        x = self.ssm(x)
        x = x + x_res
        if self.norm is not None:
            x = self.norm(x)
        x = self.output_net(x)
        return x

    def _initialize_ssm_block(self, ssm_type: str, **kwargs):
        """
        Initialize the SSM block based on the specified type.
        """
        if ssm_type == "S4":
            return S4BaseBlock(**kwargs)
        elif ssm_type == "S4D":
            return S4DBlock(**kwargs)
        elif ssm_type == "S4LowRank":
            return S4LowRankBlock(**kwargs)
        elif ssm_type == "S6":
            return S6Block(**kwargs)
        else:
            raise ValueError(f"Unknown SSM type: {ssm_type}")
