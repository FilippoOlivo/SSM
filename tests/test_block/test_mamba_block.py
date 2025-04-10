import pytest
import torch
from ssm.model.block import MambaBlock


@pytest.mark.parametrize(
    "ssm_type", ["S4", "S4D", "S4LowRank"]
)  # TODO : add S6 when ready
def test_constructor(ssm_type):
    MambaBlock(
        input_dim=5,
        expansion_factor=2,
        kernel_size=3,
        ssm_type=ssm_type,
        method="convolutional",
    )


def test_wrong_constructor():
    with pytest.raises(ValueError):
        MambaBlock(
            input_dim=5,
            expansion_factor=2,
            kernel_size=3,
            ssm_type="wrong_ssm_type",
            method="convolutional",
        )


@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize(
    "ssm_type", ["S4", "S4D", "S4LowRank"]
)  # TODO : add S6 when ready
def test_forward(normalization, ssm_type):
    model = MambaBlock(
        input_dim=5,
        expansion_factor=2,
        kernel_size=3,
        ssm_type=ssm_type,
        normalization=normalization,
        method="convolutional",
    )
    x = torch.randn(15, 10, 5)
    y = model(x)
    assert y.shape == (15, 10, 5)


@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize(
    "ssm_type", ["S4", "S4D", "S4LowRank"]
)  # TODO : add S6 when ready
def test_backward(ssm_type, normalization):
    model = MambaBlock(
        input_dim=5,
        expansion_factor=2,
        kernel_size=3,
        ssm_type=ssm_type,
        normalization=normalization,
        method="convolutional",
    )
    x = torch.randn(15, 10, 5, requires_grad=True)
    y = model(x)
    loss = torch.mean(y)
    loss.backward()
    assert x.grad.shape == x.shape
