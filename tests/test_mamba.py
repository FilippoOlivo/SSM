import pytest
import torch
from ssm.model.mamba import Mamba


@pytest.mark.parametrize("n_layers", [1, 4])
def test_constructor(n_layers):
    Mamba(
        n_layers=n_layers,
        input_dim=5,
        expansion_factor=2,
        kernel_size=3,
        ssm_type="S4D",
        method="convolutional",
    )


def test_wrong_constructor():
    with pytest.raises(TypeError):
        Mamba(
            n_layers=1,
        )


@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize(
    "ssm_type", ["S4", "S4D", "S4LowRank"]
)  # TODO : add S6 when ready
def test_forward(normalization, ssm_type):
    model = Mamba(
        n_layers=1,
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
    model = Mamba(
        n_layers=1,
        input_dim=5,
        expansion_factor=2,
        kernel_size=3,
        ssm_type=ssm_type,
        normalization=normalization,
        method="convolutional",
    )
    x = torch.randn(15, 10, 5)
    y = model(x)
    x.requires_grad = True
    y = model(x)
    l = torch.mean(y)
    l.backward()
    assert x.grad.shape == x.shape
