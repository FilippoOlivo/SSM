import torch
import pytest
from ssm.model import S4

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("method", ["recurrent", "fourier"])
@pytest.mark.parametrize("hippo", [True, False])
def test_s4_constructor(method, hippo):
    model = S4(
        method=method,
        input_dim=5,
        output_dim=5,
        hidden_dim=10,
        hippo=hippo,
    )
    assert model.block.hidden_dim == 10
    assert model.block.input_dim == 5
    assert model.mixing_fc.in_features == 5
    assert model.mixing_fc.out_features == 5
    assert model.block.A.shape == (5, 10, 10)
    assert model.block.B.shape == (5, 10, 1)
    assert model.block.C.shape == (5, 1, 10)


@pytest.mark.parametrize("method", ["recurrent", "fourier"])
@pytest.mark.parametrize("hippo", [True, False])
def test_s4_forward(method, hippo):
    model = S4(
        method=method,
        input_dim=5,
        output_dim=5,
        hidden_dim=10,
        hippo=hippo,
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("method", ["recurrent", "fourier"])
@pytest.mark.parametrize("hippo", [True, False])
def test_s4_backrward(method, hippo):
    model = S4(
        method=method,
        input_dim=5,
        output_dim=5,
        hidden_dim=10,
        hippo=hippo,
    )
    y = model.forward(x)
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape
