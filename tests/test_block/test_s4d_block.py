import pytest
import torch
from ssm.model.block import S4DBlock

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("method", ["recurrent", "fourier"])
@pytest.mark.parametrize("hippo", [True, False])
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
def test_s4d_constructor(method, hippo, complex, discretisation):
    model = S4DBlock(
        method=method,
        hidden_dim=10,
        hippo=hippo,
        input_dim=5,
        complex=complex,
        discretization=discretisation,
    )
    assert model.A.shape == (5, 10)
    assert model.B.shape == (5, 10)
    assert model.C.shape == (5, 10)
    model.discretize()
    assert model.A_bar.shape == (5, 10)
    assert model.B_bar.shape == (5, 10)


@pytest.mark.parametrize("hippo", [True, False])
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
def test_s4d_recurrent_forward(hippo, complex, discretisation):
    model = S4DBlock(
        method="recurrent",
        hidden_dim=10,
        hippo=hippo,
        input_dim=5,
        complex=complex,
        discretization=discretisation,
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
def test_s4d_fourier_backward(hippo, complex, discretisation):
    model = S4DBlock(
        method="recurrent",
        hidden_dim=10,
        hippo=hippo,
        input_dim=5,
        complex=complex,
        discretization=discretisation,
    )
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape


@pytest.mark.parametrize("hippo", [True, False])
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
def test_s4d_fourier_forward(hippo, complex, discretisation):
    model = S4DBlock(
        method="fourier",
        hidden_dim=10,
        hippo=hippo,
        input_dim=5,
        complex=complex,
        discretization=discretisation,
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
def test_s4_fourier_backward(hippo, complex, discretisation):
    model = S4DBlock(
        method="fourier",
        hidden_dim=10,
        hippo=hippo,
        input_dim=5,
        complex=complex,
        discretization=discretisation,
    )
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape
