import torch
import pytest
from ssm.model.block import S6Block

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
def test_s6_constructor(hippo):
    model = S6Block(hidden_dim=10, input_dim=5, hippo=hippo)
    assert model.A.shape == (5, 10)
    assert hasattr(model, "sb")
    assert hasattr(model, "sc")
    assert hasattr(model, "tau_delta")


@pytest.mark.parametrize("hippo", [True, False])
def test_s6_forward(hippo):
    model = S6Block(hidden_dim=10, input_dim=5, hippo=hippo)
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
def test_s6_backward(hippo):
    model = S6Block(hidden_dim=10, input_dim=5, hippo=hippo)
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape
