import pytest
import torch
from ssm.model import Transformer

x = torch.randn(20, 25, 4)
y = torch.randn(20, 25, 4)

hid_dim = 10
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.1


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_transformer_constructor(heads, activation):

    Transformer(
        model_dim=x.shape[2],
        hidden_dim=hid_dim,
        heads=heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        activation=activation,
    )

    # Invalid number of heads
    with pytest.raises(ValueError):
        Transformer(
            model_dim=x.shape[2],
            hidden_dim=hid_dim,
            heads=3,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            activation=activation,
        )


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("src_mask", [None, torch.ones(25, 25)])
@pytest.mark.parametrize("tgt_mask", [None, torch.ones(25, 25)])
@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_transformer_forward(heads, src_mask, tgt_mask, activation):

    model = Transformer(
        model_dim=x.shape[2],
        hidden_dim=hid_dim,
        heads=heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        activation=activation,
    )

    output_ = model(x, y, src_mask=src_mask, tgt_mask=tgt_mask)
    assert output_.shape == x.shape


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("src_mask", [None, torch.ones(25, 25)])
@pytest.mark.parametrize("tgt_mask", [None, torch.ones(25, 25)])
@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_transformer_backward(heads, src_mask, tgt_mask, activation):

    model = Transformer(
        model_dim=x.shape[2],
        hidden_dim=hid_dim,
        heads=heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        activation=activation,
    )

    output_ = model(x.requires_grad_(), y, src_mask=src_mask, tgt_mask=tgt_mask)
    _ = torch.mean(output_).backward()
    assert x.grad.shape == x.shape
