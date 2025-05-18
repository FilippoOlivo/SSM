import torch


class Transformer(torch.nn.Module):
    """
    Wrapper class for torch.nn.Transformer.
    """

    def __init__(
        self,
        model_dim,
        hidden_dim,
        heads,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        activation="gelu",
        **kwargs,
    ):
        """
        Initialization of the Transformer model.

        :param int model_dim: The input dimension.
        :param int hidden_dim: The hidden state dimension.
        :param int heads: The number of attention heads.
        :param int num_encoder_layers: Number of encoder layers. Default is 2.
        :param int num_decoder_layers: Number of decoder layers. Default is 2.
        :param float dropout: Dropout rate. Default is 0.1.
        :param str activation: The activation function. Must be one of
            `'gelu'` or `'relu'`. Default is `'gelu'`.
        :param dict kwargs: Additional keyword arguments used in the model.
        :raises ValueError: If the number of heads is not a divisor of the
            input dimension.
        """
        super().__init__()

        # Check if the number of heads is a divisor of the input dimension
        if model_dim % heads != 0:
            raise ValueError(
                "The number of heads must be a divisor of the input dimension."
            )

        # Initialize the transformer model
        self.transformer = torch.nn.Transformer(
            d_model=model_dim,
            nhead=heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

    def forward(self, x, y, src_mask=None, tgt_mask=None):
        """
        Forward pass through the Transformer model with optional masks.

        :param torch.Tensor x: Input tensor (batch, seq length, dimension).
        :param torch.Tensor y: Target tensor (batch, seq length, dimension).
        :param torch.Tensor src_mask: Source mask. Default is `None`.
        :param torch.Tensor tgt_mask: Target mask. Default is `None`.
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return self.transformer(x, y, src_mask=src_mask, tgt_mask=tgt_mask)
