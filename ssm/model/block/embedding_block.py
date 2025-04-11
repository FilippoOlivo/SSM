import torch


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, model, embedding_dim):
        """
        Initialize the embedding block.

        :param int input_dim: Dimension of the input.
        :param int output_dim: Dimension of the output.
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(embedding_dim, model.input_dim)
        self.model = model

    def forward(self, x):
        """
        Forward pass of the embedding block.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        x = self.embedding(x)
        x = self.model(x)
        return x
