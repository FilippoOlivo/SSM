import torch
import torch.nn as nn
from torch.nn.functional import pad
import torch.nn.functional as F
from ...utils import compute_hippo, compute_dplr


class S4AdvandedBlock(nn.Module):
    """
    S4 block with recurrent/sequential computation.

    :param int hidden_dim: Dimension of the hidden state.
    :param float dt: Time step.
    :param bool hippo: Whether to use the HIPPO matrix.
    :param bool fixed: Whether to fix the parameters.
    """

    def __new__(cls, method, hidden_dim, L, dt=0.1):
        instance = super().__new__(cls)
        if not isinstance(method, str):
            raise ValueError("Method must be a string.")
        if not isinstance(hidden_dim, int):
            raise ValueError("Hidden dimension must be an integer.")
        if not isinstance(dt, float):
            raise ValueError("Time step must be a float.")

        if method == "convolutional":
            instance.forward = instance.forward_convolutional
        elif method == "recurrent":
            instance.forward = instance.forward_recurrent
        else:
            raise ValueError(f"Unknown method: {method}")
        return instance

    def __init__(
        self,
        hidden_dim: int,
        L: int,  # Sequence length
        method: str = "convolutional",
        dt: float = 0.1,
    ):
        super().__init__()
        A = compute_hippo(hidden_dim)
        # Define parameters
        self.B = torch.nn.Parameter(torch.rand(hidden_dim))
        self.C_tilde = torch.nn.Parameter(torch.rand(hidden_dim))
        self.Lambda, self.P, self.Q = compute_dplr(A)
        if method == "convolutional":
            self.register_buffer("omega", self._init_omega(L))
        elif method == "recurrent":
            self.register_buffer("A_bar", torch.zeros(hidden_dim, hidden_dim))
            self.register_buffer("B_bar", torch.zeros(hidden_dim, 1))
        self.dt = torch.tensor([dt])
        self.L = L
        self.hidden_dim = hidden_dim

    @staticmethod
    def _init_omega(L):
        return torch.exp(2j * torch.pi * torch.arange(L) / L)  # (L,)

    def _cauchy_dot(self, a0, a1, b0, b1, denominator):
        k00 = (a0 * b0 / denominator).sum(-1)
        k01 = (a0 * b1 / denominator).sum(-1)
        k10 = (a1 * b0 / denominator).sum(-1)
        k11 = (a1 * b1 / denominator).sum(-1)
        return k00, k01, k10, k11

    def compute_K(self):
        a0, a1 = self.C_tilde.conj(), self.Q.conj()
        b0, b1 = self.B, self.P
        g = (2.0 / self.dt) * (
            (1.0 - self.omega) / (1.0 + self.omega)
        ).unsqueeze(-1)
        denominator = g - self.Lambda
        k00, k01, k10, k11 = self._cauchy_dot(a0, a1, b0, b1, denominator)
        c = 2.0 / (1.0 + self.omega)
        K_hat = c * (k00 - k01 * (1.0 + k11) * k10)
        return torch.fft.irfft(K_hat, n=self.L).unsqueeze(-1).unsqueeze(-1)

    def forward_convolutional(self, x):
        K = self.compute_K()
        # Apply zero-padding to avoid circular convolution effects
        x_padded = pad(x, (0, 0, 0, self.L - 1))

        # Compute the convolution using the Fourier transform
        x_fft = torch.fft.rfft(x_padded, dim=1)
        K_fft = torch.fft.rfft(
            pad(K, (0, 0, 0, 0, 0, x_padded.shape[1] - K.shape[0])), dim=0
        )
        y_fft = torch.einsum("bfi,foi->bfo", x_fft, K_fft)

        # Compute the inverse Fourier transform
        y = torch.fft.irfft(y_fft, n=x_padded.shape[1], dim=1)[:, : self.L, :]
        return y

    def discretize(self):
        # Compute discretized matrices using bilinear transform
        P = self.P.unsqueeze(-1)
        Q = self.Q.unsqueeze(-1)
        I = torch.eye(self.hidden_dim)
        D = torch.diag(2 / (self.dt - self.Lambda))  # D term

        # A0 (Forward Euler)
        A0 = I + (self.dt / 2) * (torch.diag(self.Lambda) - P @ Q.T)

        identity_rank = torch.eye(P.shape[1])
        woodbury_inv = torch.inverse(identity_rank + Q.T @ D @ P)
        A1 = D - D @ P @ woodbury_inv @ Q.T @ D

        A = A1 @ A0
        B = self.B.to(A1.dtype)
        B = 2 * A1 @ B
        print(A.shape)
        return A, B.unsqueeze(-1)

    def forward_recurrent(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        sequence_length = x.shape[1]
        x = x.permute(1, 0, 2)

        # Initialize y and h
        y = torch.empty(x.shape[0], x.shape[1], 1, dtype=torch.complex64)
        h = torch.empty(
            x.shape[0] + 1, x.shape[1], self.hidden_dim, dtype=torch.complex64
        )

        # Discretize the continuous-time dynamics
        A, B = self.discretize()
        C = self.C_tilde.unsqueeze(0).to(torch.complex64)
        # Compute the hidden states and the output
        h[0] = torch.zeros(x.shape[1], self.hidden_dim)
        for t in range(sequence_length):
            xt = x[t].to(A.dtype)
            h[t + 1] = torch.einsum("ij,bj->bi", A, h[t]) + torch.einsum(
                "ij,bj->bi", B, xt
            )
            y[t] = torch.einsum("ij,bj->bi", C, h[t + 1])
        return y
