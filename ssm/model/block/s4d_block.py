import torch
from torch.nn.functional import pad
from ...utils import compute_hippo_diagonal


class S4DBlock(torch.nn.Module):

    def __new__(cls, method, **kwargs):
        """
        TODO
        """
        instance = super().__new__(cls)
        if method == "recurrent":
            instance.forward = instance.forward_recurrent
        elif method == "convolutional":
            instance.forward = instance.forward_convolutional
        else:
            raise ValueError(f"Unknown method: {method}")
        return instance

    def __init__(
        self,
        method,
        input_dim,
        hidden_dim,
        dt=0.1,
        hippo=False,
        discretization="bilinear",
        complex=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dt = dt
        dtype = torch.complex64 if complex else torch.float32

        if hippo:
            A = compute_hippo_diagonal(hidden_dim, complex=complex)
            A = A.unsqueeze(0).expand(input_dim, -1).clone()
        else:
            A = torch.rand(input_dim, hidden_dim, dtype=dtype)

        self.A = torch.nn.Parameter(A)
        self.B = torch.nn.Parameter(
            torch.rand(input_dim, hidden_dim, dtype=dtype)
        )
        self.C = torch.nn.Parameter(
            torch.rand(input_dim, hidden_dim, dtype=dtype)
        )

        self.register_buffer(
            "A_bar", torch.zeros(input_dim, hidden_dim, dtype=dtype)
        )
        self.register_buffer(
            "B_bar", torch.zeros(input_dim, hidden_dim, dtype=dtype)
        )

        if discretization == "bilinear":
            self.discretize = self._discretize_bilinear
        elif discretization == "zoh":
            self.discretize = self._discretize_zoh
        else:
            raise ValueError(f"Unknown discretization method: {discretization}")

    def _discretize_bilinear(self):
        tmp = 1 + self.A * self.dt / 2
        tmp2 = 1 - self.A * self.dt / 2
        self.A_bar = 1 / tmp2 * tmp
        self.B_bar = 1 / tmp2 * self.B * self.dt

    def _discretize_zoh(self):
        # Apply element-wise exponentiation
        self.A_bar = torch.exp(self.A * self.dt)
        # Compute the B_bar matrix
        self.B_bar = 1 / self.A_bar * (self.A_bar - 1) * self.B * self.dt

    def vandermonde_matrix(self, L):
        exponents = torch.arange(L)
        V = self.A_bar.unsqueeze(-1) ** exponents
        return V

    def compute_K(self, L):
        V = self.vandermonde_matrix(L)
        S = self.B_bar * self.C
        return torch.bmm(S.unsqueeze(1), V).squeeze(1).real

    def forward_recurrent(self, x):
        self.discretize()
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Discretize dynamics for all channels at once
        self.discretize()

        h = torch.zeros(
            batch_size, self.input_dim, self.hidden_dim
        )  # [B, input_dim, hidden_dim]

        # [B, L, input_dim]
        y = torch.zeros((*x.shape[:-1], self.input_dim), dtype=x.dtype)

        # Iterate over time steps
        for t in range(seq_len):
            x_t = x[:, t, :]  # Extract the input at time t
            h = self.A_bar.unsqueeze(0) * h + self.B_bar.unsqueeze(
                0
            ) * x_t.unsqueeze(2)
            y[:, t, :] = torch.sum(h * self.C.unsqueeze(0), dim=-1).real
        return y

    def forward_convolutional(self, x):
        # Dicretize the A and B matrices
        self.discretize()
        # Store the dimensions
        B, L, D = x.shape
        K = self.compute_K(L)

        # Reshape input to [B, D, L]
        x_reshaped = x.transpose(1, 2)

        # Compute
        fft_length = 2 * L

        # Pad input and kernel to avoid circular convolution effects
        # the total length of the padded sequence is 2 * L
        x_padded = pad(x_reshaped, (0, L))
        K_padded = pad(K, (0, L))

        # Compute FFT of input and kernel
        x_fft = torch.fft.rfft(x_padded, dim=2)
        K_fft = torch.fft.rfft(K_padded, dim=1)

        # Element-wise multiplication in frequency domain
        K_fft = K_fft.unsqueeze(0)  # [1, input_dim, total_length//2+1]
        y_fft = x_fft * K_fft  # [B, input_dim, total_length//2+1]

        # Inverse FFT -> [B, input_dim, fft_length]
        y = torch.fft.irfft(y_fft, n=fft_length, dim=2)

        # Cut the output to the original length -> [B, L, input_dim]
        y = y[:, :, :L]
        # Transpose the output to the original shape -> [B, L, input_dim]
        return y.transpose(1, 2)
