import torch
from .s4_block_interface import S4BlockInterface
from ...utils import (
    compute_S4DInv,
    compute_S4DLin,
    compute_S4DQuad,
    compute_S4DReal,
)

class S4DBlock(S4BlockInterface):
    r"""
    Implementation of the diagonal S4 block.

    This block is a variant of the S4 block that uses a diagonal matrix for the
    hidden-to-hidden dynamics. It is designed to simplify both the logic and
    implementation of the S4 block while maintaining the same functionality.

    This block supports two forward pass methods: recurrent, and convolutional.

    - **Recurrent**: It applies discretized dynamics for sequential processing.
    - **Convolutional**: It uses the Fourier transform to compute convolutions.

    The block is defined by the following equations:

    .. math::
        \dot{h}(t) = Ah(t) + Bx(t),
        y(t) = Ch(t),

    where :math:`h(t)` is the hidden state, :math:`x(t)` is the input,
    :math:`y(t)` is the output, :math:`A`is a hidden-to-hidden diagonal matrix,
    :math:`B` is the input-to-hidden matrix, and :math:`C` is the
    hidden-to-output matrix.

    .. seealso::
        **Original Reference**: Gu, A., Gupta, A., Goel, K., and Re, G. (2022).
        "On the Parameterization and Initialization of Diagonal State Space
        Models".
        arXiv:2206.11893.
        DOI: `<https://arxiv.org/pdf/2206.11893>_`.
    """

    def __init__(
        self,
        input_dim,
        hid_dim,
        dt=0.1,
        initialization="S4D-Inv",
        discretization="bilinear",
    ):
        """
        Initialization of the diagonal S4 block.

        :param int input_dim: The input dimension.
        :param int hid_dim: The hidden state dimension.
        :param float dt: The time step for discretization.
        :param str initialization: The method for initializing the A matrix.
            Options are: S4D-Inv, S4D-Lin, S4D-Quad, S4D-Real, real, complex.
        :param str discretization: The method for discretizing the dynamics.
            Options are: bilinear, zoh.
        """
        super().__init__(input_dim=input_dim, hid_dim=hid_dim, dt=dt)

        A = self.initialize_A(hid_dim, init_method=initialization)
        A = torch.nn.Parameter(A.unsqueeze(0).repeat(input_dim, 1))

        # COmpute B_bar and C_bar matrices
        if discretization == "bilinear":
            self.discretize = self._discretize_bilinear
        elif discretization == "zoh":
            self.discretize = self._discretize_zoh
        else:
            raise ValueError(f"Unknown discretization method: {discretization}")

    def _discretize_bilinear(self):
        """
        Discretization of the continuous-time dynamics to obtain the matrices
        :math:`A_{bar}` and :math:`B_{bar}`.
        """
        self._discretize()

    def _discretize_zoh(self):
        """
        Discretization of the continuous-time dynamics to obtain the matrices
        :math:`A_{bar}` and :math:`B_{bar}`.
        """
        self.A_bar = torch.exp(self.A * self.dt)
        self.B_bar = 1 / self.A_bar * (self.A_bar - 1) * self.B * self.dt

    def vandermonde_matrix(self, L):
        """
        Compute the Vandermonde matrix for the diagonal S4 block.

        :param int L: The length of the sequence.
        :return: The Vandermonde matrix.
        :rtype: torch.Tensor
        """
        exponents = torch.arange(L, device=self.A_bar.device)
        V = self.A_bar.unsqueeze(-1) ** exponents
        return V

    def compute_K(self, L):
        """
        Computation of the kernel K used in the convolutional method.
        """
        # Compute the Vandermonde matrix
        V = self.vandermonde_matrix(L)

        # Compute the kernel K using the Vandermonde matrix
        S = self.B_bar * self.C
        return torch.bmm(S.unsqueeze(1), V).squeeze(1).real
    

    @staticmethod
    def initialize_A(hid_dim, init_method="S4D-Inv"):
        """
        Initialization of the A matrix.

        :param int hid_dim: The hidden state dimension.
        :param str init_method: The method for initializing the A matrix.
            Options are: S4D-Inv, S4D-Lin, S4D-Quad, S4D-Real, real, complex.
        :return: The initialized A matrix.
        :rtype: torch.Tensor
        :raises ValueError: If an unknown initialization method is provided.
        """
        if init_method == "S4D-Inv":
            return compute_S4DInv(hid_dim)

        elif init_method == "S4D-Lin":
            return compute_S4DLin(hid_dim)

        elif init_method == "S4D-Quad":
            return compute_S4DQuad(hid_dim)

        elif init_method == "S4D-Real":
            return compute_S4DReal(hid_dim)

        elif init_method == "real":
            return torch.rand(hid_dim)

        elif init_method == "complex":
            return torch.rand(hid_dim) + 1j * torch.rand(hid_dim)

        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
