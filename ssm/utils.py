import torch


def compute_hippo(N):
    """
    Definition of the HIPPO initialization for the hidden-to-hidden matrix.

    :param int N: The shape of the matrix.
    :return: A :math:`(N, N)` matrix initialized with the HIPPO method.
    :rtype: torch.Tensor
    """
    P = torch.sqrt(torch.arange(1, 2 * N, 2, dtype=torch.float32))
    A = 0.5 * (P[:, None] * P[None, :])
    A = torch.tril(A, diagonal=-1) - torch.diag(
        torch.arange(N, dtype=torch.float32)
    )
    return -A


def compute_dplr(A):
    """
    TODO
    """
    # Compute p and q in a vectorized manner
    N = A.shape[0]
    indices = torch.arange(1, N + 1, dtype=torch.float32)
    p = 0.5 * torch.sqrt(2 * indices + 1.0)
    q = 2 * p
    # Construct S efficiently
    S = A + p[:, None] * q[None, :]
    Lambda, V = torch.linalg.eig(S)
    Vc = V.conj().T
    p, q = p.to(Vc.dtype), q.to(Vc.dtype)
    p = Vc @ p
    q = Vc @ q
    return Lambda, p, q
