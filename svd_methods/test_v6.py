import torch
from torch import Tensor

@torch.no_grad()
def chol_qr(Y_bf16, eye_fp32, base_eps=1e-5, max_eps=10.0, max_tries=6, use_eigh_last=True):
    Y = Y_bf16.float()
    G = torch.matmul(Y.mH, Y)
    G = 0.5 * (G + G.mH)

    d = torch.diagonal(G, dim1=-2, dim2=-1)
    scale = d.mean(dim=-1, keepdim=True).clamp_min_(1e-12).unsqueeze(-1)

    eps = base_eps
    for _ in range(max_tries):
        R, info = torch.linalg.cholesky_ex(G + (eps * scale) * eye_fp32, upper=True)
        if (info == 0).all():
            Q = torch.linalg.solve_triangular(R, Y, upper=True, left=False)
            del R, info, d, scale, G, Y
            return Q.to(Y_bf16.dtype)
        del R, info
        eps = min(eps * 10.0, max_eps)

    if use_eigh_last:
        try:
            L, V = torch.linalg.eigh(G)
            L = torch.clamp(L, min=max(1e-4, eps))
            G_spd = torch.matmul(V, torch.matmul(torch.diag_embed(L), V.mH))
            R = torch.linalg.cholesky(G_spd, upper=True)
            Q = torch.linalg.solve_triangular(R, Y, upper=True, left=False)
            del L, V, G_spd, R, d, scale, G, Y
            return Q.to(Y_bf16.dtype)
        except Exception:
            pass

    Q, _ = torch.linalg.qr(Y, mode="reduced")
    del d, scale, G, Y
    return Q.to(Y_bf16.dtype)

@torch.no_grad()
def householder_qr(Y_bf16):
    Y = Y_bf16.float()
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    del Y
    return Q.to(Y_bf16.dtype)

@torch.no_grad()
def orthonormalize(Y_bf16, eye_q_fp32, orth="chol", base_eps=1e-6, max_eps=10.0, max_tries=6, use_eigh_last=True):
    if orth == "chol":
        return chol_qr(Y_bf16, eye_q_fp32, base_eps, max_eps, max_tries, use_eigh_last)
    return householder_qr(Y_bf16)

@torch.no_grad()
def randomized_svd_bf16(
    tensor_reshaped: Tensor,
    rank: int,
    n_iter: int = 2,
    oversample: int = 4,
    M: Tensor | None = None,
    base_eps: float = 1e-5,
    max_eps: float = 10.0,
    max_tries: int = 6,
    use_eigh_last: bool = True,
    power_dtype: str = "bf16",
    orth: str = "chol",
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Ultimate fused version: Randomized SVD supporting arbitrary dimensions, 
    dynamic regularization, automatic transposition, and extreme memory release.
    Function signature matches the external caller perfectly.
    """
    m, n = tensor_reshaped.shape[-2:]
    batch_shape = tensor_reshaped.shape[:-2]
    
    # Support wide matrices: if m < n, transpose first to accelerate computation and save memory
    is_wide = m < n
    
    X_bf16 = tensor_reshaped.to(torch.bfloat16)
    if is_wide:
        X_bf16 = X_bf16.mH
        if M is not None:
            M = M.mH
        m, n = n, m  # After swapping, force m >= n

    # Calculate actual subspace dimension
    k = min(rank, m, n)
    q_actual = min(k + oversample, m, n)

    X_pow = X_bf16 if power_dtype == "bf16" else X_bf16.float()

    if M is not None:
        M_pow = M.to(X_pow.dtype).broadcast_to(X_pow.shape)
        M_bf16 = M.to(torch.bfloat16).broadcast_to(X_bf16.shape)
    else:
        M_pow, M_bf16 = None, None

    eye_q = torch.eye(q_actual, device=tensor_reshaped.device, dtype=torch.float32)

    # Initial random projection
    R_rand = torch.randn(*batch_shape, n, q_actual, device=tensor_reshaped.device, dtype=X_pow.dtype)
    Y = torch.matmul(X_pow, R_rand)
    if M_pow is not None:
        Y = Y - torch.matmul(M_pow, R_rand)
    del R_rand

    Q = orthonormalize(Y.to(torch.bfloat16), eye_q, orth, base_eps, max_eps, max_tries, use_eigh_last)
    del Y

    X_t = X_pow.mH
    M_t = M_pow.mH if M_pow is not None else None

    # Power Iteration (Subspace Iteration)
    for _ in range(n_iter):
        Y = torch.matmul(X_t, Q.to(X_pow.dtype))
        if M_t is not None:
            Y = Y - torch.matmul(M_t, Q.to(X_pow.dtype))
        del Q

        Q = orthonormalize(Y.to(torch.bfloat16), eye_q, orth, base_eps, max_eps, max_tries, use_eigh_last)
        del Y

        Y = torch.matmul(X_pow, Q.to(X_pow.dtype))
        if M_pow is not None:
            Y = Y - torch.matmul(M_pow, Q.to(X_pow.dtype))
        del Q

        Q = orthonormalize(Y.to(torch.bfloat16), eye_q, orth, base_eps, max_eps, max_tries, use_eigh_last)
        del Y

    # Project back to small matrix Bproj
    Bproj = torch.matmul(Q.mH, X_bf16)
    if M_bf16 is not None:
        Bproj = Bproj - torch.matmul(Q.mH, M_bf16)

    # Small SVD calculation
    U_small, S, Vh_small = torch.linalg.svd(Bproj.float(), full_matrices=False)
    del Bproj

    U_tall = torch.matmul(Q, U_small.to(torch.bfloat16))
    Vh_tall = Vh_small.to(torch.bfloat16)
    S = S.to(torch.bfloat16)
    del U_small, Vh_small

    # Cleanup remaining tensors
    del Q, eye_q, X_t, X_pow, X_bf16
    if M_pow is not None:
        del M_pow, M_bf16, M_t

    # --- Final shape matching and truncation ---
    r = min(rank, S.shape[-1])
    
    # Compatible with PyTorch's original wide matrix reverse flip, ensuring output is always U, S, Vh
    if is_wide:
        # If the original matrix is wide: A_orig = A_tall.mH = (U S Vh).mH = Vh.mH S U.mH
        U_orig = Vh_tall.mH
        Vh_orig = U_tall.mH
        return U_orig[..., :, :r], S[..., :r], Vh_orig[..., :r, :]
    else:
        # If the original matrix is tall
        return U_tall[..., :, :r], S[..., :r], Vh_tall[..., :r, :]