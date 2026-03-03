import torch
from torch import Tensor

@torch.no_grad()
def chol_qr(Y_bf16, eye_fp32, base_eps=1e-5, max_eps=10.0, max_tries=6, use_eigh_last=True):
    """
    Perform a numerically robust Cholesky–QR orthonormalization on a batch of
    bfloat16 matrices.

    This function:
    - Forms the Gram matrix G = Y^H Y (symmetrized to avoid Hermitian drift)
    - Adds an adaptive diagonal regularization eps * scale * I to improve SPD-ness
    - Tries Cholesky factorization with exponentially increased eps
    - Optionally falls back to eigen-decomposition if Cholesky keeps failing
    - Finally falls back to standard QR as a last resort
    """
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
    """
    Orthonormalize the columns of Y_bf16 using either:
    - "chol": Cholesky–QR with adaptive regularization (fast, memory friendly)
    - "house": Standard Householder QR (more stable but heavier)
    """
    if orth == "chol":
        return chol_qr(Y_bf16, eye_q_fp32, base_eps, max_eps, max_tries, use_eigh_last)
    return householder_qr(Y_bf16)

@torch.no_grad()
def randomized_svd_bf16(
    tensor_reshaped: Tensor,
    rank: int,
    n_iter: int = 4,
    oversample: int = 0,
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

    High‑level description:
    - Optionally transpose wide matrices so that m >= n for numerical stability
    - Work either in pure fp32 ("fp32" path) or mostly in bf16 ("bf16" path)
    - Build a low‑dimensional random subspace Q that approximates the range of A
    - Perform subspace power iteration to improve spectral separation
    - Project A to a small matrix B = Q^H A, SVD B, and lift back to U, Vh
    - Aggressively delete intermediates to reduce peak memory usage
    """
    # ------------------------------------------------------------------
    # 1) Basic shape handling and possible wide‑matrix transpose
    # ------------------------------------------------------------------
    m, n = tensor_reshaped.shape[-2:]
    batch_shape = tensor_reshaped.shape[:-2]

    # If the matrix is "wide" (more columns than rows), we transpose so that
    # the effective operator has m >= n. This usually improves numerical
    # behavior and matches what PyTorch does internally for some decompositions.
    is_wide = m < n

    # ------------------------------------------------------------------
    # 2) Choose working precision for power iteration and projection
    # ------------------------------------------------------------------
    # fp32 path:
    #   - keep the full input in float32
    #   - never quantize to bfloat16
    #   - this matches the behavior / accuracy of torch.svd_lowrank
    #
    # bf16 path:
    #   - quantize the input once to bfloat16
    #   - perform power iteration and projection in bfloat16
    #   - saves memory and bandwidth, good for very large problems
    if power_dtype == "fp32":
        X_pow = tensor_reshaped.float()
        if is_wide:
            X_pow = X_pow.mH
            if M is not None:
                M = M.mH
            m, n = n, m
        X_bf16 = None  # not used in fp32 path
    else:
        X_bf16 = tensor_reshaped.to(torch.bfloat16)
        if is_wide:
            X_bf16 = X_bf16.mH
            if M is not None:
                M = M.mH
            m, n = n, m
        X_pow = X_bf16

    # Target rank (k) is clipped by the effective matrix dimensions.
    k = min(rank, m, n)
    # q_actual is the dimension of the random subspace; oversampling helps
    # capture slightly more of the spectrum to improve approximation quality.
    q_actual = min(k + oversample, m, n)

    # Optional mean / baseline tensor M that will be subtracted out everywhere.
    # We maintain a version in the same dtype as X_pow, and (when needed) a
    # bf16 version synchronized with X_bf16.
    if M is not None:
        M_pow = M.to(X_pow.dtype).broadcast_to(X_pow.shape)
        M_bf16 = None if power_dtype == "fp32" else M.to(torch.bfloat16).broadcast_to(X_bf16.shape)
    else:
        M_pow, M_bf16 = None, None

    # Pre‑allocate an identity matrix in float32 for the orthonormalization step.
    eye_q = torch.eye(q_actual, device=tensor_reshaped.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 3) Initial random subspace: Y = (A - M) * R
    # ------------------------------------------------------------------
    R_rand = torch.randn(*batch_shape, n, q_actual, device=tensor_reshaped.device, dtype=X_pow.dtype)
    Y = torch.matmul(X_pow, R_rand)
    if M_pow is not None:
        Y = Y - torch.matmul(M_pow, R_rand)
    del R_rand

    # For fp32 path we keep Y in float32 all the way through orthonormalization
    # so that we closely match torch.svd_lowrank behavior.
    # For bf16 path we cast to bfloat16 to save memory/compute.
    Y_for_orth = Y if power_dtype == "fp32" else Y.to(torch.bfloat16)
    Q = orthonormalize(Y_for_orth, eye_q, orth, base_eps, max_eps, max_tries, use_eigh_last)
    del Y

    # ------------------------------------------------------------------
    # 4) Prepare transposed operator for power iteration
    # ------------------------------------------------------------------
    X_t = X_pow.mH
    M_t = M_pow.mH if M_pow is not None else None

    # ------------------------------------------------------------------
    # 5) Power iteration (a.k.a. subspace iteration)
    #    Repeatedly apply A^H A to refine Q and separate singular values.
    # ------------------------------------------------------------------
    for _ in range(n_iter):
        # Y = (A^H - M^H) * Q
        Y = torch.matmul(X_t, Q.to(X_pow.dtype))
        if M_t is not None:
            Y = Y - torch.matmul(M_t, Q.to(X_pow.dtype))
        del Q

        # Re‑orthonormalize Y to get an updated Q
        Y_for_orth = Y if power_dtype == "fp32" else Y.to(torch.bfloat16)
        Q = orthonormalize(Y_for_orth, eye_q, orth, base_eps, max_eps, max_tries, use_eigh_last)
        del Y

        # Y = (A - M) * Q
        Y = torch.matmul(X_pow, Q.to(X_pow.dtype))
        if M_pow is not None:
            Y = Y - torch.matmul(M_pow, Q.to(X_pow.dtype))
        del Q

        # Re‑orthonormalize again to keep the subspace well‑conditioned
        Y_for_orth = Y if power_dtype == "fp32" else Y.to(torch.bfloat16)
        Q = orthonormalize(Y_for_orth, eye_q, orth, base_eps, max_eps, max_tries, use_eigh_last)
        del Y

    # ------------------------------------------------------------------
    # 6) Projection: B = Q^H (A - M)
    # ------------------------------------------------------------------
    X_proj = X_pow if power_dtype == "fp32" else X_bf16
    M_proj = M_pow if power_dtype == "fp32" else M_bf16
    Q_proj = Q.to(X_proj.dtype)  # match dtype of the projected operator
    Bproj = torch.matmul(Q_proj.mH, X_proj)
    if M_proj is not None:
        Bproj = Bproj - torch.matmul(Q_proj.mH, M_proj)

    # ------------------------------------------------------------------
    # 7) Small SVD on B
    # ------------------------------------------------------------------
    U_small, S, Vh_small = torch.linalg.svd(Bproj.float(), full_matrices=False)
    del Bproj

    # ------------------------------------------------------------------
    # 8) Lift back to tall singular vectors and cast dtypes
    # ------------------------------------------------------------------
    # fp32 path: keep all in fp32 (no bf16),
    # bf16 path: cast to bf16 for downstream usage.
    U_tall = torch.matmul(Q, U_small.to(Q.dtype))
    Vh_tall = Vh_small.to(Q.dtype)
    S = S.to(Q.dtype)
    if power_dtype != "fp32":
        U_tall = U_tall.to(torch.bfloat16)
        Vh_tall = Vh_tall.to(torch.bfloat16)
        S = S.to(torch.bfloat16)
    del U_small, Vh_small

    # ------------------------------------------------------------------
    # 9) Aggressive cleanup of intermediates to free memory early
    # ------------------------------------------------------------------
    del Q, eye_q, X_t, X_pow
    if X_bf16 is not None:
        del X_bf16
    if M_pow is not None:
        del M_pow, M_t
    if M_bf16 is not None:
        del M_bf16

    # --- Final shape matching and truncation ---
    # At this point U_tall, S, Vh_tall correspond to the possibly transposed
    # operator. We:
    # - truncate to the requested rank
    # - undo the "wide" transpose if we flipped the operator at the beginning
    r = min(rank, S.shape[-1])
    
    if is_wide:
        U_orig = Vh_tall.mH
        Vh_orig = U_tall.mH
        return U_orig[..., :, :r], S[..., :r], Vh_orig[..., :r, :]
    else:
        return U_tall[..., :, :r], S[..., :r], Vh_tall[..., :r, :]