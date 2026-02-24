import torch

@torch.no_grad()
def chol_qr(Y_bf16, eye_fp32, base_eps=1e-6, max_eps=10.0, max_tries=6, use_eigh_last=True):
    """
    Orthonormalize a tall-skinny matrix Y using Cholesky decomposition.
    Includes an iterative ridge regularization and an EIGH fallback for stability.
    """
    # Convert to FP32 for numerical stability in Gram matrix calculation
    Y = Y_bf16.float()
    G = torch.bmm(Y.transpose(1, 2), Y)
    
    # Force symmetry to avoid small numerical asymmetries
    G = 0.5 * (G + G.transpose(1, 2))
    
    # Dynamic scaling based on the average diagonal value of the Gram matrix
    d = torch.diagonal(G, dim1=-2, dim2=-1)
    scale = d.mean(dim=-1, keepdim=True).clamp_min_(1e-12).unsqueeze(-1)
    
    eps = base_eps
    for _ in range(max_tries):
        # Attempt Cholesky decomposition with increasing regularization
        R, info = torch.linalg.cholesky_ex(G + (eps * scale) * eye_fp32, upper=True)
        if (info == 0).all():
            Q = torch.linalg.solve_triangular(R, Y, upper=True, left=False)
            del R, G, Y # Immediate memory release
            return Q.to(Y_bf16.dtype)
        eps = min(eps * 10.0, max_eps)
    
    # Fallback 1: Spectral decomposition (EIGH) if Cholesky fails multiple times
    if use_eigh_last:
        try:
            L, V = torch.linalg.eigh(G)
            # Clip eigenvalues to ensure positive definiteness
            L = torch.clamp(L, min=max(1e-4, eps))
            G_spd = V @ (torch.diag_embed(L) @ V.transpose(1, 2))
            R = torch.linalg.cholesky(G_spd, upper=True)
            Q = torch.linalg.solve_triangular(R, Y, upper=True, left=False)
            del L, V, G_spd, R, G, Y
            return Q.to(Y_bf16.dtype)
        except Exception:
            pass
            
    # Fallback 2: Standard Householder QR (Slowest but most robust)
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    del G, Y
    return Q.to(Y_bf16.dtype)

@torch.no_grad()
def randomized_svd_bf16_cholqr_v5(
    A,
    rank,
    n_iter=4,
    oversample=4,
    base_eps=1e-6,
    max_eps=10.0,
    max_tries=6,
    use_eigh_last=True,
):
    """
    Memory-efficient Randomized SVD using Cholesky QR and Power Iteration.
    Performs subspace iteration with QR at every step to prevent singular value decay.
    """
    B_sz, m, n = A.shape
    k = min(rank, m, n)
    q = min(k + oversample, m, n)

    # Cast to BF16 for faster GEMM operations
    X = A.to(torch.bfloat16)
    
    # Pre-allocate identity matrix for CholQR regularization
    eye_q = torch.eye(q, device=A.device, dtype=torch.float32).unsqueeze(0)

    # Initial random projection: Y = A * R
    R_rand = torch.randn(B_sz, n, q, device=A.device, dtype=torch.bfloat16)
    Y = torch.bmm(X, R_rand)
    del R_rand # Clear random matrix immediately
    
    # Initial orthonormalization
    Q = chol_qr(Y, eye_q, base_eps, max_eps, max_tries, use_eigh_last)
    del Y 

    # Power Iteration: (A * A^T)^n_iter * Q
    X_t = X.transpose(1, 2) # View, no extra memory cost
    for _ in range(n_iter):
        # Project to A^T: Y = A^T * Q
        Y = torch.bmm(X_t, Q)
        del Q
        Q = chol_qr(Y, eye_q, base_eps, max_eps, max_tries, use_eigh_last)
        del Y
        
        # Project to A: Y = A * Q
        Y = torch.bmm(X, Q)
        del Q
        Q = chol_qr(Y, eye_q, base_eps, max_eps, max_tries, use_eigh_last)
        del Y

    # Construct the low-rank approximation in the projected subspace: B = Q^T * A
    Bproj = torch.bmm(Q.transpose(1, 2), X)
    
    # Compute SVD of the small Bproj matrix in FP32
    U_small, S, Vh = torch.linalg.svd(Bproj.float(), full_matrices=False)
    del Bproj 
    
    # Back-project U to the original space: U = Q * U_small
    U = torch.bmm(Q, U_small.to(torch.bfloat16))
    del Q, U_small, eye_q 

    # Truncate to desired rank
    r = min(k, S.shape[-1])
    return U[:, :, :r], S[:, :r].to(torch.bfloat16), Vh[:, :r, :].to(torch.bfloat16)