import torch

@torch.no_grad()
def chol_qr(Y_bf16, eye_fp32, base_eps=1e-6, max_eps=10.0, max_tries=6, use_eigh_last=True):
    """
    Robust Cholesky QR with mixed precision and eigen-spectrum correction.
    High accuracy (reconstructs G) + High speed (early exit).
    """
    Y = Y_bf16.float()
    
    # 1. Compute Gram Matrix
    G = torch.bmm(Y.transpose(1, 2), Y)
    G = 0.5 * (G + G.transpose(1, 2))  # Symmetrize

    # 2. Dynamic Scaling (Trace-based)
    # Using mean of diag is faster and stable enough for scaling
    d = torch.diagonal(G, dim1=-2, dim2=-1)
    scale = d.mean(dim=-1, keepdim=True).clamp_min_(1e-12).unsqueeze(-1) # [bs, 1, 1]

    eps = base_eps
    
    # 3. Fast Jitter Loop (Attempts to solve without expensive eig)
    for _ in range(max_tries):
        # eps * scale adapts to the magnitude of G
        R, info = torch.linalg.cholesky_ex(G + (eps * scale) * eye_fp32, upper=True)
        
        if (info == 0).all():
            return torch.linalg.solve_triangular(R, Y, upper=True, left=False).to(Y_bf16.dtype)
        
        eps = min(eps * 10.0, max_eps)

    # 4. High-Precision Fallback: Eigendecomposition Reconstruction (The "V2" Magic)
    # Only runs if jitter fails. Maintains spectral accuracy better than diagonal shift.
    if use_eigh_last:
        try:
            # eigh is necessary here to reconstruct G, not just shift eigenvalues
            L, V = torch.linalg.eigh(G) 
            
            # Key to high accuracy: Clamp small negative values, keep large ones intact
            L = torch.clamp(L, min=max(1e-4, eps)) 
            
            # Reconstruct G_spd = V * diag(L) * V^T
            G_spd = V @ (torch.diag_embed(L) @ V.transpose(1, 2))
            
            # Final Cholesky on the perfectly reconstructed matrix
            R = torch.linalg.cholesky(G_spd, upper=True)
            return torch.linalg.solve_triangular(R, Y, upper=True, left=False).to(Y_bf16.dtype)
        except Exception:
            pass # Fall through to QR

    # 5. Ultimate Fallback: Slow but safe Householder QR
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    return Q.to(Y_bf16.dtype)


@torch.no_grad()
def randomized_svd_bf16(tensor, rank, n_iter=8, oversample=4):
    """
    Randomized SVD optimized for BFloat16 and high-rank approximation.
    """
    B, M, N = tensor.shape
    k = min(rank, M, N)
    q = min(k + oversample, N)
    
    X = tensor.to(torch.bfloat16)
    XT = X.transpose(1, 2)
    
    # Pre-allocate Identity (broadcastable)
    eye = torch.eye(q, device=tensor.device, dtype=torch.float32).unsqueeze(0)

    # 1. Randomized Initialization
    Omega = torch.randn((B, N, q), device=tensor.device, dtype=torch.bfloat16)
    Y = torch.bmm(X, Omega)
    
    # 2. Power Iteration with Stabilization
    for _ in range(n_iter):
        Y = torch.bmm(X, torch.bmm(XT, Y))
        
        # Critical for BF16: Normalize to prevent overflow/underflow
        denom = torch.linalg.vector_norm(Y, dim=1, keepdim=True, dtype=torch.float32).clamp_min_(1e-8)
        Y = (Y / denom).to(torch.bfloat16)
        
        # Orthogonalize to prevent subspace collapse
        Y = chol_qr(Y, eye)

    # 3. Final Orthogonalization
    Q = chol_qr(Y, eye)

    # 4. Project and Solve Small SVD
    # B_proj = Q^T * X
    B_proj = torch.bmm(Q.transpose(1, 2), X)
    
    # Run SVD in FP32 for precision
    U_small, S, Vh = torch.linalg.svd(B_proj.float(), full_matrices=False)
    
    # 5. Reconstruct High-Dimension U
    U = torch.bmm(Q, U_small.to(torch.bfloat16))
    
    # Truncate to requested rank
    return U[:, :, :k], S[:, :k].to(torch.bfloat16), Vh[:, :k, :].to(torch.bfloat16)