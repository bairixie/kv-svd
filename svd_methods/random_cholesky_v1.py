@torch.no_grad()
def _chol_qr_hybrid(Y_bf16, reg_eps=1e-4, eye_cache=None):
    """
    Fast QR for tall-skinny matrices using Cholesky-based orthonormalization.
    Falls back to torch.linalg.qr if numerical issues are detected.
    """
    Y_fp32 = Y_bf16.to(torch.float32)
    gram = torch.bmm(Y_fp32.transpose(1, 2), Y_fp32)
    
    if eye_cache is not None:
        gram = gram + reg_eps * eye_cache
    else:
        q = Y_fp32.shape[-1]
        gram = gram + reg_eps * torch.eye(q, device=Y_fp32.device, dtype=torch.bfloat16)
    
    try:
        R = torch.linalg.cholesky(gram, upper=True)
        Q_fp32 = torch.linalg.solve_triangular(R, Y_fp32, upper=True, left=False)
    except RuntimeError:
        Q_fp32, _ = torch.linalg.qr(Y_fp32, mode="reduced")
    
    return Q_fp32.to(Y_bf16.dtype)

@torch.no_grad()
def randomized_svd_bf16_cholqr_v1(tensor_reshaped, rank, n_iter=4, oversample=4):
    """
    Randomized SVD with Cholesky QR (replacing Householder QR):
      - All large GEMM in BF16 (KV, Y, Q, B)
      - Cholesky QR in FP32 on Y (with implicit normalization)
      - SVD in FP32 on small B via _svd_small_fp32 (eager, not compiled)
    
    Key difference from svd_randomized_svd_bf16_cholqr:
      - Uses Cholesky QR instead of Householder QR
      - Cholesky QR provides implicit normalization (like Householder QR)
      - More efficient for tall-skinny matrices (q << m)
      - Automatic fallback to Householder QR if Cholesky fails
    
    Structure is identical to original houseqr version, only QR method changed.
    """
    device = tensor_reshaped.device
    bs, sl, n = tensor_reshaped.shape
    k = min(rank, sl, n)
    q = min(k + oversample, n)
    
    x2d = tensor_reshaped.contiguous().to(torch.bfloat16)
    x2d_t = x2d.transpose(1, 2)
    eye_q = torch.eye(q, device=device, dtype=torch.bfloat16)

    Omega = torch.randn(bs, n, q, device=device, dtype=torch.bfloat16)
    Y = torch.bmm(x2d, Omega)
    del Omega

    for _ in range(n_iter):
        Y = torch.bmm(x2d, torch.bmm(x2d_t, Y))
        Y = _chol_qr_hybrid(Y, eye_cache=eye_q)

    Q = _chol_qr_hybrid(Y, eye_cache=eye_q)
    del Y, eye_q
    
    B = torch.bmm(Q.transpose(1, 2), x2d)
    del x2d, x2d_t

    U_hat, S, Vh = torch.linalg.svd(B.to(torch.float32), full_matrices=False)
    del B

    U_hat = U_hat.to(torch.bfloat16)
    S = S.to(torch.bfloat16)
    Vh = Vh.to(torch.bfloat16)

    U = torch.bmm(Q, U_hat)
    del Q, U_hat

    r = min(rank, S.shape[-1])
    return U[:, :, :r], S[:, :r], Vh[:, :r, :]