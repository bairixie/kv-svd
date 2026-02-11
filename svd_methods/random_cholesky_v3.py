@torch.no_grad()
def chol_qr(Y_bf16, eye, base_eps=1e-6, max_eps=10.0, max_tries=4, use_eigh_last=True, fallback_to_qr=True,):
    Y = Y_bf16.float()

    # Gram: [bs, q, q]
    G = torch.bmm(Y.transpose(1, 2), Y)
    G = 0.5 * (G + G.transpose(1, 2))  # symmetrize

    # scale ~ trace/q (more robust than mean diag, but both OK)
    d = torch.diagonal(G, dim1=-2, dim2=-1)
    scale = (d.mean(dim=-1, keepdim=True).clamp_min(1e-12)).view(-1, 1, 1)

    eps = float(base_eps)
    last_info = None

    # 1) cheap jitter attempts
    for _ in range(max_tries):
        R, info = torch.linalg.cholesky_ex(G + (eps * scale) * eye, upper=True)
        if (info == 0).all():
            return torch.linalg.solve_triangular(R, Y, upper=True, left=False).to(Y_bf16.dtype)
        last_info = info
        eps = min(eps * 10.0, max_eps)

    # 2) SPD correction 
    if use_eigh_last:
        # lambda_min per batch element
        # G is symmetric; eigvalsh is stable & cheaper than eigh
        lam_min = torch.linalg.eigvalsh(G).min(dim=-1).values.view(-1, 1, 1)  # [bs,1,1]

        # shift = max(0, -lam_min + tiny)
        # tiny uses scale so it's magnitude-aware
        tiny = (1e-6 * scale)
        shift = (-lam_min + tiny).clamp_min(0.0)

        R, info = torch.linalg.cholesky_ex(G + shift * eye, upper=True)
        if (info == 0).all():
            return torch.linalg.solve_triangular(R, Y, upper=True, left=False).to(Y_bf16.dtype)
        last_info = info

    if fallback_to_qr:
        # Householder QR fallback (stable)
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        return Q.to(Y_bf16.dtype)

    raise RuntimeError(
        f"chol_qr failed. tries={max_tries}, base_eps={base_eps:.2e}, max_eps={max_eps:.2e}, "
        f"Y={tuple(Y_bf16.shape)}, G={tuple(G.shape)}, last_info_min={int(last_info.min().item()) if last_info is not None else 'NA'}"
    )


@torch.no_grad()
def randomized_svd_bf16_cholqr_v2(tensor_reshaped, rank, n_iter=8, oversample=4, breakdown=None):
    device = tensor_reshaped.device
    bs, sl, n = tensor_reshaped.shape
    k = min(rank, sl, n)
    q = min(k + oversample, n)

    x = tensor_reshaped.contiguous().to(torch.bfloat16)
    xt = x.transpose(1, 2)

    # IMPORTANT: no unsqueeze(0) needed; [q,q] broadcasts to [bs,q,q]
    eye = torch.eye(q, device=device, dtype=torch.bfloat16)

    Omega = torch.empty((bs, n, q), device=device, dtype=torch.bfloat16)
    Omega.normal_()
    Y = torch.bmm(x, Omega)
    del Omega

    for _ in range(n_iter):
        Y = torch.bmm(x, torch.bmm(xt, Y))
        # keep normalization behavior (it matters for score)
        denom = torch.linalg.vector_norm(Y, dim=1, keepdim=True, dtype=torch.float32).clamp_min(1e-8).to(Y.dtype)
        Y = Y / denom
        Y = chol_qr(Y, eye)

    Q = chol_qr(Y, eye)
    del Y, eye

    B = torch.bmm(Q.transpose(1, 2), x)
    del x, xt

    U_hat, S, Vh = torch.linalg.svd(B.float(), full_matrices=False)
    del B

    U = torch.bmm(Q, U_hat.to(torch.bfloat16))
    S = S.to(torch.bfloat16)
    Vh = Vh.to(torch.bfloat16)
    del Q, U_hat

    r = min(rank, S.shape[-1])
    return U[:, :, :r], S[:, :r], Vh[:, :r, :]