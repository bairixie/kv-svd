import torch

@torch.no_grad()
def chol_qr(
    Y_bf16,
    eye_fp32,
    base_eps=1e-6,
    max_eps=10.0,
    max_tries=4,
    use_eigh_last=True,
    fallback_to_qr=True,   
):
    Y = Y_bf16.float()
    G = torch.bmm(Y.transpose(1, 2), Y)
    G = 0.5 * (G + G.transpose(1, 2))

    d = torch.diagonal(G, dim1=-2, dim2=-1)
    scale = (d.mean(dim=-1, keepdim=True).clamp_min(1e-12)).view(-1, 1, 1)
    eps = float(base_eps)
    last_info = None

    for attempt in range(max_tries):
        jitter = (eps * scale)
        R, info = torch.linalg.cholesky_ex(G + jitter * eye_fp32, upper=True)
        if (info == 0).all():
            Q = torch.linalg.solve_triangular(R, Y, upper=True, left=False).to(Y_bf16.dtype)
            return Q
        last_info = info
        eps = min(eps * 10.0, max_eps)

    if use_eigh_last:
        try:
            eigvals, eigvecs = torch.linalg.eigh(G)
            min_ev = max(1e-4, eps * 1.5)
            eigvals = torch.clamp(eigvals, min=min_ev)
            G_spd = torch.bmm(torch.bmm(eigvecs, torch.diag_embed(eigvals)), eigvecs.transpose(1, 2))
            G_spd = 0.5 * (G_spd + G_spd.transpose(1, 2))
            R, info = torch.linalg.cholesky_ex(G_spd, upper=True)
            if (info == 0).all():
                Q = torch.linalg.solve_triangular(R, Y, upper=True, left=False).to(Y_bf16.dtype)
                return Q
            last_info = info
        except Exception:
            pass

    if fallback_to_qr:
        # Householder QR fallback (stable)
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        return Q.to(Y_bf16.dtype)

    raise RuntimeError(
        f"chol_qr failed. tries={max_tries}, base_eps={base_eps:.2e}, max_eps={max_eps:.2e}, "
        f"Y={tuple(Y_bf16.shape)}, G={tuple(G.shape)}, last_info_min={int(last_info.min().item()) if last_info is not None else 'NA'}"
    )


@torch.no_grad()
def randomized_svd_bf16_cholqr(tensor_reshaped, rank, n_iter=8, oversample=4, 
                              base_eps=1e-6, max_eps=10.0, max_tries=4, use_eigh_last=True, breakdown=None):
    device = tensor_reshaped.device

    bs, sl, n = tensor_reshaped.shape
    k = min(rank, sl, n)
    q = min(k + oversample, n)

    x = tensor_reshaped.contiguous().to(torch.bfloat16)
    xt = x.transpose(1, 2)
    eye = torch.eye(q, device=device, dtype=torch.float32).unsqueeze(0)

    Omega = torch.empty((bs, n, q), device=device, dtype=torch.bfloat16)
    Omega.normal_()
    Y = torch.bmm(x, Omega)
    del Omega

    for _ in range(n_iter):
        Y = torch.bmm(x, torch.bmm(xt, Y))
        denom = torch.linalg.vector_norm(Y, dim=1, keepdim=True, dtype=torch.float32).clamp_min(1e-8).to(Y.dtype)
        Y = Y / denom
        Y = chol_qr(Y, eye, base_eps=base_eps, max_eps=max_eps, max_tries=max_tries, use_eigh_last=use_eigh_last)

    Q = chol_qr(Y, eye, base_eps=base_eps, max_eps=max_eps, max_tries=max_tries, use_eigh_last=use_eigh_last)
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