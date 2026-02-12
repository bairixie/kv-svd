#full_svd
U, S, V_h = torch.linalg.svd(tensor_reshaped, full_matrices=False)

#svd_lowrank
U_trunc, S_trunc, V_trunc = torch.svd_lowrank(tensor_reshaped, q=rank, niter=n_iter)
