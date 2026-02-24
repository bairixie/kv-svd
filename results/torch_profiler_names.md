# Torch Profiler Name Mapping (xKV SVD)

This repo profiles the *full inference loop* (not a standalone SVD kernel),
because xKV triggers SVD during **prefill cache merge**.

## A) Our stage labels (record_function tags)

### SVD v5 (randomized SVD core)
- svd_v5/total
  One complete randomized SVD call (A -> U,S,Vh), excluding fake_svd multiply-back.

- svd_v5/prep
  Convert A to BF16, allocate eye_q, small setup.

- svd_v5/projection
  Draw Omega and compute Y = A*Omega, plus the first orthogonalization chol_qr(Y).

- svd_v5/power_iteration/total
  Subspace iteration loop (Halko-style alternating A^T and A).

- svd_v5/power_iteration/at_q
  Y = A^T * Q

- svd_v5/power_iteration/a_q
  Y = A * Q

- svd_v5/power_iteration/qr1 / qr2
  chol_qr(Y) after A^T*Q / after A*Q

- svd_v5/chol_qr/total
  Total cost of one chol_qr call.

- svd_v5/chol_qr/gram
  Gram matrix G = Y^T * Y

- svd_v5/chol_qr/cholesky_ex
  Cholesky factorization with jitter retries.

- svd_v5/chol_qr/solve_triangular
  Solve to obtain Q from R (triangular solve).

- svd_v5/chol_qr/jitter_loop
  Retry loop that increases eps until Cholesky succeeds.

- svd_v5/small_svd_fp32
  SVD on the small projected matrix B (in FP32).

- svd_v5/project_to_lowrank
  B = Q^T * A

- svd_v5/reconstruction
  U = Q * U_small

### Dispatch layer (tells which method was used)
- fake_svd/dispatch/svd_v5
  fake_svd() executed and dispatched to v5.

- full_svd/total
  torch.linalg.svd path.

- lowrank_svd/total
  torch.svd_lowrank path.

## B) PyTorch / CUDA operator names (automatic)

- aten::mm / aten::bmm / aten::matmul
  Matrix multiplications. Usually map to GEMM kernels.

- void cutlass::Kernel2<...bf16...>
  Actual GPU GEMM kernels (CUTLASS). Dominates BF16 matmul cost.

- aten::_flash_attention_forward / aten::_scaled_dot_product_flash_attention
  FlashAttention kernels from the model forward pass (not SVD-specific).

- ProfilerStep*
  Profiler window step boundary (controlled by schedule).