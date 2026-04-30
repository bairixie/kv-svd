# Project Notes

This repository is organized as a personal research showcase for the KV-SVD
project. It is meant to explain the method, preserve the final implementation,
and keep the main logs/figures that support the blog narrative.

## What to Read First

1. `blog.md` for the research story, motivation, method, and results.
2. `svd_methods/random_cholesky_v6.py` for the final randomized SVD kernel.
3. `svd_methods/svd_api.py` for the small wrapper used to select full SVD,
   PyTorch low-rank SVD, or the Cholesky-QR method.
4. `plot/cholqr_v6/` for the main figures used in the write-up.

## Final Method

The reported method is `cholqr_v6`:

- 16-bit projection and power iteration for the large GEMMs.
- Cholesky-QR orthogonalization with FP32 internal Gram formation.
- Gram symmetrization, adaptive diagonal regularization, optional SPD repair,
  and Householder QR fallback.
- FP32 small SVD on the projected matrix.

Earlier `cholqr_v1` through `cholqr_v5` files are retained as research history,
not as recommended entry points.

## Result Artifacts

The `results/` directory contains copied logs and JSON outputs from xKV runs.
These files are included as evidence for the blog and figures, not as a
standalone benchmark harness.
