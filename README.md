# kv-svd

Fast randomized SVD for LLM KV-cache compression, with end-to-end benchmarking and analysis.

**Code:** [github.com/bairixie/kv-svd](https://github.com/bairixie/kv-svd)  
**Full write-up (method, experiments, figures):** [blog.md](blog.md)

---

## What we're solving

In long-context LLM inference, the **KV cache** grows linearly with sequence length and becomes a major **memory and compute bottleneck**. A standard approach is to approximate it with a **low-rank SVD**. Exact SVD (`torch.linalg.svd`) is too slow and memory-heavy for online use; even **randomized SVD** via `torch.svd_lowrank` leaves a lot of hardware efficiency on the table.

This repo provides **faster, numerically robust randomized SVD kernels** designed for KV-cache shapes and integrates with the [xKV](https://github.com/abdelfattah-lab/xKV) codebase for end-to-end latency and accuracy evaluation.

---

## What we built

- **Randomized SVD implementations (`cholqr_v1`–`cholqr_v6`)** in `svd_methods/`:
  - **16-bit power iteration** (FP16/BF16) on Tensor Cores for speed.
  - **Cholesky-based QR** for tall-skinny orthogonalization, with Gram symmetrization, adaptive regularization, SPD-repair fallback, and optional Householder QR.
  - Small core SVD kept in **FP32** for stability.
- **cholqr_v6** is the main kernel used in the reported experiments: **4.1× faster** than `torch.svd_lowrank` (392.0s → 95.7s SVD CUDA time), with SVD’s share of total profiling time dropping from 14.2% to 3.5%.
- **Baselines**: full SVD and `torch.svd_lowrank` (see `svd_baselines.py`).
- **Benchmark results** under `results/` (from xKV runs): logs and JSON for full_svd, lowrank_svd, and cholqr_v1–v6.
- **Pre-generated figures** in `plot/` (including `plot/cholqr_v6/` for the main paper figures).

All details (algorithm, setup, tables, and figure interpretations) are in **[blog.md](blog.md)**.

---

## Experimental setup (main comparison)

- **Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Task:** RULER Variable Tracking (`ruler/vt`), **65,536-token context**
- **Example KV shape:** `[1, 32, 65295, 128]` (batch=1, heads=32, seq_len≈65k, head_dim=128)
- **Example config:** layer group size (LGS)=4, rank K=256, value rank V=384, n_iter=4
- **Precision:** KV and large GEMMs in FP16; small SVD and critical steps in FP32

We measure SVD CUDA time (total and per-stage), task accuracy on `ruler/vt`, and trade-offs over power iterations, rank, and LGS.

---

## Methods compared

| Method | Description |
|--------|-------------|
| **Full SVD** | `torch.linalg.svd` — exact, slow, memory-heavy; accuracy upper bound. |
| **Low-rank SVD** | `torch.svd_lowrank` — PyTorch randomized SVD; baseline for speed/accuracy. |
| **cholqr_v1–v6** | Custom randomized SVD: Cholesky QR + 16-bit power iteration; v6 is the main production kernel with orth choice (chol/house) and full per-stage breakdown. |

---

## Repository layout

```
kv-svd/
├── blog.md                 # Full write-up: method, experiments, figures
├── README.md               # This file
├── svd_methods/            # SVD implementations and high-level API
│   ├── svd_baselines.py   # Full SVD & torch.svd_lowrank baselines
│   ├── svd_api.py         # Unified wrapper: 'full' / 'lowrank' / 'cholqr'
│   ├── random_cholesky_v1.py … random_cholesky_v6.py   # cholqr kernels
├── results/                # Benchmark outputs (from xKV)
│   ├── full_svd/           # Full SVD runs
│   ├── lowrank_svd/        # torch.svd_lowrank runs
│   ├── cholqr_v1/ … cholqr_v6/   # Custom kernel runs (vt, fwe, niah_*, etc.)
├── plot/                   # Figures and plot outputs
│   ├── cholqr_v6/          # Main figures (SVD time proportion, stage breakdown, accuracy)
│   ├── cholqr_v3/          # Legacy comparison figures for cholqr_v3
│   └── fig_*.png           # All-methods comparison figures
```

- **SVD code:** Used by xKV via plug-in; not run standalone in this repo.
- **Results:** Produced by [xKV](https://github.com/abdelfattah-lab/xKV). Copy or symlink xKV logs/JSON into `results/*` as needed.
- **Plots:** Figures in `plot/` are generated from those results (plotting scripts may live in a separate workflow or script set). Key figures for the blog are in `plot/cholqr_v6/`.

---

## Running benchmarks (in xKV)

End-to-end KV-cache benchmarks are run inside the **xKV** repo, not here. Example (RULER VT, 65k context, xKV-4):

```bash
# In the xKV repo
CUDA_VISIBLE_DEVICES=... OMP_NUM_THREADS=... torchrun --standalone --nnodes=1 --nproc_per_node 4 \
  evaluate/eval_acc.py \
  --datalen 65536 \
  --batch_size 1 \
  --dataset_name "ruler/vt" \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --xKV --merge_k --merge_v \
  --rank_k 256 --rank_v 384 \
  --layer_group_size 4 \
  --start_layer_idx 0 --end_layer_idx -1
```

Logs and JSON produced by xKV can be copied into this repo’s `results/*` for local analysis and plotting.

**Naming convention (typical):**

- `xKV_LGS{LGS}_RK{RK}_RV{RV}_NITER{N}_svd_benchmark.json` — latency records  
- `xKV_LGS{LGS}_RK{RK}_RV{RV}_NITER{N}.log` — accuracy and high-level stats  

---

## Randomized SVD variants (cholqr_v1–v6)

| Version | Notes |
|--------|--------|
| **v1** | Cholesky QR + fixed jitter; fast but numerically sensitive. |
| **v2** | Dynamic jitter, full `eigh`-based SPD correction, explicit normalization. |
| **v3** | Same stability as v2; cheaper `eigvalsh`-based shifts. |
| **v4** | 16-bit-oriented, trace-scaled jitter, optional eigen-clamping, mixed-precision normalization. |
| **v5** | Further tuning and options for KV-cache workloads. |
| **v6** | Main kernel: `randomized_svd_fp16()` with `orth` (chol / house), `power_dtype`, transpose handling, and aggressive memory release; exposed via the `'cholqr'` option in `svd_methods/svd_api.py`, and used for the reported 4.1× speedup and figures in [blog.md](blog.md). |

---

## Environment

- **For xKV:** Follow xKV’s requirements (PyTorch, CUDA, etc.).
- **For local plotting/analysis:** `matplotlib`, `numpy`, and optionally `torch`. If `~/.matplotlib` is not writable, set `MPLCONFIGDIR` (e.g. `export MPLCONFIGDIR=/tmp/matplotlib-cache`).

Commands in this README assume the project root is the repo root (where `blog.md` and `svd_methods/` live).

---

## Citation and links

- **xKV (KV-cache compression):** [abdelfattah-lab/xKV](https://github.com/abdelfattah-lab/xKV)
- **This implementation:** [bairixie/kv-svd](https://github.com/bairixie/kv-svd)
- **Full method and experiments:** [blog.md](blog.md)
