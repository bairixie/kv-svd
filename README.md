# kv-svd

Fast randomized SVD kernels for LLM KV-cache compression and end‑to‑end benchmarking.

This repo contains:
- Custom randomized SVD implementations for KV-cache compression (cholqr_v1–v4).
- Baseline SVD methods (full SVD, `torch.svd_lowrank`).
- Benchmark scripts that measure latency, breakdown by stage, and accuracy on `ruler/vt`.
- Plotting utilities to compare different SVD methods, power iterations, and layer group sizes.

All analysis code and figures are designed to be reproducible directly from this repository.

---

## 1. Repository layout

- `svd_methods/`
  - `random_householdQR.py`: randomized SVD with Householder QR / `torch.linalg.qr`.
  - `random_cholesky_v4.py`: latest cholqr_v4 randomized SVD kernel (Cholesky‑based QR + power iteration).
- `results/`
  - `full_svd/`: full SVD benchmarks and logs.
  - `lowrank_svd/`: `torch.svd_lowrank` benchmarks and logs.
  - `cholqr_v1/`, `cholqr_v2/`, `cholqr_v3/`, `cholqr_v4/`:
    - `power_iterations/`: vary `n_iter` for fixed LGS / rank / RV.
    - `layer_group_size/`: vary layer group size (LGS) for fixed `n_iter`.
- `plot/`
  - `fig_all_methods_comparison.py`: main script to compare **all SVD methods** (Full, Lowrank, cholqr_v1–v4).
  - `fig_cholqr_v4_comparison.py`: detailed analysis for cholqr_v4 (power iteration & layer group size).
  - `cholqr_v1/`, `cholqr_v2/`, `cholqr_v3/`, `cholqr_v4/`: saved figures for each randomized SVD variant.

---

## 2. Data generation pipeline

### 2.1 Run SVD benchmarks

Benchmarks are launched from the training / evaluation scripts in your environment (outside this repo) and write JSON / log files into `results/*`.  
Each experiment is parameterized by:

- **Layer Group Size (LGS)**: how many transformer layers are merged into one KV block (e.g. 2, 4, 8, 16).
- **Rank (RK)** and **Rank‑Value (RV)**: low‑rank dimension for K/V (e.g. RK=256, RV=384; RK=512, RV=768).
- **Power iteration `n_iter`**: number of power iterations used in the randomized SVD.

File names follow the pattern:

```text
xKV_LGS{LGS}_RK{RK}_RV{RV}_NITER{N}_svd_benchmark.json   # latency records
xKV_LGS{LGS}_RK{RK}_RV{RV}_NITER{N}.log                  # accuracy + high‑level stats
```

For cholqr_v2/cholqr_v3 (with detailed breakdown logs) the pattern is:

```text
..._NITER{N}_svd_benchmark_breakdown.log
```

### 2.2 Aggregate summaries by rank

Some raw JSON files store only per‑call `records` without pre‑computed averages per rank.  
To create rank‑wise summaries for full SVD, lowrank, cholqr_v1, and cholqr_v4, run:

```bash
cd "/Users/mozhihao/Desktop/svd research/kv-svd"
python results/aggregate_cholqr_v4_by_rank.py    # also processes cholqr_v4 power_iterations & layer_group_size
```

This script:
- Groups records by `rank`.
- Computes `count`, `total_time_ms`, `avg_time_ms`, `min_time_ms`, `max_time_ms`.
- For cholqr_v4, also computes `avg_breakdown_ms` for each stage (Preparation, Projection, Power Iteration, etc.).
- Rewrites each `*_svd_benchmark.json` in place with a new `summary` section.

### 2.3 Generate an English summary log for cholqr_v4

To inspect all cholqr_v4 results in one place:

```bash
python results/dump_cholqr_v4_summary_log.py
```

This writes:

- `results/cholqr_v4/cholqr_v4_aggregated_summary.log`

Each block corresponds to one JSON file and includes:
- Accuracy (baseline score) extracted from the `.log` file.
- Total test time (e.g. `46:57`) from the `Testing: 100%|...` line.
- Per‑rank aggregated latency and per‑stage average breakdown.

---

## 3. Plotting and analysis

### 3.1 All‑methods comparison (Full vs Lowrank vs cholqr_v1–v4)

Use `plot/fig_all_methods_comparison.py` to compare **all SVD implementations** under a specific configuration:

```bash
python -m plot.fig_all_methods_comparison --lgs 4 --rank 256 --rv 384 --n_iter 8
python -m plot.fig_all_methods_comparison --lgs 8 --rank 512 --rv 768 --n_iter 8
```

Arguments:
- `--lgs`: layer group size.
- `--rank`: SVD rank (RK).
- `--rv`: value rank (RV), usually `rank * 1.5` (e.g., 256 → 384, 512 → 768).
- `--n_iter`: power iteration count used for the randomized methods.

The script:
- Automatically locates the appropriate JSON / log files for:
  - `results/full_svd`
  - `results/lowrank_svd`
  - `results/cholqr_v1`, `cholqr_v2`, `cholqr_v3`, `cholqr_v4`
  - Falls back between `power_iterations` and `layer_group_size` directories when needed.
- Builds a figure with two subplots:
  1. **Latency Comparison** (average latency in ms).
  2. **Accuracy Comparison** (baseline score on `ruler/vt`).
- Saves to:

```text
plot/fig_all_methods_comparison_LGS{LGS}_RK{RK}_RV{RV}_NITER{N}.png
```

Cholqr_v4 is treated as an additional randomized SVD method and appears alongside cholqr_v1/v2/v3.

There is also a helper mode to list all available configurations detected in `results/*`:

```bash
python -m plot.fig_all_methods_comparison list
```

This prints all `(LGS, Rank, RV, n_iter)` combinations inferred from filenames.

### 3.2 Power iteration and layer‑group analysis for cholqr_v4

`plot/fig_cholqr_v4_comparison.py` provides detailed analysis for cholqr_v4 only.

**Power iteration comparison (fixed LGS, rank, RV):**

```bash
python -m plot.fig_cholqr_v4_comparison power_iter --lgs 4 --rank 256 --rv 384
python -m plot.fig_cholqr_v4_comparison power_iter --lgs 8 --rank 512 --rv 768
```

Each figure contains three panels:
- Latency vs `n_iter` (2, 4, 8, 16 – pulled from `power_iterations` and, when needed, `layer_group_size`).
- Accuracy vs `n_iter`.
- Latency breakdown vs `n_iter` (stacked by stage).

The resulting files are saved under:

```text
plot/cholqr_v4/fig_cholqr_v4_power_iteration_LGS{LGS}_RK{RK}_RV{RV}.png
```

**Layer group size comparison (fixed rank, RV, `n_iter`):**

```bash
python -m plot.fig_cholqr_v4_comparison layer_group --rank 512 --rv 768 --n_iter 8
```

Each figure contains:
- Latency vs LGS (2, 4, 8, 16).
- Accuracy vs LGS.
- Latency breakdown vs LGS.

Outputs:

```text
plot/cholqr_v4/fig_cholqr_v4_layer_groups_RK{RK}_RV{RV}_NITER{N}.png
```

### 3.3 Per‑algorithm power‑iteration plots for cholqr_v1–v3

For legacy comparison with cholqr_v1–v3, use modes inside `fig_all_methods_comparison.py`:

```bash
# Single algorithm, power iteration sweep
python -m plot.fig_all_methods_comparison power_iter --algorithm cholqr_v2 --rank 256 --lgs 4 --rv 384

# All three algorithms at once (same rank / LGS / RV)
python -m plot.fig_all_methods_comparison all_power_iter --rank 256 --lgs 4 --rv 384
```

These produce figures like:
- `plot/fig_power_iteration_comparison_cholqr_v2_LGS4_RK256_RV384.png`

Design is analogous to the cholqr_v4 plots (latency, accuracy, breakdown).

---

## 4. Interpretation notes

- **Full SVD** is the slowest but provides a strong accuracy baseline.
- **Lowrank SVD (`torch.svd_lowrank`)** offers a speed/accuracy trade‑off controlled by `niter` and rank.
- **cholqr_v1–v3** progressively optimize:
  - numerical stability (Householder vs Cholesky QR),
  - precision (BF16 compute with FP32 QR/SVD),
  - and breakdown logging.
- **cholqr_v4** further improves the kernel and is currently the main focus:
  - Provides detailed per‑stage breakdown via aggregated JSON summaries.
  - Achieves significantly lower latency than full SVD while keeping accuracy competitive, especially around moderate `n_iter` (e.g., 8).
  - Exhibits clear trade‑offs between LGS (coarser grouping improves compression but may reduce accuracy) and `n_iter` (more iterations improve accuracy but increase latency).

When analyzing new experiments, the typical workflow is:
1. Run benchmarks to fill `results/*`.
2. Aggregate summaries via the `aggregate_*.py` scripts.
3. Regenerate plots using the `plot/*.py` utilities.
4. Inspect both the overall comparison figures and the per‑algorithm power‑iteration / LGS‑sweep plots.

---

## 5. Environment notes

- The plotting scripts rely on:
  - `matplotlib`
  - `numpy`
  - `torch` (only for reference in some methods)
- If `~/.matplotlib` is not writable on your system, you can set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-cache
```

to avoid warnings and speed up Matplotlib imports.

All paths in this README assume the project root is:

```text
/Users/mozhihao/Desktop/svd research/kv-svd
```

If you clone the repo elsewhere, simply `cd` into the new root before running the commands above.

