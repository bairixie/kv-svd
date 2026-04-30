# KV-SVD

Hardware-efficient randomized SVD for online LLM KV-cache compression.

**Code:** [github.com/bairixie/kv-svd](https://github.com/bairixie/kv-svd)  
**Full write-up (method, experiments, figures):** [blog.md](blog.md)

This is a **personal research showcase repository**. It preserves the final SVD
method, the development path, and the evidence used in the write-up. It is not
packaged as a standalone reproduction benchmark; end-to-end evaluation was run
inside the upstream xKV framework.

---

## Research question

Long-context LLM inference is increasingly limited by the **KV cache**, whose memory footprint grows linearly with sequence length and number of layers. xKV reduces this footprint by applying cross-layer SVD to grouped KV-cache blocks, but the SVD must run online during prefill. Exact SVD (`torch.linalg.svd`) is too slow and memory-heavy for this setting; PyTorch randomized SVD (`torch.svd_lowrank`) is faster, but still spends most of its time in FP32 matrix multiplies and Householder QR.

This repository studies a narrower systems question: **how much of randomized SVD for KV-cache compression can be moved onto GPU-friendly 16-bit matrix operations without losing task accuracy?**

---

## Main contribution

- **`cholqr_v6` is the reported method.** It keeps the randomized SVD algorithmic structure, but runs the large projection/power-iteration GEMMs in 16-bit and replaces Householder QR with a Cholesky-QR orthogonalization path designed for tall-skinny matrices.
- **Numerical safeguards are explicit.** Cholesky QR uses FP32 Gram formation, Gram symmetrization, adaptive diagonal regularization, optional eigendecomposition-based SPD repair, and Householder QR as the final fallback.
- **The small core SVD remains FP32.** This preserves stability and matches PyTorch's current requirement that `torch.linalg.svd` not receive FP16/BF16 inputs.
- **Measured result:** `cholqr_v6` is **4.1× faster** than `torch.svd_lowrank` in the reported SVD CUDA-time comparison (392.0s → 96.7s), while matching average RULER accuracy within 0.12 percentage points.
- **Baselines**: full SVD and `torch.svd_lowrank` (see `svd_baselines.py`).
- **Benchmark results** under `results/` (from xKV runs): logs and JSON for full_svd, lowrank_svd, and cholqr_v1–v6.
- **Pre-generated figures** in `plot/`, with the main figures in `plot/cholqr_v6/`.

The full narrative, algorithm, experiments, tables, and figure interpretation are in **[blog.md](blog.md)**.

---

## Experimental setup (main comparison)

- **Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Task:** RULER Variable Tracking (`ruler/vt`), **65,536-token context**
- **Example KV shape:** `[1, 32, 65295, 128]` (batch=1, heads=32, seq_len≈65k, head_dim=128)
- **Example config:** layer group size (LGS)=4, rank K=256, value rank V=384, n_iter=4
- **Precision:** KV and large GEMMs in 16-bit; small SVD and Cholesky internals in FP32

The blog reports SVD CUDA time, per-stage time breakdowns, and RULER task accuracy across FWE, NIAH MultiKey, NIAH Single1, and VT.

---

## Methods compared

| Method | Description |
|--------|-------------|
| **Full SVD** | `torch.linalg.svd` — exact, slow, memory-heavy; accuracy upper bound. |
| **Low-rank SVD** | `torch.svd_lowrank` — PyTorch randomized SVD; baseline for speed/accuracy. |
| **cholqr_v1–v6** | Research iterations of the custom randomized SVD kernel. v6 is the main method used in the reported figures. |

---

## Repository layout

```
kv-svd/
├── blog.md                 # Full write-up: method, experiments, figures
├── README.md               # This file
├── docs/                   # Project and result notes
├── svd_methods/            # SVD implementations and high-level API
│   ├── svd_baselines.py    # Full SVD and torch.svd_lowrank baselines
│   ├── svd_api.py          # Unified wrapper: 'full' / 'lowrank' / 'cholqr'
│   └── random_cholesky_v*.py   # Cholesky-QR randomized SVD variants
├── results/                # Benchmark outputs (from xKV)
│   ├── full_svd/           # Full SVD runs
│   ├── lowrank_svd/        # torch.svd_lowrank runs
│   └── cholqr_v1/ … cholqr_v6/   # Custom kernel runs
├── plot/                   # Figures and plot outputs
│   ├── cholqr_v6/          # Main figures (SVD time proportion, stage breakdown, accuracy)
│   └── fig_*.png           # All-methods comparison figures
```

- **SVD code:** Final method code is in `svd_methods/random_cholesky_v6.py` and exposed through `svd_methods/svd_api.py`.
- **Results:** Stored logs/JSON are copied from xKV runs and kept as supporting evidence for the blog.
- **Plots:** Key figures for the public narrative are in `plot/cholqr_v6/`.

---

## Using the SVD API

```python
from svd_methods.svd_api import SVDConfig, run_svd

config = SVDConfig(method="cholqr", rank=256, n_iter=4, oversample=4)
U, S, Vh = run_svd(
    tensor,
    config,
    power_dtype="bf16",  # options: "fp32", "bf16", "fp16", "fp8", "fp8_e5m2"
    orth="chol",        # options: "chol" or "house"
)
```

For comparison baselines, set `method="full"` or `method="lowrank"`.

---

## Optional Evaluation Context

End-to-end KV-cache benchmarks were run inside the **xKV** repo. The command
below documents the evaluation context used for the reported RULER VT runs; it
is included for orientation rather than as a full reproduction script.

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

Logs and JSON produced by xKV were copied into this repo’s `results/*` folders
as research artifacts.

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
| **v6** | Main method: `randomized_svd_fp16()` with selectable orthogonalization (`chol` / `house`), configurable `power_dtype`, wide-matrix transpose handling, and per-stage behavior matching the blog narrative. Exposed through `method="cholqr"` in `svd_methods/svd_api.py`. |

---

## Scope

- This repository is intended to make the final method and research story easy
  to inspect.
- It does not vendor xKV, model checkpoints, datasets, or a full benchmark
  environment.
- Earlier `cholqr_v1`–`cholqr_v5` files are retained to show the research
  progression; `cholqr_v6` is the method to read first.

---

## Citation and links

- **xKV (KV-cache compression):** [abdelfattah-lab/xKV](https://github.com/abdelfattah-lab/xKV)
- **This implementation:** [bairixie/kv-svd](https://github.com/bairixie/kv-svd)
- **Full method and experiments:** [blog.md](blog.md)
