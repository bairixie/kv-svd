# Faster Online Randomized SVD for LLM KV-Cache Compression

---

## Outline

1. **Introduction**
   - 1.1 Memory Bottleneck in Long-Context LLM Inference
   - 1.2 KV-Cache Compression via SVD
   - 1.3 The Online SVD Latency Problem
   - 1.4 Contributions

2. **Background: Singular Value Decomposition**
   - 2.1 Definition: A = UΣVᵀ
   - 2.2 Low-Rank Approximation and Eckart-Young Optimality
   - 2.3 Why Dominant Singular Vectors Are Sufficient

3. **SVD for KV-Cache Compression**
   - 3.1 Per-Layer SVD Baseline
   - 3.2 Cross-Layer SVD: xKV
   - 3.3 Why Online SVD Is Expensive

4. **Randomized SVD and Its Limitations in Practice**
   - 4.1 Why Full SVD Is Wasteful
   - 4.2 Randomized SVD: Four-Stage Algorithm
   - 4.3 Limitations of `torch.svd_lowrank`

5. **Our Method**
   - 5.1 Overview: Same Four Stages, Two Targeted Changes
   - 5.2 Optimization 1: 16-bit Power Iteration
   - 5.3 Optimization 2: Numerically Robust Cholesky QR
      - 5.3.1 Basic Cholesky QR
      - 5.3.2 Gram Matrix Symmetrization
      - 5.3.3 Adaptive Shift Regularization
      - 5.3.4 Eigh SPD-Repair Fallback
      - 5.3.5 Householder QR as Final Safety Net

6. **Experiments**
   - 6.1 Setup
   - 6.2 End-to-End SVD Latency
   - 6.3 Stage-Level Breakdown
   - 6.4 Accuracy vs. Speed Trade-off

7. **Limitations and Future Work**

8. **References**

---

## Full Draft

### 1. Introduction

#### 1.1 Memory Bottleneck in Long-Context LLM Inference

Large Language Models (LLMs) have rapidly extended their effective context windows —
open-source models now routinely support sequences of hundreds of thousands of tokens
[Chang et al., 2025]. This expanded capability unlocks a wide
range of applications: large-scale document retrieval, repository-level code understanding,
and long-horizon reasoning. However, it introduces a severe memory bottleneck.

During autoregressive generation, the attention mechanism requires access to every previously
computed Key and Value (KV) state. These are cached to avoid redundant computation — a
structure known as the **KV-Cache**. Its memory consumption grows as O(L · d · num_layers),
where L is the sequence length and d is the per-head hidden dimension. For a model with 32
layers, head dimension 512, and a context of 128k tokens stored in 16-bit precision, the
KV-Cache alone requires tens of gigabytes — often comparable to the model weights themselves.
This inflated footprint severely constrains the number of concurrent requests and reduces
overall inference throughput [Chang et al., 2025].

#### 1.2 KV-Cache Compression via SVD

Among the approaches proposed to reduce KV-Cache memory — quantization, token eviction,
and low-rank decomposition — **SVD-based compression** stands out for its theoretical
grounding. KV-Cache matrices empirically exhibit rapid singular value decay: most of their
information content is concentrated in a small number of dominant directions. Truncating to
the top-k singular components yields the best possible rank-k approximation guaranteed by
the Eckart-Young theorem, meaning no other rank-k matrix can approximate the KV-Cache
more accurately.

A recent line of work, **xKV** [Chang et al., 2025], takes this further by observing that
the dominant singular vectors of KV-Caches are not only low-rank within a single layer but
are remarkably well-aligned *across multiple adjacent layers*. By horizontally concatenating
the KV-Caches of a group of G layers and applying a single shared SVD, xKV extracts a
common low-rank subspace representing all layers jointly — achieving up to **6.8× higher
compression** than prior inter-layer methods while simultaneously improving accuracy by 2.7%.

#### 1.3 The Online SVD Latency Problem

Unlike weight-compression methods that perform SVD once offline and amortize the cost
across all subsequent inferences, xKV requires computing SVD **online** — during the
prefill phase of each individual request — because the KV-Cache is input-dependent and
changes with every sequence.

In practice, this online SVD constitutes a significant and growing fraction of prefill
latency — a cost we characterize in detail in Section 3.3. Even the approximate
`torch.svd_lowrank` leaves substantial hardware efficiency on the table, motivating the
optimizations in this work.

The standard tool for approximate SVD in PyTorch is `torch.svd_lowrank`, which implements
the randomized algorithm of Halko et al. [2011]. While substantially cheaper than full SVD,
it has two performance limitations that leave significant hardware efficiency on the table.
Addressing these limitations is the focus of this work.

#### 1.4 Contributions

We present a faster, numerically robust implementation of randomized SVD for online
KV-Cache compression. Our implementation is publicly available at
https://github.com/bairixie/kv-svd. Our method follows the same four-stage algorithmic
structure as `torch.svd_lowrank` but introduces two targeted optimizations:

- **16-bit power iteration.** We cast the KV-Cache matrix to 16-bit floating point for
  all matrix multiplications in Stages 1–3, reducing memory bandwidth pressure and
  enabling full Tensor Core utilization. This alone reduces the matrix multiply time in
  the power iteration stage from 91.5s to 22.3s (**4.1× faster**).

- **Numerically robust Cholesky QR orthogonalization.** We replace the Householder QR
  used in `torch.svd_lowrank` with a Cholesky QR method incorporating explicit Gram
  matrix symmetrization, adaptive shift regularization [Fukaya et al., 2020], an
  eigendecomposition-based SPD-repair fallback [Yamazaki et al., 2015], and a final
  Householder QR safety net. Orthogonalization time in the power iteration stage drops
  from 222.6s to 37.3s (**6× faster**).

Combined, our method reduces total SVD CUDA time from **392.0s to 95.7s** — a **4.1×
overall speedup** — and reduces SVD's share of total profiling time from 14.2% to 3.5%.

---

### 2. Background: Singular Value Decomposition

#### 2.1 Definition: A = UΣVᵀ

For any real matrix A ∈ ℝᵐˣⁿ, the **Singular Value Decomposition** (SVD) produces the
factorization [Lee, IBM]:

> **A = U Σ Vᵀ**

where U ∈ ℝᵐˣᵐ and V ∈ ℝⁿˣⁿ are orthogonal matrices, and Σ ∈ ℝᵐˣⁿ is diagonal
with non-negative entries σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0 in decreasing order. The values σᵢ
are called **singular values**; the columns of U and V are the corresponding **left and
right singular vectors**.

Geometrically, SVD decomposes any linear transformation into three interpretable operations:
a rotation in the input space (V), a scaling along orthogonal axes (Σ), and a rotation in
the output space (U). Large singular values correspond to directions that carry significant
structure and energy in the data; small singular values correspond to directions dominated
by noise [Lee, IBM]. In practice, a **thin SVD** is computed retaining only the top
r = min(m, n) components.

#### 2.2 Low-Rank Approximation and the Eckart-Young Theorem

Truncating the SVD to the top-k components yields the **rank-k approximation**:

> **Aₖ = Uₖ Σₖ Vₖᵀ**

The **Eckart-Young theorem** guarantees that Aₖ is the unique minimizer of the
approximation error over all rank-k matrices in both the Frobenius and spectral norms:

> **‖A − Aₖ‖_F = min_{rank(B) ≤ k} ‖A − B‖_F = √(σₖ₊₁² + σₖ₊₂² + ...)**

No other rank-k approximation can beat the truncated SVD. This optimality guarantee is
the theoretical foundation for SVD-based compression throughout machine learning — from
PCA and recommender systems to the KV-Cache compression methods discussed in this work.

#### 2.3 Why Dominant Singular Vectors Are Sufficient

Real-world matrices in deep learning rarely have uniformly distributed singular values.
Instead, singular values typically exhibit **rapid spectral decay**: σ₁ ≫ σ₂ ≫ ... ≫ σᵣ.
When this holds, the rank-k approximation Aₖ with k ≪ r already captures nearly all
of the matrix's variance, enabling massive storage savings with negligible information loss.

For LLM KV-Caches specifically, xKV [Chang et al., 2025] demonstrates via eigenvalue
analysis that capturing 95% of cumulative variance requires only a small fraction of the
total rank. Moreover, when the KV-Caches of multiple adjacent layers are concatenated,
the required rank fraction decreases further — because layers share nearly identical
dominant subspaces and a single basis can represent all of them.

---

### 3. SVD for KV-Cache Compression

#### 3.1 Per-Layer SVD Baseline

The simplest application of SVD to KV-Cache compression processes each layer independently.
For layer ℓ with KV-Cache Xℓ ∈ ℝᴸˣᵈ, we compute the rank-k approximation
Xℓ ≈ Uₖ Σₖ Vₖᵀ and store the compressed pair (UₖΣₖ) ∈ ℝᴸˣᵏ and Vₖᵀ ∈ ℝᵏˣᵈ,
achieving a compression ratio of d/k. As shown in xKV [Chang et al., 2025], per-layer
SVD maintains strong accuracy at moderate compression (e.g., 2.5×) but degrades at
high compression (e.g., 8×) because each layer's approximation error compounds
independently.

#### 3.2 Cross-Layer SVD: xKV

**xKV** [Chang et al., 2025] identifies that KV-Caches of adjacent transformer layers
share **highly aligned dominant singular vectors**, as quantified by Centered Kernel
Alignment (CKA). Despite having low token-level cosine similarity, many layer pairs
exhibit high CKA — indicating that the latent subspaces spanned by their KV-Caches
are structurally similar even when individual token representations differ.

Exploiting this, xKV concatenates the KV-Caches of G adjacent layers horizontally:

> **[Xℓ₁, ..., XℓG] ∈ ℝᴸˣ⁽ᴳᵈ⁾**

and applies a single SVD to extract a **shared low-rank basis** A ∈ ℝᴸˣʳ alongside
layer-specific reconstruction matrices Bℓᵢ ∈ ℝʳˣᵈ. With group size G = 4, xKV
achieves near-baseline accuracy at 8× compression on Llama-3.1-8B — a level
inaccessible to per-layer methods. Our experiments use G = 4, matching this setting.

#### 3.3 Why Online SVD Is Expensive

Both per-layer and cross-layer SVD must be computed **online** during prefill, since the
KV-Cache is input-dependent. Full SVD of an m × n matrix costs O(mn²) flops. As
measured on an RTX A6000 GPU, `torch.linalg.svd` (full SVD) accounts for **73.4%** of
total profiling time, and even the approximate `torch.svd_lowrank` accounts for
**14.2%** (392.0s across 96 samples). Critically, full SVD caused out-of-memory errors
at 96 samples and could only be profiled at approximately 10 samples — underscoring
that it is not viable for production workloads with large group sizes and long contexts.

---

### 4. Randomized SVD and Its Limitations in Practice

#### 4.1 Why Full SVD Is Wasteful

For KV-Cache compression we need only the **top-k singular values and vectors**. Full
SVD computes all min(m, n) components, wasting a factor of min(m, n)/k operations —
often 10× to 100× more than necessary. Randomized SVD [Halko et al., 2011] addresses
this by identifying the k-dimensional dominant subspace from the outset, reducing the
dominant cost from O(mn²) to O(mnk).

#### 4.2 Randomized SVD: Four-Stage Algorithm

Both `torch.svd_lowrank` and our method follow the same four-stage structure
(Algorithms 4.4 and 5.1 of Halko et al. [2011]):

---

**Stage 1 — Setup**

Normalize the problem for subsequent computation. If A is wider than tall (m < n),
transpose it so all stages operate on a tall matrix — ensuring that intermediate
matrices are tall-and-skinny, the regime where Cholesky QR excels. The working dtype
is resolved and any mean matrix M is prepared.

```
if m < n:  A ← Aᵀ,  M ← Mᵀ
X ← cast(A, working_dtype)
eye_q ← identity(k+p, dtype=working_dtype)
```

---

**Stage 2 — Random Projection**

Draw a random Gaussian matrix R ∈ ℝⁿˣ⁽ᵏ⁺ᵖ⁾ (p = oversampling = 4 is good option)
and form the sketch:

> Y = (A − M) · R  ∈ ℝᵐˣ⁽ᵏ⁺ᵖ⁾

Orthonormalize to obtain an initial subspace basis Q. With high probability, this
subspace contains the dominant k-dimensional subspace of A [Halko et al., 2011].

```
R ← randn(n, k+p, dtype=working_dtype)
Y ← (A − M) @ R
Q ← orthonormalize(Y)
```

---

**Stage 3 — Power Iteration**

Refine Q by alternately applying Aᴴ and A:

```
for _ in range(n_iter):
    Q ← orthonormalize( (A−M)ᴴ @ Q )
    Q ← orthonormalize( (A−M)  @ Q )
```

Each full iteration squares the eigenvalue ratios (σᵢ/σⱼ)^(2·n_iter), rapidly
amplifying dominant directions. Re-orthonormalization after each half-step prevents
numerical overflow. With n_iter = 4, this is the single most expensive stage,
accounting for 62.2%–80.1% of total SVD time depending on implementation.

---

**Stage 4 — Project and Recover**

Project A onto Q, compute the small SVD of B, and lift back:

```
B      ← Qᴴ @ (A − M)               # small (k+p) × n matrix
Û, S, Vᵀ ← svd(B.float(), full=False)  # cast to FP32: torch.linalg.svd does not support 16-bit input
U      ← Q @ Û                        # lift to original space
truncate to top-k; undo transpose if needed
```

Total cost is dominated by the (2·n_iter + 1) multiplications with A, each costing
O(mn(k+p)) — a factor of n/(k+p) cheaper than full SVD.

---

#### 4.3 Limitations of `torch.svd_lowrank`

`torch.svd_lowrank` implements the above structure with two performance limitations:

**1. All-FP32 computation, no Tensor Core utilization.**
All matrix multiplications in Stages 1–3 run in FP32. Modern NVIDIA GPUs (Ampere,
Hopper) deliver substantially higher throughput for 16-bit matrix multiplications via
Tensor Cores. In our profiling, the matrix multiply component of the power iteration
alone accounts for 91.5s in the FP32 baseline.

**2. Householder QR is the orthogonalization bottleneck.**
Each `orthonormalize(·)` call uses `torch.linalg.qr` (Householder reflections).
While backward-stable, Householder QR involves sequential panel factorizations that
expose limited parallelism for the tall-and-skinny shapes here. In our profiling,
Householder QR in the power iteration stage accounts for **222.6s** — 56.9% of the
total baseline SVD time of 392.0s.

---

### 5. Our Method

#### 5.1 Overview: Same Four Stages, Two Targeted Changes

Our method is structurally identical to `torch.svd_lowrank`. We introduce exactly
two modifications: (1) 16-bit computation for all large matrix operations, and (2)
Cholesky QR for orthogonalization. The key design principle is to maximize 16-bit
coverage for bandwidth-bound operations while performing a surgical FP32 upgrade
only where precision is non-negotiable.

The complete dtype schedule is shown below:

| Stage | Operation | `torch.svd_lowrank` | Ours (16-bit path) |
|-------|-----------|---------------------|--------------------|
| 1. Setup | Cast input, `eye_q` | FP32 | **16-bit** (input → `low_dtype`; `eye_q` also 16-bit) |
| 2. Random Projection | `Y = A·R`, orthogonalize | FP32 · Householder QR | **16-bit matmul** · **Cholesky QR** |
| 3. Power Iteration | `AᴴQ`, `AQ`, orthogonalize | FP32 · Householder QR | **16-bit matmuls** · **Cholesky QR** |
| 4a. Projection | `B = Qᴴ(A−M)` | FP32 | **16-bit** (`Q`, `A`, `B` all 16-bit) |
| 4b. Small SVD | `svd(B)` | FP32 | **FP32** (`Bproj.float()` — `torch.linalg.svd` does not accept 16-bit input) |
| 4c. Lift & truncate | `U = Q·Û` | FP32 | **16-bit** (cast back to `low_dtype`) |

Two design choices in this table deserve emphasis:

**`chol_qr` is 16-bit-in / 16-bit-out, with an internal FP32 upgrade.**
The orthonormalization routine receives a 16-bit matrix, immediately upcasts to FP32
for Gram matrix computation and Cholesky factorization (where numerical stability
matters), then returns the orthonormal Q in 16-bit. Inter-stage memory traffic stays
in 16-bit; the factorization itself runs in FP32.

**The small SVD of B must be computed in FP32 due to a PyTorch constraint.**
`torch.linalg.svd` does not accept 16-bit tensors as input and will raise a runtime
error if passed a bfloat16 or float16 matrix. The explicit `.float()` cast before
the SVD call is therefore a hard requirement, not an optional precision choice.
Fortunately, at this point B has shape (k+p) × n — e.g., 4 × 512 with oversampling
p = 0 — so the FP32 cast and SVD computation cost negligible memory and time relative
to the large matrix operations in Stages 2–3.

#### 5.2 Optimization 1: 16-bit Power Iteration

The power iteration (Stage 3) consists of repeated large matrix multiplications:

> Q ← orth(Aᴴ Q),    Q ← orth(A Q)

where A is the (grouped, concatenated) KV-Cache of shape L × (G·d). Three properties
make this stage ideal for precision reduction:

**Memory-bandwidth bound.** The dominant cost is reading A from GPU HBM at every
iteration. Reducing element size from 32-bit to 16-bit halves memory traffic, directly
reducing execution time on bandwidth-bound kernels.

**Approximation-tolerant.** The power iteration estimates a subspace, not an exact
result. Small rounding errors from 16-bit arithmetic are equivalent to a slight
perturbation of the input — precisely the setting that randomized SVD is designed to
handle robustly [Halko et al., 2011]. Subsequent iterations further suppress
single-step errors.

**Not the final computation.** The precision-sensitive step — computing singular
values — occurs in Stage 4b (FP32). Stage 3 produces only an intermediate orthonormal
basis Q; numerical errors here affect only the subspace estimate, not the final output.

In our experiments on an RTX A6000, switching the matrix multiply component of the
power iteration from FP32 to 16-bit reduces that sub-cost from **91.5s to 22.3s**
(4.1×), consistent with the expected gain from Tensor Core utilization and halved
memory bandwidth.

**On 16-bit format choice.** Our implementation supports both IEEE float16 and
bfloat16 as the working precision. As shown in Figure 2, both formats yield
essentially identical task accuracy and wall-clock performance on these workloads
(fp16·Cholesky-QR: 0.3750; bf16·Cholesky-QR: 0.3646, a modest difference reflecting
run-to-run variability and the stochastic nature of randomized SVD). Either format
may be selected based on hardware or software constraints.

#### 5.3 Optimization 2: Numerically Robust Cholesky QR

Each `orthonormalize(Y)` call in Stages 2 and 3 takes a tall-and-skinny matrix
Y ∈ ℝᵐˣ⁽ᵏ⁺ᵖ⁾ (m ≫ k+p). All internal computation is performed in FP32 regardless
of the input dtype; the result is cast back to 16-bit before returning.

##### 5.3.1 Basic Cholesky QR

Cholesky QR [Fukaya et al., 2014] exploits the algebraic identity: if Y = QR then
YᵀY = RᵀR. The R factor of the QR decomposition is simultaneously the Cholesky
factor of the Gram matrix G = YᵀY:

```
G = Yᵀ Y                   # (k+p)×(k+p), one SYRK call
R = chol(G, upper=True)    # small Cholesky factor
Q = Y · R⁻¹                # triangular solve (TRSM)
```

Compared to Householder QR, Cholesky QR requires roughly **half the total flop count**
for tall-skinny matrices [Fukaya et al., 2014]. The dominant operations — SYRK and
TRSM — are Level-3 BLAS routines that achieve near-peak GPU throughput and map
efficiently to the GPU memory hierarchy. Householder QR's sequential panel updates
expose far less parallelism for small k+p.

##### 5.3.2 Gram Matrix Symmetrization

Before factorizing, we explicitly symmetrize G:

```
G = Yᵀ Y
G = 0.5 · (G + Gᴴ)
```

In finite precision arithmetic, floating-point rounding in the matrix product Yᵀ Y
accumulates small off-diagonal asymmetries. Explicit symmetrization eliminates this
drift before it reaches `cholesky_ex`, reducing spurious factorization failures that
would otherwise trigger the more expensive fallback paths.

##### 5.3.3 Adaptive Shift Regularization

Following the shifted Cholesky QR framework of Fukaya et al. [2020], we add a
scale-invariant diagonal regularization term before factorizing:

> G_shifted = G + ε · scale · I

where scale = mean(diag(G)).clamp(min=1e-12) makes the shift invariant to the
overall magnitude of Y. We use `torch.linalg.cholesky_ex` — which returns an integer
info tensor rather than raising a Python exception — for batch-aware failure detection:

```
scale ← mean(diag(G)).clamp(min=1e-12)
ε ← base_eps                            # e.g., 1e-5

for attempt in 1 … max_tries:           # e.g., 6 attempts
    R, info ← cholesky_ex(G + ε·scale·I, upper=True)
    if all(info == 0):                   # success on all batch elements
        Q ← solve_triangular(R, Y_f32, upper=True, left=False)
        return cast(Q, 16-bit)
    ε ← min(ε · 10, max_eps)            # exponential backoff
```

In the common case (well-conditioned Y), the first attempt with ε = base_eps succeeds
and the shift is numerically negligible. The exponential backoff handles progressively
more ill-conditioned matrices without manual tuning.

##### 5.3.4 Eigh SPD-Repair Fallback

If all shifted Cholesky attempts fail, we apply an eigendecomposition-based repair
inspired by Yamazaki et al. [2015]. Rather than dividing by square roots of
eigenvalues (which amplifies noise near zero), we reconstruct a strictly positive
definite approximation of G and Cholesky-factorize that:

```
L, V ← eigh(G)
L    ← clamp(L, min = max(1e-4, ε))     # clamp near-zero eigenvalues
G_spd ← V · diag(L) · Vᴴ               # rebuild a strictly SPD matrix
R    ← cholesky(G_spd, upper=True)
Q    ← solve_triangular(R, Y_f32, upper=True, left=False)
return cast(Q, 16-bit)
```

The reconstruct-then-Cholesky design keeps the triangular solve well-conditioned:
the R factor of a strictly positive definite matrix has diagonal entries bounded
away from zero by construction, avoiding the amplification of clamped eigenvalue
errors.

##### 5.3.5 Householder QR as Final Safety Net

If the eigh repair path encounters any exception (e.g., non-finite values in G),
we fall back to standard Householder QR:

```
Q, _ ← torch.linalg.qr(Y_f32, mode="reduced")
return cast(Q, 16-bit)
```

This recovers exactly the behavior of `torch.svd_lowrank`, making our implementation
**strictly more robust than the baseline** — it can never perform worse. In practice,
this path is almost never triggered; it exists as a correctness guarantee.

The complete three-tier strategy is summarized as Algorithm 1:

```
Algorithm 1: chol_qr(Y_16bit)
─────────────────────────────────────────────────────────────────────
Input:  Y ∈ ℝᵐˣ⁽ᵏ⁺ᵖ⁾  in 16-bit
Output: Q ∈ ℝᵐˣ⁽ᵏ⁺ᵖ⁾  with orthonormal columns, in 16-bit

 1.  Y_f32 ← cast Y to float32
 2.  G     ← Y_f32ᴴ Y_f32
 3.  G     ← 0.5 · (G + Gᴴ)                      [symmetrize]
 4.  scale ← mean(diag(G)).clamp(min=1e-12)

     // Tier 1: shifted Cholesky QR
 5.  ε ← base_eps
 6.  for attempt = 1 … max_tries:
 7.      R, info ← cholesky_ex(G + ε·scale·I, upper=True)
 8.      if all(info == 0):
 9.          Q ← solve_triangular(R, Y_f32, upper=True, left=False)
10.          return cast(Q, 16-bit)
11.      ε ← min(ε · 10, max_eps)

     // Tier 2: eigh SPD-repair
12.  try:
13.      L, V ← eigh(G)
14.      L    ← clamp(L, min = max(1e-4, ε))
15.      G_spd ← V · diag(L) · Vᴴ
16.      R    ← cholesky(G_spd, upper=True)
17.      Q    ← solve_triangular(R, Y_f32, upper=True, left=False)
18.      return cast(Q, 16-bit)
19.  except: pass

     // Tier 3: Householder QR safety net
20.  Q, _ ← qr(Y_f32, mode="reduced")
21.  return cast(Q, 16-bit)
─────────────────────────────────────────────────────────────────────
```

---

### 6. Experiments

#### 6.1 Setup

**Hardware.** All experiments are conducted on a single NVIDIA RTX A6000 GPU.
All timing figures report self CUDA time measured via the PyTorch profiler, collected
during the profiling (prefill) phase only — steady-state decode-phase evaluation
is left for future work.

**Benchmark.** We evaluate within the xKV framework [Chang et al., 2025], using the
official xKV codebase at https://github.com/abdelfattah-lab/xKV. Our SVD implementation
is available at https://github.com/bairixie/kv-svd. We use the **Variable Tracking (VT)**
task from the RULER benchmark [Hsieh et al., 2024] as our primary accuracy metric.
RULER is a long-context evaluation suite designed to measure a model's ability to
retrieve and reason over information at various positions within a long context. VT
specifically tests the model's capacity to track variable assignments across a long
document — a task that stresses long-range dependency resolution and is sensitive to
KV-Cache compression artifacts.

**Configuration.** We use layer **group size G = 4** throughout (consistent with the
xKV setting that achieves highest compression with acceptable accuracy loss), and
**n_iter = 4** power iteration steps. The full SVD baseline (`torch.linalg.svd`) could
only be profiled at approximately 10 samples due to GPU out-of-memory errors at 96
samples; `torch.svd_lowrank` and our method (bf16 · Cholesky-QR) both ran for
**96 samples**.

**Baselines compared:**
- `torch.linalg.svd` — full SVD in FP32 (memory-limited reference only)
- `torch.svd_lowrank` — randomized SVD, FP32, Householder QR
- fp32 · Cholesky-QR — our Cholesky QR only, FP32 matmuls
- bf16 · Householder QR — 16-bit matmuls, Householder QR
- fp16 · Cholesky-QR — 16-bit matmuls, Cholesky QR
- **bf16 · Cholesky-QR** — our full method (16-bit matmuls + Cholesky QR)

---

#### 6.2 End-to-End SVD Latency

![Figure 1: SVD Time as Proportion of Total Profiling Duration](plot/cholqr_v6/Figure_1_SVD_Time_Proportion.png)

Table 1 and Figure 1 summarize the end-to-end SVD CUDA time and its proportion of
total profiling time across the three main conditions.

**Table 1. End-to-End SVD timing comparison (RTX A6000, profiling phase).**

| Method | Samples | Total Time | SVD Time | SVD % |
|--------|---------|------------|----------|-------|
| Full SVD (`torch.linalg.svd`, fp32) | ~10 | 541.7s | 397.5s | 73.4% |
| `torch.svd_lowrank` (fp32 · Householder QR) | 96 | 2758.5s | 392.0s | 14.2% |
| **bf16 · Cholesky-QR (ours)** | 96 | 2758.5s | **95.7s** | **3.5%** |

Several observations follow from these results. First, full SVD is not a viable online
method at this scale: it consumes 73.4% of total profiling time and causes OOM at 96
samples, confirming that randomized approaches are necessary. Second, `torch.svd_lowrank`
is already a major improvement over full SVD in relative terms, but 392.0s of SVD time
across 96 samples represents a real throughput bottleneck. Third, our method reduces
SVD CUDA time by **4.1×** (392.0s → 95.7s) and drops SVD's share of total profiling
from 14.2% to 3.5% — a level where SVD is no longer a dominant bottleneck. The
"Other Inference Tasks" time (2758.5 − 95.7 = 2662.8s) is shared across all conditions,
confirming that the speedup is attributable entirely to the SVD itself.

---

#### 6.3 Stage-Level Breakdown

![Figure 3: Randomized SVD — CUDA Time by Stage](plot/cholqr_v6/Figure_3_CUDA_Time_by_Stage.png)

Figure 3 decomposes the total SVD time into four stages and further separates each
stage into its matrix multiply and orthogonalization sub-costs.

**Table 2. Stage-level CUDA time breakdown (n_iter=4, group size 4, RTX A6000).**

| Stage | fp32 · Householder QR | bf16 · Cholesky-QR (ours) | Speedup |
|-------|----------------------|--------------------------|---------|
| 1. Setup (dtype cast / alloc) | 0.017s (0.0%) | 3.60s (3.8%) | — |
| 2. Random Projection | 39.3s (10.0%) | 11.1s (11.6%) | 3.5× |
| 3. Power Iteration (×4) | 314.1s (80.1%) | 59.5s (62.2%) | 5.3× |
|   — Matrix Multiply | 91.5s | 22.3s | 4.1× |
|   — Orthogonalization | 222.6s | 37.3s | 6.0× |
| 4. Project & Recover | 38.6s (9.9%) | 21.5s (22.5%) | 1.8× |
| **Total** | **392.0s** | **95.7s** | **4.1×** |

The breakdown reveals three distinct sources of speedup:

**Stage 1 (Setup) costs slightly more.** The dtype cast of the full KV-Cache matrix
from FP32 to 16-bit adds 3.60s of overhead that does not exist in the FP32 baseline
(0.017s). This is an unavoidable one-time cost and is fully amortized by the savings
in the subsequent stages.

**Stage 3 (Power Iteration) is the primary bottleneck and the primary gain.**
In the baseline, power iteration accounts for 80.1% of total SVD time (314.1s).
Our method reduces this to 62.2% (59.5s), a 5.3× speedup. The two sub-costs
tell different stories: the **matrix multiply** component improves by 4.1× (91.5s →
22.3s) due to Tensor Core utilization from 16-bit precision; the **orthogonalization**
component improves by 6.0× (222.6s → 37.3s) due to Cholesky QR replacing Householder
QR. The orthogonalization speedup is larger in absolute terms — Householder QR's
sequential panel structure was particularly ill-suited to the tall-and-skinny shapes
and small k+p dimensions in this setting.

**Stage 4 (Project & Recover) shows a modest 1.8× gain**, driven by the 16-bit
projection `B = Qᴴ(A−M)`. The small SVD of B remains in FP32 and is not a bottleneck
(it operates on a (k+p) × n matrix of negligible size relative to the full KV-Cache).

---

#### 6.4 Accuracy vs. Speed Trade-off

![Figure 2: SVD Accuracy Comparison on VT task](plot/cholqr_v6/Figure_2_SVD_Accuracy_Comparison_VT.png)

Figure 2 shows task accuracy on VT for all six configurations, with `torch.svd_lowrank`
(fp32 · Householder QR) as the baseline at 0.3979.

**Table 3. Accuracy on Variable Tracking (VT) task, RULER benchmark.**

| Method | Accuracy | Δ vs. baseline |
|--------|----------|----------------|
| Torch lowrank (fp32 · Householder QR) — baseline | 0.3979 | — |
| fp32 · Cholesky-QR | 0.3792 | −0.0187 |
| bf16 · Householder QR | 0.3813 | −0.0166 |
| fp16 · Cholesky-QR | 0.3750 | −0.0229 |
| **bf16 · Cholesky-QR (ours)** | **0.3646** | **−0.0333** |

Several observations are important here. First, **each optimization independently
incurs a modest accuracy cost**: switching to Cholesky QR alone (fp32 · Cholesky-QR:
0.3792) and switching to 16-bit alone (bf16 · Householder QR: 0.3813) both produce
small but measurable drops relative to the fp32 Householder baseline. These costs
combine in our full method (bf16 · Cholesky-QR: 0.3646), which shows the largest
accuracy gap.

The source of this degradation is well-understood. Cholesky QR is less numerically
stable than Householder QR for ill-conditioned inputs [Fukaya et al., 2014]; while our
three-tier fallback strategy (Section 5.3) mitigates catastrophic failures, it cannot
fully eliminate the mild instabilities that arise in the common case. Similarly, 16-bit
arithmetic introduces rounding errors in the power iteration that slightly degrade the
quality of the estimated subspace relative to the FP32 baseline.

The accuracy cost must be weighed against the 4.1× latency reduction. On the VT task,
the drop from 0.3979 to 0.3646 represents a real trade-off that practitioners must
evaluate based on their deployment constraints. In latency-critical settings where
the SVD bottleneck currently limits throughput, our method enables significantly more
requests to be served with an acceptable accuracy cost. In accuracy-critical settings,
the fp32 · Householder baseline or fp32 · Cholesky-QR may be preferable, offering
partial speedup (not separately measured here at full end-to-end scale) with smaller
accuracy impact.

---

### 7. Limitations and Future Work

**Profiling phase only.** All experiments measure the prefill (profiling) phase.
Decoding-phase evaluation — where the compressed KV-Cache is used to generate tokens —
is a critical next step to assess the end-to-end impact on generation quality and
throughput.

**Single task and model.** Results are reported on the VT task from RULER with a
fixed model（Meta-Llama-3.1-8B-Instruct）. Broader evaluation across RULER tasks (N-S, N-MK, QA, etc.), multiple
models (Qwen2.5, DeepSeek), and context lengths would provide a more complete picture
of the accuracy-speed trade-off.

**Fixed rank and group size.** We use a fixed rank k and group size G = 4 across all
experiments. Adaptive rank allocation — allocating more bits to layers or tasks that
are more sensitive to compression — is a promising direction for recovering some of
the accuracy cost identified in Section 6.4.

**Accuracy-stability of Cholesky QR.** The accuracy gap introduced by Cholesky QR
relative to Householder QR warrants further investigation. Exploring CholeskyQR2
[Fukaya et al., 2014] — which applies Cholesky QR twice to improve orthogonality —
may partially close this gap at modest additional cost.

---

### 8. References

1. **Halko, N., Martinsson, P. G., & Tropp, J. A.** (2011). Finding structure with
   randomness: Probabilistic algorithms for constructing approximate matrix
   decompositions. *SIAM Review*, 53(2), 217–288.
   https://doi.org/10.1137/090771806

2. **Fukaya, T., Nakatsukasa, Y., Yanagisawa, Y., & Yamamoto, Y.** (2014). CholeskyQR2:
   A simple and communication-avoiding algorithm for computing a tall-skinny QR
   factorization on a large-scale parallel system. *Proceedings of ScalA 2014*, IEEE,
   pp. 31–38. https://doi.org/10.1109/ScalA.2014.11

3. **Fukaya, T., Kannan, R., Nakatsukasa, Y., Yamamoto, Y., & Yanagisawa, Y.** (2020).
   Shifted Cholesky QR for computing the QR factorization of ill-conditioned matrices.
   *SIAM Journal on Scientific Computing*, 42(1), A477–A503.
   https://doi.org/10.1137/18M1218212

4. **Yamazaki, I., Tomov, S., & Dongarra, J.** (2015). Mixed-precision Cholesky QR
   factorization and its case studies on multicore CPU with multiple GPUs.
   *SIAM Journal on Scientific Computing*, 37(3), C307–C330.
   https://doi.org/10.1137/14M0973773

5. **Chang, C.-C., Lin, C.-Y., Akhauri, Y., Lin, W.-C., Wu, K.-C., Ceze, L., &
   Abdelfattah, M. S.** (2025). xKV: Cross-layer SVD for KV-cache compression.
   *arXiv:2503.18893*.

6. **Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y.,
   & Ginsburg, B.** (2024). RULER: What's the real context size of your long-context
   language models? *arXiv:2404.06654*.

7. **Lee, F.** What is singular value decomposition (SVD)? IBM Think.
   https://www.ibm.com/think/topics/singular-value-decomposition