import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def load_avg_time_from_json_exact_or_match(path: Path, target_rank: int, prefer: str | None = None, target_niter: int | None = None) -> float:
    """
    Robust loader:
    - First try exact key match if `prefer` is provided and exists.
    - Otherwise, search in data["summary"] for a key that matches:
        - contains "(rank=<target_rank>)"
        - optionally contains "niter=<target_niter>" (if target_niter is not None)
      If multiple candidates exist, prefer the one that matches niter if provided.
    """
    with path.open("r") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    if not isinstance(summary, dict) or not summary:
        raise KeyError(f"{path} missing or empty 'summary' dict.")

    if prefer is not None and prefer in summary:
        return summary[prefer]["avg_time_ms"]

    # rank match
    rank_pat = f"(rank={target_rank})"
    candidates = [k for k in summary.keys() if rank_pat in k]

    if not candidates:
        raise KeyError(
            f"No key contains '{rank_pat}' in {path}\nAvailable keys:\n"
            + "\n".join(sorted(summary.keys()))
        )

    # if niter specified, filter further
    if target_niter is not None:
        niter_pat_1 = f"niter={target_niter}"
        niter_pat_2 = f"n_iter={target_niter}"
        niter_candidates = [k for k in candidates if (niter_pat_1 in k or niter_pat_2 in k)]
        if niter_candidates:
            candidates = niter_candidates

    # pick first deterministic by sorting
    chosen = sorted(candidates)[0]
    item = summary[chosen]
    if "avg_time_ms" not in item:
        raise KeyError(f"Key '{chosen}' found but missing 'avg_time_ms' in {path}")
    return item["avg_time_ms"]


def load_avg_time_from_breakdown_log(path: Path) -> float:
    """
    Parse lines like:
      Average Time: 180.80 ms
    and return the float value in ms.
    """
    txt = path.read_text()
    m = re.search(r"Average Time:\s*([\d\.]+)\s*ms", txt)
    if not m:
        raise ValueError(f"Could not find 'Average Time' in {path}")
    return float(m.group(1))


def main():
    # Repo root: kv-svd/<...>/plot/cholqr_v3/this_file.py -> parents[2] == kv-svd
    repo_root = Path(__file__).resolve().parents[2]

    # Configuration
    target_rank = 256
    target_niter = 8  # change to 4/8/... as needed

    # Paths to existing benchmark files
    path_full = repo_root / "results" / "full_svd" / "xKV_LGS4_RK256_RV384_FULL_SVD_svd_benchmark.json"
    path_lowrank = (
        repo_root
        / "results"
        / "lowrank_svd"
        / f"xKV_LGS4_RK256_RV384_LOWRANK_NITER{target_niter}_svd_benchmark.json"
    )
    path_cholqr_v1 = (
        repo_root
        / "results"
        / "cholqr_v1"
        / "power_iteration"
        / f"xKV_LGS4_RK256_RV384_NITER{target_niter}_svd_benchmark.json"
    )
    path_cholqr_v3 = (
        repo_root
        / "results"
        / "cholqr_v3"
        / "power_iteration_chol"
        / f"xKV_LGS4_RK256_RV384_NITER{target_niter}_svd_benchmark_breakdown.log"
    )

    # Load average latencies (milliseconds)
    avg_full = load_avg_time_from_json_exact_or_match(
        path_full,
        target_rank=target_rank,
        prefer=f"Full SVD (torch.linalg.svd) (rank={target_rank})",
    )

    avg_lowrank = load_avg_time_from_json_exact_or_match(
        path_lowrank,
        target_rank=target_rank,
        # don't hardcode niter=4; match niter dynamically
        target_niter=target_niter,
    )

    avg_cholqr_v1 = load_avg_time_from_json_exact_or_match(
        path_cholqr_v1,
        target_rank=target_rank,
        prefer=f"Custom Randomized SVD (rank={target_rank})",
    )

    avg_cholqr_v3 = load_avg_time_from_breakdown_log(path_cholqr_v3)

    methods = [
        "Full SVD",
        f"Lowrank SVD\n(niter={target_niter})",
        "Randomized SVD\n(cholqr_v1)",
        "Randomized SVD\n(cholqr_v3)",
    ]
    times = [avg_full, avg_lowrank, avg_cholqr_v1, avg_cholqr_v3]

    plt.figure(figsize=(6, 4), dpi=200)
    bars = plt.bar(range(len(methods)), times)

    plt.ylabel("Average latency (ms)")
    plt.xticks(range(len(methods)), methods, rotation=15, ha="right")
    plt.title(
        f"SVD methods comparison (LGS=4, rank={target_rank}, n_iter={target_niter})\n"
        "Full vs Lowrank vs Randomized (cholqr_v1 / cholqr_v3)"
    )

    # Annotate each bar with its value
    for bar, t in zip(bars, times):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{t:.0f} ms",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    out_path = repo_root / "plot" / "cholqr_v3" / "fig_svd_methods_comparison_cholqr_v3.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    print(f"Saved: {out_path}")
    print("Times (ms):")
    for name, t in zip(methods, times):
        print(f"  {name}: {t:.3f}")


if __name__ == "__main__":
    main()
