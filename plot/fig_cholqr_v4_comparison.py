import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from plot.fig_all_methods_comparison import load_accuracy_from_log


def load_cholqr_v4_stats_from_json(
    path: Path, target_rank: int
) -> Tuple[float, Dict[str, float]]:
    """
    Load averaged latency and breakdown for cholqr_v4 from an aggregated
    *_svd_benchmark.json file.

    Assumes the file has already been processed by
    results/aggregate_cholqr_v4_by_rank.py and contains entries like:
      "randomized (Rank 512) (rank=512)": { ... }
    """
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    if not summary:
        raise ValueError(f"No 'summary' in {path}")

    # Prefer keys that explicitly contain "(rank=target_rank)"
    candidates = []
    for key, info in summary.items():
        if f"(rank={target_rank})" in key:
            candidates.append((key, info))

    if not candidates:
        # Fallback: any entry that mentions Rank {target_rank}
        for key, info in summary.items():
            if f"Rank {target_rank}" in key or str(target_rank) in key:
                candidates.append((key, info))

    if not candidates:
        raise KeyError(
            f"Could not find aggregated entry for rank={target_rank} in {path}"
        )

    # Use the first candidate
    _, info = candidates[0]
    avg_time = float(info.get("avg_time_ms"))
    breakdown = info.get("avg_breakdown_ms") or {}

    # Normalize breakdown keys to nice labels
    pretty_breakdown: Dict[str, float] = {}
    mapping = {
        "preparation_ms": "Preparation",
        "projection_ms": "Projection (BMM)",
        "power_iteration_ms": "Power Iteration",
        "final_qr_ms": "Final QR",
        "project_to_lowrank_ms": "Project to Low-rank",
        "small_svd_fp32_ms": "Small SVD",
        "reconstruction_ms": "Reconstruction",
    }
    for raw_name, pretty in mapping.items():
        if raw_name in breakdown:
            try:
                pretty_breakdown[pretty] = float(breakdown[raw_name])
            except (TypeError, ValueError):
                continue

    return avg_time, pretty_breakdown


def plot_cholqr_v4_power_iterations(
    target_lgs: int = 8,
    target_rank: int = 512,
    target_rv: int = 768,
) -> Path:
    """
    For a fixed LGS/rank/rv, compare different power_iteration (n_iter) values
    for cholqr_v4 using:
      - total latency (avg_time_ms)
      - accuracy (from log)
      - latency breakdown (avg_breakdown_ms)
    """
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "results" / "cholqr_v4" / "power_iterations"

    candidate_niters = [2, 4, 8, 16]

    n_iters: List[int] = []
    latencies: List[float] = []
    accuracies: List[float] = []
    breakdowns: List[Dict[str, float]] = []

    print(
        f"Loading cholqr_v4 power_iteration data (LGS={target_lgs}, "
        f"rank={target_rank}, RV={target_rv})..."
    )

    for n in candidate_niters:
        # Primary: power_iterations directory
        json_path = (
            base_dir
            / f"xKV_LGS{target_lgs}_RK{target_rank}_RV{target_rv}_NITER{n}_svd_benchmark.json"
        )
        log_path = (
            base_dir
            / f"xKV_LGS{target_lgs}_RK{target_rank}_RV{target_rv}_NITER{n}.log"
        )

        # Fallback: some n_iter (e.g., 8, 16) are only stored under layer_group_size
        if not json_path.exists():
            lg_dir = (
                repo_root
                / "results"
                / "cholqr_v4"
                / "layer_group_size"
            )
            alt_json = (
                lg_dir
                / f"xKV_LGS{target_lgs}_RK{target_rank}_RV{target_rv}_NITER{n}_svd_benchmark.json"
            )
            alt_log = (
                lg_dir
                / f"xKV_LGS{target_lgs}_RK{target_rank}_RV{target_rv}_NITER{n}.log"
            )
            if alt_json.exists():
                json_path = alt_json
                log_path = alt_log

        if not json_path.exists():
            print(f"  Skipping n_iter={n}: JSON not found -> {json_path.name}")
            continue

        try:
            latency, breakdown = load_cholqr_v4_stats_from_json(
                json_path, target_rank=target_rank
            )
        except Exception as e:
            print(f"  ERROR reading {json_path.name}: {e}")
            continue

        acc = None
        if log_path.exists():
            try:
                acc = load_accuracy_from_log(log_path)
            except Exception as e:
                print(f"  Warning: could not parse accuracy from {log_path.name}: {e}")

        n_iters.append(n)
        latencies.append(latency)
        accuracies.append(acc)
        breakdowns.append(breakdown)

        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        print(
            f"  n_iter={n:2d}: latency={latency:7.2f} ms, "
            f"accuracy={acc_str}"
        )

    if not n_iters:
        raise RuntimeError("No power_iteration data loaded for cholqr_v4.")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)

    # 1) Latency vs n_iter
    ax1 = axes[0]
    ax1.plot(n_iters, latencies, marker="o", linewidth=2, color="#2CA02C")
    ax1.set_xlabel("Power Iteration (n_iter)", fontsize=12)
    ax1.set_ylabel("Average Latency (ms)", fontsize=12)
    ax1.set_title("cholqr_v4 - Latency vs Power Iteration", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_xticks(n_iters)
    for n, lat in zip(n_iters, latencies):
        ax1.text(n, lat, f"{lat:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2) Accuracy vs n_iter
    ax2 = axes[1]
    valid_acc = [(n, a) for n, a in zip(n_iters, accuracies) if a is not None]
    if valid_acc:
        n_acc, acc_vals = zip(*valid_acc)
        ax2.plot(n_acc, acc_vals, marker="o", linewidth=2, color="#9467BD")
        ax2.set_xlabel("Power Iteration (n_iter)", fontsize=12)
        ax2.set_ylabel("Accuracy (Baseline Score)", fontsize=12)
        ax2.set_title("cholqr_v4 - Accuracy vs Power Iteration", fontsize=13, fontweight="bold")
        ax2.grid(alpha=0.3, linestyle="--")
        ax2.set_xticks(n_iters)
        for n, acc in valid_acc:
            ax2.text(n, acc, f"{acc:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        ax2.text(
            0.5,
            0.5,
            "No accuracy data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("cholqr_v4 - Accuracy vs Power Iteration", fontsize=13, fontweight="bold")

    # 3) Latency Breakdown (stacked)
    ax3 = axes[2]
    stage_order = [
        "Preparation",
        "Projection (BMM)",
        "Power Iteration",
        "Final QR",
        "Project to Low-rank",
        "Small SVD",
        "Reconstruction",
    ]
    stage_colors = [
        "#8c564b",
        "#17becf",
        "#ff7f0e",
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]

    x_pos = range(len(n_iters))
    bottom = [0.0] * len(n_iters)
    for stage, color in zip(stage_order, stage_colors):
        values = []
        for bd in breakdowns:
            values.append(bd.get(stage, 0.0) if bd is not None else 0.0)
        ax3.bar(x_pos, values, bottom=bottom, label=stage, color=color, alpha=0.85)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax3.set_xlabel("Power Iteration (n_iter)", fontsize=12)
    ax3.set_ylabel("Latency (ms)", fontsize=12)
    ax3.set_title("cholqr_v4 - Latency Breakdown", fontsize=13, fontweight="bold")
    ax3.set_xticks(list(x_pos))
    ax3.set_xticklabels([str(n) for n in n_iters])
    ax3.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.suptitle(
        f"cholqr_v4 Power Iteration Effect (LGS={target_lgs}, rank={target_rank}, RV={target_rv})",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    out_dir = repo_root / "plot" / "cholqr_v4"
    out_path = out_dir / f"fig_cholqr_v4_power_iteration_LGS{target_lgs}_RK{target_rank}_RV{target_rv}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved power_iteration figure to: {out_path}")
    return out_path


def plot_cholqr_v4_layer_groups(
    target_rank: int = 512,
    target_rv: int = 768,
    target_niter: int = 8,
) -> Path:
    """
    For a fixed rank/rv/n_iter, compare different LGS values for cholqr_v4 using:
      - total latency (avg_time_ms)
      - accuracy (from log)
      - latency breakdown (avg_breakdown_ms)
    """
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "results" / "cholqr_v4" / "layer_group_size"

    candidate_lgs = [2, 4, 8, 16]

    lgs_values: List[int] = []
    latencies: List[float] = []
    accuracies: List[float] = []
    breakdowns: List[Dict[str, float]] = []

    print(
        f"Loading cholqr_v4 layer_group_size data (rank={target_rank}, "
        f"RV={target_rv}, n_iter={target_niter})..."
    )

    for lgs in candidate_lgs:
        json_path = (
            base_dir
            / f"xKV_LGS{lgs}_RK{target_rank}_RV{target_rv}_NITER{target_niter}_svd_benchmark.json"
        )
        log_path = (
            base_dir
            / f"xKV_LGS{lgs}_RK{target_rank}_RV{target_rv}_NITER{target_niter}.log"
        )

        if not json_path.exists():
            print(f"  Skipping LGS={lgs}: JSON not found -> {json_path.name}")
            continue

        try:
            latency, breakdown = load_cholqr_v4_stats_from_json(
                json_path, target_rank=target_rank
            )
        except Exception as e:
            print(f"  ERROR reading {json_path.name}: {e}")
            continue

        acc = None
        if log_path.exists():
            try:
                acc = load_accuracy_from_log(log_path)
            except Exception as e:
                print(f"  Warning: could not parse accuracy from {log_path.name}: {e}")

        lgs_values.append(lgs)
        latencies.append(latency)
        accuracies.append(acc)
        breakdowns.append(breakdown)

        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        print(
            f"  LGS={lgs:2d}: latency={latency:7.2f} ms, "
            f"accuracy={acc_str}"
        )

    if not lgs_values:
        raise RuntimeError("No layer_group_size data loaded for cholqr_v4.")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)

    # 1) Latency vs LGS
    ax1 = axes[0]
    ax1.plot(lgs_values, latencies, marker="o", linewidth=2, color="#2CA02C")
    ax1.set_xlabel("Layer Group Size (LGS)", fontsize=12)
    ax1.set_ylabel("Average Latency (ms)", fontsize=12)
    ax1.set_title("cholqr_v4 - Latency vs Layer Group Size", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_xticks(lgs_values)
    for lgs, lat in zip(lgs_values, latencies):
        ax1.text(lgs, lat, f"{lat:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2) Accuracy vs LGS
    ax2 = axes[1]
    valid_acc = [(l, a) for l, a in zip(lgs_values, accuracies) if a is not None]
    if valid_acc:
        lgs_acc, acc_vals = zip(*valid_acc)
        ax2.plot(lgs_acc, acc_vals, marker="o", linewidth=2, color="#9467BD")
        ax2.set_xlabel("Layer Group Size (LGS)", fontsize=12)
        ax2.set_ylabel("Accuracy (Baseline Score)", fontsize=12)
        ax2.set_title("cholqr_v4 - Accuracy vs Layer Group Size", fontsize=13, fontweight="bold")
        ax2.grid(alpha=0.3, linestyle="--")
        ax2.set_xticks(lgs_values)
        for lgs, acc in valid_acc:
            ax2.text(lgs, acc, f"{acc:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        ax2.text(
            0.5,
            0.5,
            "No accuracy data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("cholqr_v4 - Accuracy vs Layer Group Size", fontsize=13, fontweight="bold")

    # 3) Latency Breakdown (stacked)
    ax3 = axes[2]
    stage_order = [
        "Preparation",
        "Projection (BMM)",
        "Power Iteration",
        "Final QR",
        "Project to Low-rank",
        "Small SVD",
        "Reconstruction",
    ]
    stage_colors = [
        "#8c564b",
        "#17becf",
        "#ff7f0e",
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]

    x_pos = range(len(lgs_values))
    bottom = [0.0] * len(lgs_values)
    for stage, color in zip(stage_order, stage_colors):
        values = []
        for bd in breakdowns:
            values.append(bd.get(stage, 0.0) if bd is not None else 0.0)
        ax3.bar(x_pos, values, bottom=bottom, label=stage, color=color, alpha=0.85)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax3.set_xlabel("Layer Group Size (LGS)", fontsize=12)
    ax3.set_ylabel("Latency (ms)", fontsize=12)
    ax3.set_title("cholqr_v4 - Latency Breakdown", fontsize=13, fontweight="bold")
    ax3.set_xticks(list(x_pos))
    ax3.set_xticklabels([str(l) for l in lgs_values])
    ax3.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.suptitle(
        f"cholqr_v4 Layer Group Size Effect (rank={target_rank}, RV={target_rv}, n_iter={target_niter})",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    out_dir = repo_root / "plot" / "cholqr_v4"
    out_path = out_dir / f"fig_cholqr_v4_layer_groups_RK{target_rank}_RV{target_rv}_NITER{target_niter}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved layer_group_size figure to: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="cholqr_v4 detailed comparison plots "
        "(power_iteration & layer_group_size)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["power_iter", "layer_group", "both"],
        help="Which cholqr_v4 comparison to generate",
    )
    parser.add_argument("--lgs", type=int, default=8, help="Layer Group Size")
    parser.add_argument("--rank", type=int, default=512, help="Rank value")
    parser.add_argument("--rv", type=int, default=768, help="Rank-Value (RV)")
    parser.add_argument("--n_iter", type=int, default=8, help="Power iteration (n_iter)")

    args = parser.parse_args()

    if args.mode in ("power_iter", "both"):
        plot_cholqr_v4_power_iterations(
            target_lgs=args.lgs,
            target_rank=args.rank,
            target_rv=args.rv,
        )

    if args.mode in ("layer_group", "both"):
        plot_cholqr_v4_layer_groups(
            target_rank=args.rank,
            target_rv=args.rv,
            target_niter=args.n_iter,
        )


if __name__ == "__main__":
    main()

