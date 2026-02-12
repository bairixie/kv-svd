import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def load_avg_time_from_json_exact_or_match(
    path: Path, target_rank: int, prefer: str | None = None, target_niter: int | None = None
) -> float:
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


def load_avg_time_from_breakdown_log(path: Path, target_rank: int = 256) -> float:
    """
    Parse breakdown log file to extract average time for a specific rank.
    Looks for sections like:
      Method: Custom Randomized SVD (Rank 256)
      Average Time: 180.80 ms
    """
    txt = path.read_text()
    
    # Find the section for the target rank
    rank_pattern = rf"Method: Custom Randomized SVD \(Rank {target_rank}\)"
    rank_match = re.search(rank_pattern, txt)
    
    if not rank_match:
        # Fallback: try to find any "Average Time" if rank section not found
        # (for files that only have one rank)
        avg_match = re.search(r"Average Time:\s*([\d\.]+)\s*ms", txt)
        if avg_match:
            return float(avg_match.group(1))
        raise ValueError(f"Could not find rank {target_rank} section or 'Average Time' in {path}")
    
    # Extract text after the rank section
    start_pos = rank_match.end()
    section_text = txt[start_pos:start_pos + 2000]  # Look in next 2000 chars
    
    # Find Average Time in this section
    avg_match = re.search(r"Average Time:\s*([\d\.]+)\s*ms", section_text)
    if not avg_match:
        raise ValueError(f"Could not find 'Average Time' in rank {target_rank} section of {path}")
    
    return float(avg_match.group(1))


def load_latency_breakdown_from_log(path: Path, target_rank: int = 256) -> dict:
    """
    Parse breakdown log file to extract latency breakdown for all stages.
    Returns a dict with stage names and their average times in ms.
    """
    txt = path.read_text()
    
    # Find the section for the target rank
    rank_pattern = rf"Method: Custom Randomized SVD \(Rank {target_rank}\)"
    rank_match = re.search(rank_pattern, txt)
    
    if not rank_match:
        raise ValueError(f"Could not find rank {target_rank} section in {path}")
    
    # Extract text after the rank section
    start_pos = rank_match.end()
    section_text = txt[start_pos:start_pos + 5000]  # Look in next 5000 chars
    
    # Parse each stage
    breakdown = {}
    stage_patterns = [
        (r"1\. Preparation\s*:\s*([\d\.]+)\s*ms", "Preparation"),
        (r"2\. Projection \(BMM\)\s*:\s*([\d\.]+)\s*ms", "Projection (BMM)"),
        (r"3\. Power Iteration \(Total\)\s*:\s*([\d\.]+)\s*ms", "Power Iteration"),
        (r"4\. Final QR\s*:\s*([\d\.]+)\s*ms", "Final QR"),
        (r"5\. Project to Low-rank \(BMM\)\s*:\s*([\d\.]+)\s*ms", "Project to Low-rank"),
        (r"6\. Small SVD \(FP32\)\s*:\s*([\d\.]+)\s*ms", "Small SVD"),
        (r"7\. Reconstruction \(BMM\)\s*:\s*([\d\.]+)\s*ms", "Reconstruction"),
    ]
    
    for pattern, name in stage_patterns:
        match = re.search(pattern, section_text)
        if match:
            breakdown[name] = float(match.group(1))
        else:
            breakdown[name] = 0.0
    
    return breakdown


def load_power_iteration_time_from_breakdown_log(path: Path, target_rank: int = 256) -> float:
    """
    Parse breakdown log file to extract Power Iteration time for a specific rank.
    Looks for sections like:
      Method: Custom Randomized SVD (Rank 256)
      3. Power Iteration (Total): 37.65 ms
    """
    txt = path.read_text()
    
    # Find the section for the target rank
    rank_pattern = rf"Method: Custom Randomized SVD \(Rank {target_rank}\)"
    rank_match = re.search(rank_pattern, txt)
    
    if not rank_match:
        # Fallback: try to find any Power Iteration time
        pi_match = re.search(r"3\. Power Iteration \(Total\)\s*:\s*([\d\.]+)\s*ms", txt)
        if pi_match:
            return float(pi_match.group(1))
        raise ValueError(f"Could not find rank {target_rank} section or 'Power Iteration' in {path}")
    
    # Extract text after the rank section
    start_pos = rank_match.end()
    section_text = txt[start_pos:start_pos + 3000]  # Look in next 3000 chars
    
    # Find Power Iteration time in this section
    pi_match = re.search(r"3\. Power Iteration \(Total\)\s*:\s*([\d\.]+)\s*ms", section_text)
    if not pi_match:
        raise ValueError(f"Could not find 'Power Iteration' time in rank {target_rank} section of {path}")
    
    return float(pi_match.group(1))


def load_accuracy_from_log(path: Path) -> float:
    """
    Parse log file to extract accuracy (baseline score).
    Looks for lines like:
      Results for ruler/vt: {'model': '...', 'dataset': 'ruler/vt', 'baseline': 0.408333, ...}
    or:
      | meta-llama/Meta-Llama-3.1-8B-Instruct | ruler/vt  |   0.408333 |        96 |
    """
    txt = path.read_text()
    
    # Try to find baseline in Results line
    baseline_match = re.search(r"'baseline':\s*([\d\.]+)", txt)
    if baseline_match:
        return float(baseline_match.group(1))
    
    # Fallback: try to find in table format
    table_match = re.search(r"ruler/vt\s*\|\s*([\d\.]+)\s*\|", txt)
    if table_match:
        return float(table_match.group(1))
    
    raise ValueError(f"Could not find 'baseline' or accuracy value in {path}")


def main(lgs: int = 4, rank: int = 256, rank_value: int = 384, n_iter: int = 8):
    """
    Main function to generate all methods comparison plot.
    
    Args:
        lgs: Layer Group Size (2, 4, 8, 16, etc.)
        rank: Rank value (256, 384, 512, 768, etc.)
        rank_value: Rank Value (usually rank * 1.5, e.g., 384 for rank 256)
        n_iter: Number of power iterations (2, 4, 6, 8, 10, 16, etc.)
    """
    # Repo root: kv-svd/<...>/plot/this_file.py -> parents[1] == kv-svd
    repo_root = Path(__file__).resolve().parents[1]

    # Configuration
    target_rank = rank
    target_niter = n_iter
    target_lgs = lgs
    target_rv = rank_value

    # Generate filename pattern based on parameters
    file_prefix = f"xKV_LGS{target_lgs}_RK{target_rank}_RV{target_rv}"
    
    # Paths to existing benchmark files (JSON and log)
    # Try standard format first, then fallback to alternative format
    path_full_json = (
        repo_root
        / "results"
        / "full_svd"
        / f"{file_prefix}_FULL_SVD_svd_benchmark.json"
    )
    if not path_full_json.exists():
        # Try alternative format: full_svd_LGS8_RK512_RV768_svd_benchmark.json
        path_full_json = (
            repo_root
            / "results"
            / "full_svd"
            / f"full_svd_LGS{target_lgs}_RK{target_rank}_RV{target_rv}_svd_benchmark.json"
        )
    
    path_full_log = (
        repo_root
        / "results"
        / "full_svd"
        / f"{file_prefix}_FULL_SVD.log"
    )
    if not path_full_log.exists():
        # Try alternative format
        path_full_log = (
            repo_root
            / "results"
            / "full_svd"
            / f"full_svd_LGS{target_lgs}_RK{target_rank}_RV{target_rv}.log"
        )
    # Try multiple filename formats for lowrank_svd
    path_lowrank_json = (
        repo_root
        / "results"
        / "lowrank_svd"
        / f"{file_prefix}_LOWRANK_NITER{target_niter}_svd_benchmark.json"
    )
    if not path_lowrank_json.exists():
        # Try alternative format without LOWRANK prefix (for some configurations)
        alt_json = (
            repo_root
            / "results"
            / "lowrank_svd"
            / f"{file_prefix}_NITER{target_niter}_svd_benchmark.json"
        )
        if alt_json.exists():
            path_lowrank_json = alt_json
    
    path_lowrank_log = (
        repo_root
        / "results"
        / "lowrank_svd"
        / f"{file_prefix}_LOWRANK_NITER{target_niter}.log"
    )
    if not path_lowrank_log.exists():
        # Try alternative format without LOWRANK prefix
        alt_log = (
            repo_root
            / "results"
            / "lowrank_svd"
            / f"{file_prefix}_NITER{target_niter}.log"
        )
        if alt_log.exists():
            path_lowrank_log = alt_log
    # Try power_iteration directory first, then fallback to layer_group_size
    path_cholqr_v1_json = (
        repo_root
        / "results"
        / "cholqr_v1"
        / "power_iteration"
        / f"{file_prefix}_NITER{target_niter}_svd_benchmark.json"
    )
    if not path_cholqr_v1_json.exists():
        path_cholqr_v1_json = (
            repo_root
            / "results"
            / "cholqr_v1"
            / "layer_group_size"
            / f"{file_prefix}_NITER{target_niter}_svd_benchmark.json"
        )
    
    path_cholqr_v1_log = (
        repo_root
        / "results"
        / "cholqr_v1"
        / "power_iteration"
        / f"{file_prefix}_NITER{target_niter}.log"
    )
    if not path_cholqr_v1_log.exists():
        path_cholqr_v1_log = (
            repo_root
            / "results"
            / "cholqr_v1"
            / "layer_group_size"
            / f"{file_prefix}_NITER{target_niter}.log"
        )
    
    path_cholqr_v2_breakdown = (
        repo_root
        / "results"
        / "cholqr_v2"
        / "power_iteration"
        / f"{file_prefix}_NITER{target_niter}_svd_benchmark_breakdown.log"
    )
    if not path_cholqr_v2_breakdown.exists():
        path_cholqr_v2_breakdown = (
            repo_root
            / "results"
            / "cholqr_v2"
            / "layer_group_size"
            / f"{file_prefix}_NITER{target_niter}_svd_benchmark_breakdown.log"
        )
    
    path_cholqr_v2_log = (
        repo_root
        / "results"
        / "cholqr_v2"
        / "power_iteration"
        / f"{file_prefix}_NITER{target_niter}.log"
    )
    if not path_cholqr_v2_log.exists():
        path_cholqr_v2_log = (
            repo_root
            / "results"
            / "cholqr_v2"
            / "layer_group_size"
            / f"{file_prefix}_NITER{target_niter}.log"
        )
    
    path_cholqr_v3_breakdown = (
        repo_root
        / "results"
        / "cholqr_v3"
        / "power_iteration_chol"
        / f"{file_prefix}_NITER{target_niter}_svd_benchmark_breakdown.log"
    )
    if not path_cholqr_v3_breakdown.exists():
        path_cholqr_v3_breakdown = (
            repo_root
            / "results"
            / "cholqr_v3"
            / "layer_group_size_chol"
            / f"{file_prefix}_NITER{target_niter}_svd_benchmark_breakdown.log"
        )
    
    path_cholqr_v3_log = (
        repo_root
        / "results"
        / "cholqr_v3"
        / "power_iteration_chol"
        / f"{file_prefix}_NITER{target_niter}.log"
    )
    if not path_cholqr_v3_log.exists():
        path_cholqr_v3_log = (
            repo_root
            / "results"
            / "cholqr_v3"
            / "layer_group_size_chol"
            / f"{file_prefix}_NITER{target_niter}.log"
        )

    # cholqr_v4 (new): try power_iterations first, then layer_group_size
    path_cholqr_v4_json = (
        repo_root
        / "results"
        / "cholqr_v4"
        / "power_iterations"
        / f"{file_prefix}_NITER{target_niter}_svd_benchmark.json"
    )
    if not path_cholqr_v4_json.exists():
        path_cholqr_v4_json = (
            repo_root
            / "results"
            / "cholqr_v4"
            / "layer_group_size"
            / f"{file_prefix}_NITER{target_niter}_svd_benchmark.json"
        )

    path_cholqr_v4_log = (
        repo_root
        / "results"
        / "cholqr_v4"
        / "power_iterations"
        / f"{file_prefix}_NITER{target_niter}.log"
    )
    if not path_cholqr_v4_log.exists():
        path_cholqr_v4_log = (
            repo_root
            / "results"
            / "cholqr_v4"
            / "layer_group_size"
            / f"{file_prefix}_NITER{target_niter}.log"
        )

    # Load data: latency, accuracy, and power_iteration time
    print(f"Loading data for LGS={target_lgs}, rank={target_rank}, RV={target_rv}, n_iter={target_niter}...")
    print(f"Looking for files with pattern: {file_prefix}...")
    
    data = {}  # Store all data: {method_name: {"latency": float, "accuracy": float, "power_iter": float}}
    
    # Full SVD
    try:
        if not path_full_json.exists():
            print(f"  ⚠ Full SVD: JSON file not found: {path_full_json.name}")
            data["Full SVD"] = None
        else:
            data["Full SVD"] = {
                "latency": load_avg_time_from_json_exact_or_match(
                    path_full_json,
                    target_rank=target_rank,
                    prefer=f"Full SVD (torch.linalg.svd) (rank={target_rank})",
                ),
                "accuracy": load_accuracy_from_log(path_full_log) if path_full_log.exists() else None,
                "power_iter": None,  # Full SVD doesn't use power iteration
            }
            acc_str = f"{data['Full SVD']['accuracy']:.4f}" if data['Full SVD']['accuracy'] is not None else "N/A"
            print(f"  ✓ Full SVD: latency={data['Full SVD']['latency']:.2f} ms, accuracy={acc_str}")
    except Exception as e:
        print(f"  ✗ ERROR loading Full SVD: {e}")
        data["Full SVD"] = None

    # Lowrank SVD
    try:
        if not path_lowrank_json.exists():
            # Check if log file exists and has accuracy data
            if path_lowrank_log.exists():
                accuracy = load_accuracy_from_log(path_lowrank_log)
                if accuracy is not None:
                    # Even without JSON, we can still show accuracy in the comparison
                    print(f"  ⚠ Lowrank SVD: JSON file not found: {path_lowrank_json.name}")
                    print(f"     Using accuracy from log file (latency data unavailable)")
                    data["Lowrank SVD"] = {
                        "latency": None,  # No latency data available
                        "accuracy": accuracy,
                        "power_iter": None,
                    }
                    print(f"  ✓ Lowrank SVD: latency=N/A, accuracy={accuracy:.4f}")
                else:
                    print(f"  ⚠ Lowrank SVD: JSON file not found and no accuracy data in log")
                    data["Lowrank SVD"] = None
            else:
                print(f"  ⚠ Lowrank SVD: Both JSON and log files not found")
                print(f"     Tried: {path_lowrank_json.name}")
                print(f"     Tried: {path_lowrank_log.name if path_lowrank_log else 'N/A'}")
                data["Lowrank SVD"] = None
        else:
            data["Lowrank SVD"] = {
                "latency": load_avg_time_from_json_exact_or_match(
                    path_lowrank_json,
                    target_rank=target_rank,
                    target_niter=target_niter,
                ),
                "accuracy": load_accuracy_from_log(path_lowrank_log) if path_lowrank_log.exists() else None,
                "power_iter": None,  # Lowrank SVD uses internal iterations, not tracked separately
            }
            acc_str = f"{data['Lowrank SVD']['accuracy']:.4f}" if data['Lowrank SVD']['accuracy'] is not None else "N/A"
            print(f"  ✓ Lowrank SVD: latency={data['Lowrank SVD']['latency']:.2f} ms, accuracy={acc_str}")
    except Exception as e:
        print(f"  ✗ ERROR loading Lowrank SVD: {e}")
        data["Lowrank SVD"] = None

    # cholqr_v1
    try:
        if not path_cholqr_v1_json.exists():
            print(f"  ⚠ cholqr_v1: JSON file not found: {path_cholqr_v1_json.name}")
            data["cholqr_v1"] = None
        else:
            data["cholqr_v1"] = {
                "latency": load_avg_time_from_json_exact_or_match(
                    path_cholqr_v1_json,
                    target_rank=target_rank,
                    prefer=f"Custom Randomized SVD (rank={target_rank})",
                ),
                "accuracy": load_accuracy_from_log(path_cholqr_v1_log) if path_cholqr_v1_log.exists() else None,
                "power_iter": None,  # cholqr_v1 doesn't have breakdown log with power_iter time
            }
            acc_str = f"{data['cholqr_v1']['accuracy']:.4f}" if data['cholqr_v1']['accuracy'] is not None else "N/A"
            print(f"  ✓ cholqr_v1: latency={data['cholqr_v1']['latency']:.2f} ms, accuracy={acc_str}")
    except Exception as e:
        print(f"  ✗ ERROR loading cholqr_v1: {e}")
        data["cholqr_v1"] = None

    # cholqr_v2
    try:
        if not path_cholqr_v2_breakdown.exists():
            print(f"  ⚠ cholqr_v2: Breakdown log not found: {path_cholqr_v2_breakdown.name}")
            data["cholqr_v2"] = None
        else:
            data["cholqr_v2"] = {
                "latency": load_avg_time_from_breakdown_log(path_cholqr_v2_breakdown, target_rank=target_rank),
                "accuracy": load_accuracy_from_log(path_cholqr_v2_log) if path_cholqr_v2_log.exists() else None,
                "power_iter": load_power_iteration_time_from_breakdown_log(path_cholqr_v2_breakdown, target_rank=target_rank),
            }
            acc_str = f"{data['cholqr_v2']['accuracy']:.4f}" if data['cholqr_v2']['accuracy'] is not None else "N/A"
            print(f"  ✓ cholqr_v2: latency={data['cholqr_v2']['latency']:.2f} ms, accuracy={acc_str}, power_iter={data['cholqr_v2']['power_iter']:.2f} ms")
    except Exception as e:
        print(f"  ✗ ERROR loading cholqr_v2: {e}")
        data["cholqr_v2"] = None

    # cholqr_v3
    try:
        if not path_cholqr_v3_breakdown.exists():
            print(f"  ⚠ cholqr_v3: Breakdown log not found: {path_cholqr_v3_breakdown.name}")
            data["cholqr_v3"] = None
        else:
            data["cholqr_v3"] = {
                "latency": load_avg_time_from_breakdown_log(path_cholqr_v3_breakdown, target_rank=target_rank),
                "accuracy": load_accuracy_from_log(path_cholqr_v3_log) if path_cholqr_v3_log.exists() else None,
                "power_iter": load_power_iteration_time_from_breakdown_log(path_cholqr_v3_breakdown, target_rank=target_rank),
            }
            acc_str = f"{data['cholqr_v3']['accuracy']:.4f}" if data['cholqr_v3']['accuracy'] is not None else "N/A"
            print(f"  ✓ cholqr_v3: latency={data['cholqr_v3']['latency']:.2f} ms, accuracy={acc_str}, power_iter={data['cholqr_v3']['power_iter']:.2f} ms")
    except Exception as e:
        print(f"  ✗ ERROR loading cholqr_v3: {e}")
        data["cholqr_v3"] = None

    # cholqr_v4
    try:
        if not path_cholqr_v4_json.exists():
            print(f"  ⚠ cholqr_v4: JSON file not found: {path_cholqr_v4_json.name}")
            data["cholqr_v4"] = None
        else:
            data["cholqr_v4"] = {
                "latency": load_avg_time_from_json_exact_or_match(
                    path_cholqr_v4_json,
                    target_rank=target_rank,
                    # aggregated summary uses keys like "randomized (Rank 256) (rank=256)"
                    prefer=f"randomized (Rank {target_rank}) (rank={target_rank})",
                ),
                "accuracy": load_accuracy_from_log(path_cholqr_v4_log)
                if path_cholqr_v4_log.exists()
                else None,
                "power_iter": None,  # power_iteration time already reflected in total latency
            }
            acc_str = (
                f"{data['cholqr_v4']['accuracy']:.4f}"
                if data["cholqr_v4"]["accuracy"] is not None
                else "N/A"
            )
            print(
                f"  ✓ cholqr_v4: latency={data['cholqr_v4']['latency']:.2f} ms, "
                f"accuracy={acc_str}"
            )
    except Exception as e:
        print(f"  ✗ ERROR loading cholqr_v4: {e}")
        data["cholqr_v4"] = None

    # Filter out None values and build lists
    # Separate lists for latency (requires latency data) and accuracy (can work with just accuracy)
    methods_latency = []
    latencies = []
    methods_accuracy = []
    accuracies = []
    power_iters = []
    colors_latency = []
    colors_accuracy = []
    
    method_order = ["Full SVD", "Lowrank SVD", "cholqr_v1", "cholqr_v2", "cholqr_v3", "cholqr_v4"]
    method_labels = {
        "Full SVD": "Full SVD",
        "Lowrank SVD": f"Lowrank SVD\n(niter={target_niter})",
        "cholqr_v1": "Randomized SVD\n(cholqr_v1)",
        "cholqr_v2": "Randomized SVD\n(cholqr_v2)",
        "cholqr_v3": "Randomized SVD\n(cholqr_v3)",
        "cholqr_v4": "Randomized SVD\n(cholqr_v4)",
    }
    method_colors = {
        "Full SVD": "#D62728",  # red
        "Lowrank SVD": "#FF7F0E",  # orange
        "cholqr_v1": "#1F77B4",  # blue
        "cholqr_v2": "#9467BD",  # purple
        "cholqr_v3": "#2CA02C",  # green
        "cholqr_v4": "#FF1493",  # pink
    }
    
    # Build latency list (requires latency data)
    for method in method_order:
        if data.get(method) is not None and data[method].get("latency") is not None:
            methods_latency.append(method_labels[method])
            latencies.append(data[method]["latency"])
            colors_latency.append(method_colors[method])
    
    # Build accuracy list (can include methods with only accuracy data)
    for method in method_order:
        if data.get(method) is not None:
            accuracy = data[method].get("accuracy")
            if accuracy is not None:
                methods_accuracy.append(method_labels[method])
                accuracies.append(accuracy)
                colors_accuracy.append(method_colors[method])
                # Also track power_iter if available
                if data[method].get("power_iter") is not None:
                    power_iters.append(data[method].get("power_iter"))
                else:
                    power_iters.append(None)

    if not methods_latency and not methods_accuracy:
        print("\n❌ ERROR: No data loaded! Check file paths.")
        print(f"\n💡 Tip: Try running with different parameters, or check available files:")
        print(f"   Available configurations can be found in: results/")
        print(f"   Look for files matching pattern: xKV_LGS*_RK*_RV*_NITER*")
        return
    
    print(f"\n✓ Successfully loaded {len(methods_latency)} method(s) with latency data")
    if methods_accuracy:
        print(f"✓ Successfully loaded {len(methods_accuracy)} method(s) with accuracy data")

    # Create figure with 2 subplots (latency and accuracy)
    # Power iteration comparison is done separately for each algorithm
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    
    # Subplot 1: Latency
    ax1 = axes[0]
    if methods_latency:
        bars1 = ax1.bar(range(len(methods_latency)), latencies, color=colors_latency)
        ax1.set_ylabel("Average Latency (ms)", fontsize=12)
        ax1.set_xticks(range(len(methods_latency)))
        ax1.set_xticklabels(methods_latency, rotation=15, ha="right", fontsize=10)
        ax1.set_title("Latency Comparison", fontsize=13, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3, linestyle="--")
        for bar, t in zip(bars1, latencies):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{t:.0f} ms",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    else:
        ax1.text(0.5, 0.5, "No latency data available", 
                ha="center", va="center", transform=ax1.transAxes, fontsize=12)
        ax1.set_title("Latency Comparison", fontsize=13, fontweight="bold")
    
    # Subplot 2: Accuracy
    ax2 = axes[1]
    if methods_accuracy:
        bars2 = ax2.bar(range(len(methods_accuracy)), accuracies, color=colors_accuracy)
        ax2.set_ylabel("Accuracy (Baseline Score)", fontsize=12)
        ax2.set_xticks(range(len(methods_accuracy)))
        ax2.set_xticklabels(methods_accuracy, rotation=15, ha="right", fontsize=10)
        ax2.set_title("Accuracy Comparison", fontsize=13, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        for bar, acc in zip(bars2, accuracies):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{acc:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    else:
        ax2.text(0.5, 0.5, "No accuracy data available", 
                ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Accuracy Comparison", fontsize=13, fontweight="bold")
    
    
    # Overall title
    fig.suptitle(
        f"SVD Methods Comparison (LGS={target_lgs}, rank={target_rank}, RV={target_rv}, n_iter={target_niter})\n"
        "Full SVD vs Lowrank SVD vs Randomized SVD (cholqr_v1/v2/v3)\n"
        "Note: Power iteration comparison is available separately for each algorithm",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    
    plt.tight_layout()

    # Save to plot directory with updated filename
    out_path = repo_root / "plot" / f"fig_all_methods_comparison_LGS{target_lgs}_RK{target_rank}_RV{target_rv}_NITER{target_niter}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {out_path}")
    print("\nSummary:")
    print(f"{'Method':<30s} {'Latency (ms)':>12s} {'Accuracy':>12s}")
    print("-" * 60)
    # Combine all methods for summary (show latency if available, accuracy if available)
    all_methods_dict = {}
    for i, method in enumerate(methods_latency):
        all_methods_dict[method] = {"latency": latencies[i], "accuracy": None}
    for i, method in enumerate(methods_accuracy):
        if method not in all_methods_dict:
            all_methods_dict[method] = {"latency": None, "accuracy": accuracies[i]}
        else:
            all_methods_dict[method]["accuracy"] = accuracies[i]
    
    for method, data_dict in all_methods_dict.items():
        lat_str = f"{data_dict['latency']:.2f}" if data_dict['latency'] is not None else "N/A"
        acc_str = f"{data_dict['accuracy']:.4f}" if data_dict['accuracy'] is not None else "N/A"
        print(f"{method.replace(chr(10), ' '):<30s} {lat_str:>12s} {acc_str:>12s}")
    print("\nNote: For power iteration comparison, run:")
    print("  python -m plot.fig_all_methods_comparison power_iter --algorithm cholqr_v1 --rank 256 --lgs 4 --rv 384")
    print("  python -m plot.fig_all_methods_comparison power_iter --algorithm cholqr_v2 --rank 256 --lgs 4 --rv 384")
    print("  python -m plot.fig_all_methods_comparison power_iter --algorithm cholqr_v3 --rank 256 --lgs 4 --rv 384")
    print("  python -m plot.fig_all_methods_comparison all_power_iter --rank 256 --lgs 4 --rv 384  # Generate all at once")
    print("\nFor different configurations, use:")
    print("  --lgs LGS    Layer Group Size (2, 4, 8, 16, etc.)")
    print("  --rank RANK  Rank value (256, 384, 512, 768, etc.)")
    print("  --rv RV      Rank Value (usually rank * 1.5)")
    print("  --n_iter N   Number of power iterations")


def plot_power_iteration_comparison(
    algorithm: str = "cholqr_v3",
    target_rank: int = 256,
    target_lgs: int = 4,
    target_rv: int = 384,
    target_niter: int = 8,
    experiment_type: str = "power_iteration",
):
    """
    Plot comparison of the same algorithm with different power_iteration (n_iter) values.
    Shows latency, accuracy, and latency breakdown for different n_iter values.
    
    Args:
        algorithm: "cholqr_v1", "cholqr_v2", or "cholqr_v3"
        target_rank: target rank value (256, 384, 512, 768, etc.)
        target_lgs: Layer Group Size (2, 4, 8, 16, etc.)
        target_rv: Rank Value (usually rank * 1.5, e.g., 384 for rank 256)
        experiment_type: "power_iteration" or "layer_group_size"
    """
    repo_root = Path(__file__).resolve().parents[1]
    
    # Determine base path and available values based on algorithm and experiment type
    # Also set fallback paths for when primary directory doesn't have data
    if algorithm == "cholqr_v1":
        if experiment_type == "power_iteration":
            base_path = repo_root / "results" / "cholqr_v1" / "power_iteration"
            fallback_path = repo_root / "results" / "cholqr_v1" / "layer_group_size"
            comparison_values = [2, 4, 8, 16]  # n_iter values
            is_lgs_comparison = False
        else:  # layer_group_size
            base_path = repo_root / "results" / "cholqr_v1" / "layer_group_size"
            fallback_path = None
            comparison_values = [2, 4, 8]  # LGS values (based on available files)
            is_lgs_comparison = True
        has_breakdown = False  # cholqr_v1 doesn't have breakdown logs
    elif algorithm == "cholqr_v2":
        if experiment_type == "power_iteration":
            base_path = repo_root / "results" / "cholqr_v2" / "power_iteration"
            fallback_path = repo_root / "results" / "cholqr_v2" / "layer_group_size"
            comparison_values = [2, 4, 8, 16]  # n_iter values
            is_lgs_comparison = False
        else:  # layer_group_size
            base_path = repo_root / "results" / "cholqr_v2" / "layer_group_size"
            fallback_path = None
            comparison_values = [2, 4, 8, 16]  # LGS values (based on available files)
            is_lgs_comparison = True
        has_breakdown = True
    elif algorithm == "cholqr_v3":
        if experiment_type == "power_iteration":
            base_path = repo_root / "results" / "cholqr_v3" / "power_iteration_chol"
            fallback_path = repo_root / "results" / "cholqr_v3" / "layer_group_size_chol"
            comparison_values = [2, 4, 6, 8, 10, 16]  # n_iter values
            is_lgs_comparison = False
        else:  # layer_group_size
            base_path = repo_root / "results" / "cholqr_v3" / "layer_group_size_chol"
            fallback_path = None
            comparison_values = [2, 4, 8]  # LGS values (based on available files)
            is_lgs_comparison = True
        has_breakdown = True
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'cholqr_v1', 'cholqr_v2', or 'cholqr_v3'")
    
    print(f"Loading {experiment_type} comparison data for {algorithm} (rank={target_rank}, RV={target_rv})...")
    
    latencies = []
    accuracies = []
    breakdowns = []  # List of breakdown dicts for each comparison value
    valid_values = []
    
    for comp_value in comparison_values:
        try:
            if is_lgs_comparison:
                # For layer_group_size comparison, LGS varies, n_iter is fixed
                file_prefix = f"xKV_LGS{comp_value}_RK{target_rank}_RV{target_rv}"
                n_iter_fixed = target_niter
            else:
                # For power_iteration comparison, n_iter varies, LGS is fixed
                file_prefix = f"xKV_LGS{target_lgs}_RK{target_rank}_RV{target_rv}"
                n_iter_fixed = comp_value
            
            # Try primary path first, then fallback if available
            current_base = base_path
            if has_breakdown:
                breakdown_path = current_base / f"{file_prefix}_NITER{n_iter_fixed}_svd_benchmark_breakdown.log"
                if not breakdown_path.exists() and fallback_path is not None:
                    breakdown_path = fallback_path / f"{file_prefix}_NITER{n_iter_fixed}_svd_benchmark_breakdown.log"
                    if breakdown_path.exists():
                        current_base = fallback_path
                
                if not breakdown_path.exists():
                    print(f"  Skipping {experiment_type}={comp_value}: breakdown log not found")
                    continue
                # Load latency from breakdown log
                latency = load_avg_time_from_breakdown_log(breakdown_path, target_rank=target_rank)
                # Load latency breakdown
                breakdown = load_latency_breakdown_from_log(breakdown_path, target_rank=target_rank)
            else:
                # For cholqr_v1, load from JSON
                json_path = current_base / f"{file_prefix}_NITER{n_iter_fixed}_svd_benchmark.json"
                if not json_path.exists() and fallback_path is not None:
                    json_path = fallback_path / f"{file_prefix}_NITER{n_iter_fixed}_svd_benchmark.json"
                    if json_path.exists():
                        current_base = fallback_path
                
                if not json_path.exists():
                    print(f"  Skipping {experiment_type}={comp_value}: JSON file not found")
                    continue
                # Load latency from JSON
                latency = load_avg_time_from_json_exact_or_match(
                    json_path,
                    target_rank=target_rank,
                    prefer=f"Custom Randomized SVD (rank={target_rank})",
                )
                # No breakdown data for cholqr_v1
                breakdown = None
            
            log_path = current_base / f"{file_prefix}_NITER{n_iter_fixed}.log"
            
            # Load accuracy
            accuracy = None
            if log_path.exists():
                try:
                    accuracy = load_accuracy_from_log(log_path)
                except Exception as e:
                    print(f"  Warning: Could not load accuracy for {experiment_type}={comp_value}: {e}")
            
            latencies.append(latency)
            accuracies.append(accuracy)
            breakdowns.append(breakdown)
            valid_values.append(comp_value)
            
            acc_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
            if is_lgs_comparison:
                print(f"  LGS={comp_value:2d}: latency={latency:7.2f} ms, accuracy={acc_str}")
            else:
                print(f"  n_iter={comp_value:2d}: latency={latency:7.2f} ms, accuracy={acc_str}")
            
        except Exception as e:
            print(f"  ERROR loading {experiment_type}={comp_value}: {e}")
            continue
    
    if not valid_values:
        print(f"ERROR: No data loaded for {algorithm}!")
        return
    
    # Check if we have breakdown data
    has_breakdown_data = any(b is not None for b in breakdowns)
    
    # Create figure with 2 or 3 subplots depending on breakdown data availability
    if has_breakdown_data:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    
    # Determine x-axis label and title based on comparison type
    if is_lgs_comparison:
        x_label = "Layer Group Size (LGS)"
        title_suffix = "Layer Group Size"
    else:
        x_label = "Power Iteration (n_iter)"
        title_suffix = "Power Iteration"
    
    # Subplot 1: Latency vs comparison value
    ax1 = axes[0]
    ax1.plot(valid_values, latencies, marker='o', linewidth=2, markersize=8, color="#2CA02C")
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel("Total Latency (ms)", fontsize=12)
    ax1.set_title(f"{algorithm.upper()} - Latency vs {title_suffix}", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_xticks(valid_values)
    for val, lat in zip(valid_values, latencies):
        ax1.text(val, lat, f"{lat:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    # Subplot 2: Accuracy vs comparison value
    ax2 = axes[1]
    valid_acc_data = [(v, acc) for v, acc in zip(valid_values, accuracies) if acc is not None]
    if valid_acc_data:
        vals_acc, acc_vals = zip(*valid_acc_data)
        ax2.plot(vals_acc, acc_vals, marker='o', linewidth=2, markersize=8, color="#9467BD")
        ax2.set_xlabel(x_label, fontsize=12)
        ax2.set_ylabel("Accuracy (Baseline Score)", fontsize=12)
        ax2.set_title(f"{algorithm.upper()} - Accuracy vs {title_suffix}", fontsize=13, fontweight="bold")
        ax2.grid(alpha=0.3, linestyle="--")
        ax2.set_xticks(valid_values)
        for val, acc in zip(vals_acc, acc_vals):
            ax2.text(val, acc, f"{acc:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No accuracy data available", 
                ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_title(f"{algorithm.upper()} - Accuracy vs {title_suffix}", fontsize=13, fontweight="bold")
    
    # Subplot 3: Latency Breakdown (stacked bar chart) - only if breakdown data exists
    if has_breakdown_data:
        ax3 = axes[2]
        # Define stage order and colors
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
            "#8C564B",  # brown
            "#E377C2",  # pink
            "#FF7F0E",  # orange
            "#BCBD22",  # olive
            "#17BECF",  # cyan
            "#1F77B4",  # blue
            "#D62728",  # red
        ]
        
        # Prepare data for stacked bar chart
        x_pos = range(len(valid_values))
        bottom = [0] * len(valid_values)
        
        for i, stage in enumerate(stage_order):
            values = [breakdown.get(stage, 0.0) if breakdown is not None else 0.0 for breakdown in breakdowns]
            ax3.bar(x_pos, values, bottom=bottom, label=stage, color=stage_colors[i], alpha=0.8)
            bottom = [b + v for b, v in zip(bottom, values)]
        
        ax3.set_xlabel(x_label, fontsize=12)
        ax3.set_ylabel("Latency (ms)", fontsize=12)
        ax3.set_title(f"{algorithm.upper()} - Latency Breakdown", fontsize=13, fontweight="bold")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(valid_values)
        ax3.legend(loc='upper left', fontsize=8, ncol=1)
        ax3.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Overall title
    if is_lgs_comparison:
        title = f"Layer Group Size Effect Analysis - {algorithm.upper()} (rank={target_rank}, RV={target_rv}, n_iter={target_niter})\nComparison across different LGS values"
    else:
        title = f"Power Iteration Effect Analysis - {algorithm.upper()} (LGS={target_lgs}, rank={target_rank}, RV={target_rv})\nComparison across different n_iter values"
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    
    # Save to plot directory
    if is_lgs_comparison:
        out_path = repo_root / "plot" / f"fig_layer_group_size_comparison_{algorithm}_RK{target_rank}_RV{target_rv}_NITER{target_niter}.png"
    else:
        out_path = repo_root / "plot" / f"fig_power_iteration_comparison_{algorithm}_LGS{target_lgs}_RK{target_rank}_RV{target_rv}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved: {out_path}")
    print("\nSummary:")
    if is_lgs_comparison:
        print(f"{'LGS':<8s} {'Latency (ms)':>15s} {'Accuracy':>12s}")
        print("-" * 40)
        for val, lat, acc in zip(valid_values, latencies, accuracies):
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            print(f"{val:<8d} {lat:>15.2f} {acc_str:>12s}")
    else:
        print(f"{'n_iter':<8s} {'Latency (ms)':>15s} {'Accuracy':>12s}")
        print("-" * 40)
        for val, lat, acc in zip(valid_values, latencies, accuracies):
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            print(f"{val:<8d} {lat:>15.2f} {acc_str:>12s}")


def list_available_configs():
    """List available configurations from existing result files."""
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    
    configs = set()
    
    # Scan all result files
    for pattern in ["**/xKV_LGS*_RK*_RV*_NITER*.json", "**/xKV_LGS*_RK*_RV*_NITER*_breakdown.log"]:
        for file_path in results_dir.glob(pattern):
            # Skip backup files
            if file_path.name.endswith(".bak"):
                continue
            
            # Extract config from filename
            # Pattern: xKV_LGS{LGS}_RK{RK}_RV{RV}_NITER{NITER}...
            match = re.search(r"LGS(\d+)_RK(\d+)_RV(\d+)_NITER(\d+)", file_path.name)
            if match:
                lgs, rk, rv, niter = map(int, match.groups())
                configs.add((lgs, rk, rv, niter))
    
    if not configs:
        print("No configurations found in results directory.")
        return
    
    print("Available configurations (LGS, Rank, RV, n_iter):")
    print("=" * 60)
    sorted_configs = sorted(configs, key=lambda x: (x[0], x[1], x[2], x[3]))
    for lgs, rk, rv, niter in sorted_configs:
        print(f"  LGS={lgs:2d}, Rank={rk:3d}, RV={rv:3d}, n_iter={niter:2d}")
    print("=" * 60)
    print(f"\nTotal: {len(configs)} unique configurations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SVD methods comparison plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available configurations
  python -m plot.fig_all_methods_comparison list
  
  # Generate all methods comparison (default: LGS=4, rank=256, RV=384, n_iter=8)
  python -m plot.fig_all_methods_comparison
  
  # Generate all methods comparison with custom parameters
  python -m plot.fig_all_methods_comparison --lgs 8 --rank 512 --rv 768 --n_iter 8
  
  # Generate power_iteration comparison for cholqr_v3
  python -m plot.fig_all_methods_comparison power_iter --algorithm cholqr_v3 --rank 256 --lgs 4 --rv 384
  
  # Generate layer_group_size comparison for cholqr_v2
  python -m plot.fig_all_methods_comparison layer_group --algorithm cholqr_v2 --rank 512 --rv 768 --n_iter 8
  
  # Generate all power_iteration comparisons
  python -m plot.fig_all_methods_comparison all_power_iter --rank 256 --lgs 4 --rv 384
        """
    )
    
    parser.add_argument("mode", nargs="?", default="all_methods",
                       choices=["all_methods", "power_iter", "layer_group", "all_power_iter", "list"],
                       help="Plot mode: all_methods, power_iter, layer_group, all_power_iter, or list")
    
    parser.add_argument("--lgs", type=int, default=4,
                       help="Layer Group Size (default: 4)")
    parser.add_argument("--rank", type=int, default=256,
                       help="Rank value (default: 256)")
    parser.add_argument("--rv", type=int, default=384,
                       help="Rank Value (default: 384)")
    parser.add_argument("--n_iter", type=int, default=8,
                       help="Number of power iterations (default: 8)")
    parser.add_argument("--algorithm", type=str, default="cholqr_v3",
                       choices=["cholqr_v1", "cholqr_v2", "cholqr_v3"],
                       help="Algorithm name (for power_iter/layer_group modes)")
    
    args = parser.parse_args()
    
    if args.mode == "list":
        list_available_configs()
    elif args.mode == "power_iter":
        plot_power_iteration_comparison(
            algorithm=args.algorithm,
            target_rank=args.rank,
            target_lgs=args.lgs,
            target_rv=args.rv,
            target_niter=args.n_iter,
            experiment_type="power_iteration"
        )
    elif args.mode == "layer_group":
        plot_power_iteration_comparison(
            algorithm=args.algorithm,
            target_rank=args.rank,
            target_lgs=args.lgs,
            target_rv=args.rv,
            target_niter=args.n_iter,
            experiment_type="layer_group_size"
        )
    elif args.mode == "all_power_iter":
        for algo in ["cholqr_v1", "cholqr_v2", "cholqr_v3"]:
            print(f"\n{'='*60}")
            print(f"Generating plot for {algo}")
            print(f"{'='*60}")
            try:
                plot_power_iteration_comparison(
                    algorithm=algo,
                    target_rank=args.rank,
                    target_lgs=args.lgs,
                    target_rv=args.rv,
                    target_niter=args.n_iter,
                    experiment_type="power_iteration"
                )
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        # Generate all methods comparison plot
        main(lgs=args.lgs, rank=args.rank, rank_value=args.rv, n_iter=args.n_iter)
