import json
from collections import defaultdict
from pathlib import Path


def regroup_summary_by_rank(path: Path):
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    if not summary:
        return

    new_summary = {}

    for method_name, info in summary.items():
        records = info.get("records", [])
        if not records:
            # keep as-is if no records
            new_summary[method_name] = info
            continue

        # group records by rank
        by_rank = defaultdict(list)
        for r in records:
            rank = r.get("rank")
            by_rank[rank].append(r)

        # build per-rank entries
        for rank, recs in by_rank.items():
            durations = [r["duration_ms"] for r in recs]
            key = f"{method_name} (rank={rank})"
            new_summary[key] = {
                "count": len(recs),
                "total_time_ms": float(sum(durations)),
                "avg_time_ms": float(sum(durations) / len(durations)),
                "min_time_ms": float(min(durations)),
                "max_time_ms": float(max(durations)),
                "records": recs,
            }

    data["summary"] = new_summary
    path.write_text(json.dumps(data, indent=2))


def main():
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root
        / "results"
        / "full_svd"
        / "xKV_LGS4_RK256_RV384_FULL_SVD_svd_benchmark.json",
        repo_root
        / "results"
        / "full_svd"
        / "full_svd_LGS8_RK512_RV768_svd_benchmark.json",
        repo_root
        / "results"
        / "lowrank_svd"
        / "xKV_LGS4_RK256_RV384_LOWRANK_NITER4_svd_benchmark.json",
        repo_root
        / "results"
        / "lowrank_svd"
        / "xKV_LGS4_RK256_RV384_LOWRANK_NITER8_svd_benchmark.json",
    ]

    # add all cholqr_v1 jsons
    cholqr_v1_dir = repo_root / "results" / "cholqr_v1"
    for p in cholqr_v1_dir.rglob("*_svd_benchmark.json"):
        targets.append(p)

    for p in targets:
        if p.exists():
            print(f"Rewriting summary by rank for: {p}")
            regroup_summary_by_rank(p)
        else:
            print(f"Skip missing: {p}")


if __name__ == "__main__":
    main()

