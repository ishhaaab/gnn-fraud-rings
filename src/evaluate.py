"""PR-AUC, recall@1%FPR, precision@100/500, ablation, ring-size slices."""
import json
from pathlib import Path


def pr_metrics(y_true, scores) -> dict:
    """PR-AUC, recall at 1% FPR, p@100/p@500. Phase 3."""
    raise NotImplementedError("Phase 3")


def run_comparison(out: Path = Path("results/metrics.json")) -> dict:
    """Baseline vs DeepWalk vs GraphSAGE + ablation + slices. Phase 3."""
    raise NotImplementedError("Phase 3")


if __name__ == "__main__":
    print(json.dumps({"status": "skeleton — implement Phase 3"}))
