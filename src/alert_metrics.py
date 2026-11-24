"""Rank by risk with stable flight-ID tie breaks and explicit alert costs."""
import numpy as np
import pandas as pd


def alert_metrics(labels, scores):
    ranked = pd.DataFrame({"late": labels.to_numpy(), "score": scores, "id": labels.index})
    ranked = ranked.sort_values(["score", "id"], ascending=[False, True])
    n = len(ranked)
    late = int(ranked.late.sum())
    result = {}
    for percent in (5, 10, 20):
        k = int(np.ceil(n * percent / 100))
        caught = int(ranked.late.iloc[:k].sum())
        result.update({
            f"precision_at_{percent}": caught / k,
            f"lift_at_{percent}": (caught / k) / (late / n) if late else np.nan,
            f"cost_at_{percent}": (k + 5 * (late - caught)) / n,
        })
    return result
