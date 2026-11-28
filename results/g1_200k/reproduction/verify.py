"""Run from the repository root. Checks data and folds without fitting."""
import hashlib
import json
from pathlib import Path

from src.phase_b_classical import load_experiment
from src.run_classical import checkpoint_features
from src.splits import TIME, get_folds

base = Path("data/backup_200k")
for line in (base / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    with (base / name).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {name}")
g0, s0, m0 = load_experiment(base / "manifest_G0.json", "G0", True, None)
g1, s1, m1 = load_experiment(base / "manifest_G1.json", "G1", True, None)
assert len(m1) == 200_000 and m1.equals(m0) and m1.index.is_unique
assert m1[TIME].dt.month.value_counts().sort_index().tolist() == json.loads(
    (base / "sampling.json").read_text())["sampled_by_month"]
for a,b in zip(g0.checkpoints, g1.checkpoints, strict=True):
    assert a.minutes == b.minutes
    x = checkpoint_features(a, s0, m0.index)
    y = checkpoint_features(b, s1, m1.index)
    assert x.equals(y.loc[:, s0.columns])
for train,valid in get_folds(m1):
    assert m1.LATE.iloc[train].nunique() == m1.LATE.iloc[valid].nunique() == 2
    print(f"{m1[TIME].iloc[valid[0]]:%B}: {len(train):,} training, {len(valid):,} validation")
