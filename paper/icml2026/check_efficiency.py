"""Replay efficiency summaries from a local study release. No network or model calls."""
import csv
import hashlib
import sys
from pathlib import Path
from statistics import mean

root = Path(sys.argv[1])
path = root / "outputs/publication_figures/20261111_v4/efficiency_records.csv"
with path.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
groups = [[r for r in rows if r["source"] == k] for k in ("snap_facebook", "sociopatterns_friendship")]
assert [len(g) for g in groups] == [10, 1] and len({r["cohort"] for r in rows}) == len(rows) == 11
keys = ("global", "ER", "WS", "target_ER")
def means(group):
    return [mean(float(r[k]) for r in group) for k in keys]
g, er, ws, target = means(rows)
print(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
print("Global, fixed ER, fixed WS, target ER")
for name, values in zip(("Facebook", "School", "Equal-record", "Equal-source"),
                        (means(groups[0]), means(groups[1]), means(rows), list(map(mean, zip(*map(means, groups)))))):
    print(name, *(f"{v:.6f}" for v in values))
print("Gap closed relative to Global: fixed ER, fixed WS", *((b-target)/(b-g) for b in (er, ws)))
print("Target ER closer than fixed ER:", sum(float(r["target_ER"]) < float(r["ER"]) for r in rows), "/ 11")

# Rank association between density distance and efficiency distance over the
# 11 original records x 7 transfer candidates at 50 nodes (77 pairs).
def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out
def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5
summary = root / "outputs/research_extension/20261114_v2/transfer_record_summary.csv"
with summary.open(encoding="utf-8", newline="") as stream:
    table = [r for r in csv.DictReader(stream) if r["n"] == "50" and r["cohort"] != "Copenhagen"]
cell = {(r["cohort"], r["procedure"], r["metric"]): float(r["mean_w1"]) for r in table}
pairs = sorted({k[:2] for k in cell})
assert len(pairs) == 77
density = [cell[(*p, "density")] for p in pairs]
for metric in ("global_efficiency", "prop_nodes_lcc", "avg_clustering_coef"):
    print("Spearman, density W1 vs", metric, "W1:", round(spearman(density, [cell[(*p, metric)] for p in pairs]), 3))
