"""Export a public index of existing, hash-verified PNGs; never generate graphs."""
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "outputs/analysis_ready/060dcd3a86d5/all_valid_graphs_descriptive_only.csv"
PIN = "b5c9ce9ed55d7432f7fcb3847bdb7316bbe2ab7d4ea7041379c88a7cb02fd775"
FOLDERS = {
    "primary": "outputs/final_study/060dcd3a86d5/main/runs",
    "supplement_initial": "outputs/supplement_20261105_v1/runs",
    "supplement_completion": "outputs/supplement_completion_20261105_v1/runs",
}


def export():
    raw = CATALOG.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        raise ValueError("Saved-plot catalog hash mismatch")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if Counter(r["cohort"] for r in rows) != Counter(primary=890, supplement_initial=4, supplement_completion=2):
        raise ValueError("Cohort counts changed")
    if len({r["run_id"] for r in rows}) != 896:
        raise ValueError("Duplicate or missing plot identity")
    plots = []
    for row in rows:
        run_id, cohort = row["run_id"], row["cohort"]
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", run_id):
            raise ValueError("Unsafe plot identity")
        expected = FOLDERS[cohort] + "/" + run_id + ".png"
        if row["png_path"] != expected or row["valid_graph"] != "True" or row["failure"]:
            raise ValueError("Invalid saved plot row")
        if (row["eligible_primary_graph"] == "True") != (cohort == "primary"):
            raise ValueError("Primary/supplement mismatch")
        path = (ROOT / expected).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("PNG escaped evidence root")
        data = path.read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or hashlib.sha256(data).hexdigest() != row["png_sha256"]:
            raise ValueError("PNG integrity mismatch: " + run_id)
        plot = {k: row[k] for k in ("run_id", "cohort", "model", "method", "culture", "language")}
        plot.update(seed=int(row["seed"]), edges=int(row["edges"]), empty_graph=row["empty_graph"] == "True",
                    source_path=expected, png_url="./data/saved-plots/" + run_id + ".png", png_sha256=row["png_sha256"])
        plots.append(plot)
    dest = ROOT / "viewer/public/data/saved-plots.json"
    dest.write_bytes((json.dumps({"catalog_sha256": PIN, "plots": plots}, indent=2) + "\n").encode("utf-8"))
    print("Verified 896 original PNGs: 890 primary + 6 supplements; no paid calls.")


if __name__ == "__main__":
    export()
