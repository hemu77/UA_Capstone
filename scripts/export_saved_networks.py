"""Export final adjacency only for the separate 896-network 3D browser. Offline."""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "b5c9ce9ed55d7432f7fcb3847bdb7316bbe2ab7d4ea7041379c88a7cb02fd775"


def read_edges(raw, ids):
    nodes, edges = set(), set()
    for line in raw.decode("utf-8").splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        actor, *others = parts
        if actor in nodes or actor not in ids:
            raise ValueError("Duplicate or unknown adjacency row")
        nodes.add(actor)
        for other in others:
            edge = tuple(sorted((actor, other)))
            if other not in ids or actor == other or edge in edges:
                raise ValueError("Invalid or duplicate adjacency edge")
            edges.add(edge)
    if nodes != ids:
        raise ValueError("Adjacency must retain all 50 personas, including isolates")
    return sorted(map(list, edges))


def export():
    raw = (ROOT / "outputs/analysis_ready/060dcd3a86d5/all_valid_graphs_descriptive_only.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        raise ValueError("Source catalog changed")
    primary = json.loads((ROOT / "viewer/public/data/primary-viewer.json").read_bytes())
    plots = json.loads((ROOT / "viewer/public/data/saved-plots.json").read_bytes())["plots"]
    indexed = {p["run_id"]: p for p in plots}
    primary_runs = {r["run_id"]: r for r in primary["runs"]}
    ids = {p["id"] for p in primary["personas"]}
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(ids) != 50 or len(rows) != 896 or len(indexed) != 896:
        raise ValueError("Unexpected roster or catalog")
    graphs = []
    for row in rows:
        p = indexed[row["run_id"]]
        source = p["source_path"].removesuffix(".png") + ".adj"
        if row["adj_path"] != source or row["roster_sha256"] != primary["roster_id"]:
            raise ValueError("Source or roster mismatch")
        path = (ROOT / source).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Source escaped repository")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["adj_sha256"]:
            raise ValueError("Adjacency hash mismatch")
        edges = read_edges(raw, ids)
        if len(edges) != p["edges"] or row["cohort"] != p["cohort"]:
            raise ValueError("Count or cohort mismatch")
        if p["cohort"] == "primary" and sorted(primary_runs[p["run_id"]]["edges"]) != edges:
            raise ValueError("Primary replay/final graph mismatch")
        graphs.append(dict(run_id=row["run_id"], edges=edges, source=source, source_sha256=row["adj_sha256"]))
    if len({g["run_id"] for g in graphs}) != 896:
        raise ValueError("Duplicate graph")
    value = dict(schema_version=1, catalog_sha256=PIN, roster_id=primary["roster_id"], graphs=graphs)
    dest = ROOT / "viewer/public/data/saved-networks.json"
    dest.write_bytes((json.dumps(value, separators=(",", ":")) + "\n").encode())
    print("896 final adjacencies verified; 890 primary + 6 separate supplements; no paid calls.")
    print("SHA256=" + hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__ == "__main__":
    export()
