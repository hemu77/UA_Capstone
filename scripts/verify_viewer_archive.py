"""Check archive relocation and every frozen source pin without running old code."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "1275618430d87ef26e2331b1f7cedbca4a1e2027"
PATHS = ["viewer/index.html", "viewer/main.js", "viewer/style.css"] + [
    "viewer/public/data/" + name for name in (
        "networks.json", "calibration.json", "revised-calibration.json",
        "presentation.json", "artifacts", "calibration", "revised-calibration", "presentation"
    )
]


def verify():
    tree = subprocess.check_output(
        ["rtk", "proxy", "git", "ls-tree", "-r", "-z", BASELINE, "--", *PATHS], cwd=ROOT
    )
    expected = set()
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, old = entry.decode().split("\t", 1)
        _, kind, oid = metadata.split()
        if kind != "blob" or not old.startswith("viewer/"):
            raise ValueError("Unexpected archived Git entry")
        new = "viewer/archive/" + old.removeprefix("viewer/")
        raw = (ROOT / new).read_bytes()
        if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != oid:
            raise ValueError("Archived bytes changed: " + new)
        if (ROOT / old).exists():
            raise ValueError("Legacy export still in current viewer: " + old)
        expected.add(new)
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "viewer/archive").rglob("*") if p.is_file()}
    if not expected or actual != expected | {"viewer/archive/README.md"}:
        raise ValueError("Unexpected archive inventory")
    sources = json.loads((ROOT / "final_study/contract.json").read_bytes())["sources"]
    for name, digest in sources.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen source changed: " + name)
    print(f"PASS: {len(expected)} archived files byte-identical; {len(sources)} frozen source pins unchanged.")


if __name__ == "__main__":
    verify()
