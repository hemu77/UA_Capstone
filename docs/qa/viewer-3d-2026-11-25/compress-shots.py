"""Re-encode the final screenshots as 256-colour (octree) palette PNGs so the QA folder stays small (dark UI, few hues).
Usage: python3 docs/qa/viewer-3d-2026-11-25/compress-shots.py [folder]   (default: final/ next to this file)"""
import pathlib, sys
from PIL import Image

folder = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent / "final"
total = 0
for path in sorted(folder.glob("*.png")):
    image = Image.open(path)
    before = path.stat().st_size
    if image.mode != "P":
        image.convert("RGB").quantize(colors=256, method=Image.Quantize.FASTOCTREE).save(path, optimize=True)
    total += path.stat().st_size
    print(f"{path.name:34s} {image.size[0]}x{image.size[1]}  {before/1024:8.0f} KB -> {path.stat().st_size/1024:6.0f} KB")
print(f"total {total/1048576:.2f} MB")
