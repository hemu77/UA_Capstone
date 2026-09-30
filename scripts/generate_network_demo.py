from __future__ import annotations

import json
import math
import re
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TEXT = ROOT / "text-files"
OUT = ROOT / "outputs" / "demo"

W, H = 1920, 1080
FPS = 30
SECONDS = 18
FRAMES = FPS * SECONDS

PAPER = "#FBFAF6"
INK = "#172033"
MUTED = "#5F6D7A"
LINE = "#D7E0E8"
BLUE = "#1D79D3"
RED = "#E84B5F"
GOLD = "#E5A11A"
GREEN = "#16855D"
CYAN = "#14B8D4"
CARD = "#FFFFFF"
WHITE = "#FFFFFF"

GRAPH_FILES = {
    "global": TEXT / "global_gpt-4.1-mini_culture_us_0.adj",
    "local": TEXT / "local_gpt-4.1-mini_n5_culture_us_0.adj",
    "sequential": TEXT / "sequential_gpt-4.1-mini_n5_culture_us_0.adj",
    "iterative": TEXT / "iterative_gpt-4.1-mini_n5_culture_us_0.adj",
}

METHOD_ACCENT = {
    "global": BLUE,
    "local": RED,
    "sequential": GOLD,
    "iterative": GREEN,
}

METHOD_TITLES = {
    "global": "GLOBAL METHOD",
    "local": "LOCAL METHOD",
    "sequential": "SEQUENTIAL METHOD",
    "iterative": "ITERATIVE METHOD",
}

METHOD_STEPS = {
    "global": ("50-person roster", "one full-roster prompt", "friendship pairs appear"),
    "local": ("50-person roster", "one persona chooses", "choices become ties"),
    "sequential": ("50-person roster", "current graph is visible", "next choices update graph"),
    "iterative": ("start from local graph", "add/drop revisions", "revised network settles"),
}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    font_dir = Path("C:/Windows/Fonts")
    for candidate in [
        font_dir / name,
        font_dir / f"{name}.ttf",
        font_dir / f"{name}.otf",
        font_dir / "bahnschrift.ttf",
        font_dir / "segoeuib.ttf",
        font_dir / "segoeui.ttf",
        font_dir / "arial.ttf",
    ]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


F = {
    "display": font("bahnschrift", 60),
    "title": font("bahnschrift", 44),
    "section": font("bahnschrift", 30),
    "label": font("bahnschrift", 22),
    "body": font("segoeui", 20),
    "body_bold": font("segoeuib", 20),
    "small": font("segoeui", 16),
    "tiny": font("segoeui", 13),
}


def rgb(hex_value: str) -> tuple[int, int, int]:
    value = hex_value.strip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def blend(a: str, b: str, t: float) -> str:
    ca, cb = rgb(a), rgb(b)
    vals = [int(ca[i] + (cb[i] - ca[i]) * t) for i in range(3)]
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def text_size(draw: ImageDraw.ImageDraw, value: str, ft: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), value, font=ft)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, ft: ImageFont.FreeTypeFont, fill: str) -> None:
    w, h = text_size(draw, value, ft)
    draw.text((xy[0] - w / 2, xy[1] - h / 2), value, font=ft, fill=fill)


def wrap(draw: ImageDraw.ImageDraw, value: str, ft: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in value.split():
        candidate = word if not line else f"{line} {word}"
        if text_size(draw, candidate, ft)[0] <= width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, ft: ImageFont.FreeTypeFont, fill: str, width: int, gap: int = 5) -> int:
    x, y = xy
    for line in wrap(draw, value, ft, width):
        draw.text((x, y), line, font=ft, fill=fill)
        y += text_size(draw, line, ft)[1] + gap
    return y


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill: str, outline: str | None = None, width: int = 2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def parse_adj(path: Path) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(50))
    row = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for target in [int(x) for x in re.findall(r"\d+", line)]:
            if 0 <= row < 50 and 0 <= target < 50 and row != target:
                graph.add_edge(min(row, target), max(row, target))
        row += 1
    return graph


def load_personas() -> dict[int, dict[str, object]]:
    raw = json.loads((TEXT / "us_50_gpt4o_w_interests.json").read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def node_color(person: dict[str, object]) -> str:
    return BLUE if person["political affiliation"] == "Democrat" else RED


def build_positions(graphs: dict[str, nx.Graph]) -> dict[int, tuple[float, float]]:
    union = nx.Graph()
    union.add_nodes_from(range(50))
    for graph in graphs.values():
        union.add_edges_from(graph.edges())
    raw = nx.spring_layout(union, seed=32, k=0.55, iterations=500)
    xs = [raw[n][0] for n in raw]
    ys = [raw[n][1] for n in raw]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    out = {}
    for n, pt in raw.items():
        x = (pt[0] - min_x) / max(1e-9, max_x - min_x)
        y = (pt[1] - min_y) / max(1e-9, max_y - min_y)
        out[n] = (660 + x * 1040, 260 + y * 610)
    return out


def ordered_edges(graph: nx.Graph, method: str) -> list[tuple[int, int]]:
    edges = sorted((min(a, b), max(a, b)) for a, b in graph.edges())
    if method == "global":
        cx, cy = 960, 540
        return sorted(edges, key=lambda e: min(math.dist((cx, cy), (positions_global[e[0]][0], positions_global[e[0]][1])), math.dist((cx, cy), (positions_global[e[1]][0], positions_global[e[1]][1]))))
    if method in {"local", "sequential"}:
        return sorted(edges, key=lambda e: (max(e), min(e)))
    return sorted(edges, key=lambda e: (e[0] + e[1], e[0], e[1]))


def make_background() -> Image.Image:
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        draw.line((0, y, W, y), fill=blend("#FFFFFF", PAPER, t))
    return img


def draw_roster_grid(draw: ImageDraw.ImageDraw, personas: dict[int, dict[str, object]], reveal: float, highlight: int | None = None) -> None:
    x0, y0 = 96, 382
    cell = 34
    for node in range(50):
        col = node % 10
        row = node // 10
        cx = x0 + col * cell
        cy = y0 + row * cell
        if node / 50 > reveal:
            fill = "#E9EEF4"
            outline = "#E9EEF4"
        else:
            fill = node_color(personas[node])
            outline = "#142033" if node == highlight else WHITE
        r = 11 if node != highlight else 15
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=2)


def draw_nodes(draw: ImageDraw.ImageDraw, positions: dict[int, tuple[float, float]], personas: dict[int, dict[str, object]], highlight: int | None = None) -> None:
    for node, (x, y) in positions.items():
        r = 12 if node == highlight else 8
        draw.ellipse((x - r, y - r, x + r, y + r), fill=node_color(personas[node]), outline="#142033" if node == highlight else WHITE, width=2)


def draw_edges(img: Image.Image, graph: nx.Graph, edges: list[tuple[int, int]], positions: dict[int, tuple[float, float]], personas: dict[int, dict[str, object]], reveal: float, accent: str, ghost: bool = False) -> Image.Image:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    n = int(len(edges) * max(0.0, min(1.0, reveal)))
    for a, b in edges[:n]:
        ax, ay = positions[a]
        bx, by = positions[b]
        if ghost:
            col = (*rgb("#AAB7C4"), 85)
            width = 2
        else:
            same = personas[a]["political affiliation"] == personas[b]["political affiliation"]
            col = (*rgb(accent if same else "#687386"), 175)
            width = 2
        draw.line((ax, ay, bx, by), fill=col, width=width)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def draw_removed_edges(img: Image.Image, edges: list[tuple[int, int]], positions: dict[int, tuple[float, float]], reveal: float) -> Image.Image:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    n = int(len(edges) * max(0.0, min(1.0, reveal)))
    for a, b in edges[:n]:
        ax, ay = positions[a]
        bx, by = positions[b]
        draw.line((ax, ay, bx, by), fill=(*rgb("#AAB7C4"), 110), width=2)
        mx, my = (ax + bx) / 2, (ay + by) / 2
        draw.line((mx - 7, my - 7, mx + 7, my + 7), fill=(*rgb(RED), 210), width=3)
        draw.line((mx - 7, my + 7, mx + 7, my - 7), fill=(*rgb(RED), 210), width=3)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def draw_header(draw: ImageDraw.ImageDraw, method: str, t: float) -> None:
    accent = METHOD_ACCENT[method]
    draw.text((68, 48), METHOD_TITLES[method], font=F["display"], fill=accent)
    draw.text((72, 120), "network generation from the fixed 50-person roster", font=F["body"], fill=MUTED)
    steps = METHOD_STEPS[method]
    x = 70
    active = 0 if t < 0.28 else 1 if t < 0.58 else 2
    for idx, step in enumerate(steps):
        fill = accent if idx == active else "#EEF3F7"
        ink = WHITE if idx == active else MUTED
        rounded(draw, (x, 166, x + 300, 214), 24, fill, outline=accent if idx == active else LINE, width=2)
        draw.text((x + 22, 179), f"{idx + 1}. {step}", font=F["small"], fill=ink)
        x += 324


def draw_persona_source(draw: ImageDraw.ImageDraw, personas: dict[int, dict[str, object]], t: float, highlight: int | None) -> None:
    rounded(draw, (58, 250, 486, 604), 28, CARD, outline=LINE, width=2)
    draw.text((86, 280), "persona data", font=F["section"], fill=INK)
    draw.text((88, 324), "50 fixed people", font=F["body_bold"], fill=MUTED)
    draw_roster_grid(draw, personas, min(1.0, t * 2.2), highlight=highlight)
    rounded(draw, (78, 526, 456, 574), 20, "#F6F9FC", outline=LINE, width=1)
    draw.text((100, 542), "gender | age | race | religion | politics | interests", font=F["small"], fill=MUTED)


def draw_method_explanation(draw: ImageDraw.ImageDraw, method: str, t: float, highlight: int | None) -> None:
    accent = METHOD_ACCENT[method]
    rounded(draw, (58, 640, 486, 916), 28, CARD, outline=LINE, width=2)
    if method == "global":
        title = "full roster prompt"
        body = "The complete list of people is sent at once. The response is a set of friendship pairs."
    elif method == "local":
        title = "one-person choice"
        body = "A selected persona chooses friends independently from the roster."
    elif method == "sequential":
        title = "current graph shown"
        body = "A selected persona chooses while seeing the graph that already exists."
    else:
        title = "revise the graph"
        body = "The network starts from local choices, then repeated add/drop decisions reshape it."
    draw.text((86, 670), title, font=F["section"], fill=accent)
    draw_wrapped(draw, (88, 718), body, F["body"], INK, 350, gap=7)
    if highlight is not None and method in {"local", "sequential"}:
        rounded(draw, (88, 828, 452, 880), 18, "#F6F9FC", outline=accent, width=2)
        draw.text((110, 843), f"active persona: ID {highlight:02d}", font=F["body_bold"], fill=accent)
    if method == "global":
        rounded(draw, (106, 828, 434, 880), 18, "#F6F9FC", outline=accent, width=2)
        draw.text((132, 843), "all personas considered together", font=F["body_bold"], fill=accent)
    if method == "iterative":
        draw.text((106, 828), "+ add", font=F["body_bold"], fill=GREEN)
        draw.text((214, 828), "- drop", font=F["body_bold"], fill=RED)
        draw.text((326, 828), "settle", font=F["body_bold"], fill=accent)


def draw_network_frame(method: str, t: float, data: dict[str, object]) -> Image.Image:
    img = make_background()
    draw = ImageDraw.Draw(img)
    personas = data["personas"]
    positions = data["positions"]
    graphs = data["graphs"]
    graph = graphs[method]
    accent = METHOD_ACCENT[method]
    order = data["orders"][method]
    active_node = min(49, int(t * 62)) if method in {"local", "sequential"} else None

    draw_header(draw, method, t)
    draw_persona_source(draw, personas, t, active_node)
    draw_method_explanation(draw, method, t, active_node)

    rounded(draw, (540, 236, 1846, 946), 34, CARD, outline=LINE, width=2)
    draw.text((586, 266), "generated network", font=F["section"], fill=INK)

    if method == "global":
        reveal = ease((t - 0.34) / 0.48)
        img = draw_edges(img, graph, order, positions, personas, reveal, accent)
    elif method == "local":
        reveal = ease((t - 0.28) / 0.58)
        img = draw_edges(img, graph, order, positions, personas, reveal, accent)
        active_node = max(0, min(49, int(reveal * 50)))
    elif method == "sequential":
        reveal = ease((t - 0.28) / 0.58)
        prior_reveal = max(0.0, reveal - 0.12)
        img = draw_edges(img, graph, order, positions, personas, prior_reveal, "#AAB7C4", ghost=True)
        img = draw_edges(img, graph, order, positions, personas, reveal, accent)
        active_node = max(0, min(49, int(reveal * 50)))
    else:
        local_graph = graphs["local"]
        local_edges = data["orders"]["local"]
        final_edges = set(graph.edges())
        initial_edges = set(local_graph.edges())
        additions = sorted(final_edges - initial_edges)
        removals = sorted(initial_edges - final_edges)
        if t < 0.36:
            img = draw_edges(img, local_graph, local_edges, positions, personas, ease(t / 0.36), "#AAB7C4", ghost=True)
        elif t < 0.68:
            img = draw_edges(img, local_graph, local_edges, positions, personas, 1.0, "#AAB7C4", ghost=True)
            img = draw_edges(img, graph, additions, positions, personas, ease((t - 0.36) / 0.32), GREEN)
            img = draw_removed_edges(img, removals, positions, ease((t - 0.46) / 0.22))
        else:
            img = draw_edges(img, graph, order, positions, personas, ease((t - 0.68) / 0.22), accent)

    draw = ImageDraw.Draw(img)
    draw_nodes(draw, positions, personas, highlight=active_node)

    stat = data["stats"][method]
    rounded(draw, (582, 856, 850, 914), 20, "#F6F9FC", outline=LINE, width=1)
    draw.text((606, 873), f"{stat['edges']} ties generated", font=F["body_bold"], fill=accent)
    rounded(draw, (872, 856, 1188, 914), 20, "#F6F9FC", outline=LINE, width=1)
    draw.text((896, 873), "same 50 personas throughout", font=F["body_bold"], fill=INK)

    progress = int(1770 * t)
    draw.rounded_rectangle((74, 1008, 1846, 1020), radius=6, fill="#E8EEF4")
    draw.rounded_rectangle((74, 1008, 74 + progress, 1020), radius=6, fill=accent)
    return img


def graph_stats(graph: nx.Graph) -> dict[str, object]:
    return {"edges": graph.number_of_edges(), "density": nx.density(graph)}


def write_video(method: str, data: dict[str, object]) -> Path:
    path = OUT / f"{method}_network_generation.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}")
    for frame in range(FRAMES):
        t = frame / max(1, FRAMES - 1)
        img = draw_network_frame(method, t, data)
        writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def write_preview(method: str, data: dict[str, object]) -> Path:
    path = OUT / f"{method}_network_generation_preview.png"
    img = draw_network_frame(method, 0.88, data)
    img.save(path, optimize=True)
    return path


def write_html(data: dict[str, object]) -> Path:
    payload = {
        "stats": data["stats"],
        "files": {method: f"{method}_network_generation.mp4" for method in GRAPH_FILES},
    }
    path = OUT / "network_generation_methods.html"
    cards = "\n".join(
        f"""
        <section>
          <h2>{METHOD_TITLES[m].title()}</h2>
          <video src="{m}_network_generation.mp4" controls muted loop playsinline></video>
        </section>
        """
        for m in GRAPH_FILES
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Network Generation Methods</title>
<style>
body {{ margin: 0; background: #fbfaf6; color: #172033; font-family: Bahnschrift, Segoe UI, Arial, sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 30px; }}
h1 {{ font-size: 44px; margin: 0 0 8px; }}
p {{ color: #5f6d7a; font-size: 18px; }}
section {{ background: white; border: 1px solid #d7e0e8; border-radius: 24px; padding: 18px; margin: 22px 0; box-shadow: 0 16px 36px rgba(23,32,51,.08); }}
h2 {{ margin: 0 0 14px; }}
video {{ width: 100%; border-radius: 18px; background: #fff; }}
</style>
</head>
<body>
<main>
<h1>Network Generation Methods</h1>
<p>Separate clean animations for the four generation processes. Each video uses the same fixed 50-person roster and the corresponding generated network file.</p>
{cards}
<script type="application/json" id="summary">{json.dumps(payload)}</script>
</main>
</body>
</html>""",
        encoding="utf-8",
    )
    return path


def build_data() -> dict[str, object]:
    personas = load_personas()
    graphs = {method: parse_adj(path) for method, path in GRAPH_FILES.items()}
    positions = build_positions(graphs)
    global positions_global
    positions_global = positions
    orders = {method: ordered_edges(graph, method) for method, graph in graphs.items()}
    return {
        "personas": personas,
        "graphs": graphs,
        "positions": positions,
        "orders": orders,
        "stats": {method: graph_stats(graph) for method, graph in graphs.items()},
    }


def write_summary(data: dict[str, object]) -> Path:
    path = OUT / "network_generation_methods_summary.json"
    payload = {
        "source_personas": str(TEXT / "us_50_gpt4o_w_interests.json"),
        "source_graphs": {method: str(path) for method, path in GRAPH_FILES.items()},
        "stats": data["stats"],
        "outputs": {
            method: {
                "video": str(OUT / f"{method}_network_generation.mp4"),
                "preview": str(OUT / f"{method}_network_generation_preview.png"),
            }
            for method in GRAPH_FILES
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_data()
    outputs: list[Path] = []
    for method in GRAPH_FILES:
        print(f"rendering {method}")
        outputs.append(write_video(method, data))
        outputs.append(write_preview(method, data))
    outputs.append(write_html(data))
    outputs.append(write_summary(data))
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    positions_global: dict[int, tuple[float, float]] = {}
    main()
