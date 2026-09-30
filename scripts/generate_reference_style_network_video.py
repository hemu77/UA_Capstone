from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TEXT = ROOT / "text-files"
OUT = ROOT / "outputs" / "network_bonding_demo"

PERSONA_FILE = TEXT / "us_50_gpt4o_w_interests.json"
NETWORK_FILE = TEXT / "global_gpt-4.1-mini_culture_india_0.adj"

W, H = 3840, 2160
FPS = 24
DURATION = 50

TITLE = "Cross-Cultural and Linguistic Agentic Graph Simulation Framework"
SUBTITLE = "50 personas forming one generated friendship network"

WHITE = (255, 255, 255)
BLACK = (24, 24, 24)
BLUE = (30, 136, 197)
ORANGE = (245, 145, 42)
GRAY = (90, 90, 90)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT_SIZES = [28, 32, 34, 36, 42, 44, 48, 56, 64, 72, 84, 96, 118, 132]
F = {s: font(s) for s in FONT_SIZES}
FB = {s: font(s, True) for s in FONT_SIZES}


def load_personas() -> dict[int, dict]:
    raw = json.loads(PERSONA_FILE.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def load_edges() -> list[tuple[int, int]]:
    rows: list[list[int]] = []
    for line in NETWORK_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rows.append([int(x) for x in line.split()])

    edges = set()
    for src, friends in enumerate(rows):
        for dst in friends:
            if src != dst:
                edges.add(tuple(sorted((src, dst))))
    return sorted(edges)


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def build_groups(graph: nx.Graph) -> list[list[int]]:
    groups = [sorted(c) for c in nx.algorithms.community.greedy_modularity_communities(graph)]
    groups.sort(key=lambda g: (-len(g), min(g)))
    return groups


def group_centers(groups: list[list[int]]) -> list[tuple[int, int]]:
    # Spread communities like the uploaded reference plot: separated islands linked by bridge lines.
    centers = [
        (720, 1060),
        (1420, 720),
        (1420, 1500),
        (2320, 850),
        (2820, 1430),
        (3120, 620),
    ]
    return centers[: len(groups)]


def build_positions(graph: nx.Graph, groups: list[list[int]]) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    positions: dict[int, tuple[int, int]] = {}
    group_of: dict[int, int] = {}
    centers = group_centers(groups)

    for gi, nodes in enumerate(groups):
        sub = graph.subgraph(nodes)
        if len(nodes) == 1:
            raw = {nodes[0]: np.array([0.0, 0.0])}
        else:
            raw = nx.spring_layout(sub, seed=900 + gi, iterations=500, k=0.72)

        vals = np.array(list(raw.values()))
        max_abs = max(float(np.abs(vals).max()), 1e-6)
        cx, cy = centers[gi]
        scale_x = 280 + 18 * len(nodes)
        scale_y = 185 + 10 * len(nodes)

        for n in nodes:
            p = raw[n] / max_abs
            positions[n] = (int(cx + p[0] * scale_x), int(cy + p[1] * scale_y))
            group_of[n] = gi

    return positions, group_of


def initial_positions() -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for n in range(50):
        col = n % 10
        row = n // 10
        out[n] = (900 + col * 215, 650 + row * 150)
    return out


def positions_at(t: float, final_positions: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
    start = initial_positions()
    move = ease((t - 8.0) / 8.0)
    out = {}
    for n in range(50):
        sx, sy = start[n]
        fx, fy = final_positions[n]
        out[n] = (int(sx + (fx - sx) * move), int(sy + (fy - sy) * move))
    return out


def order_edges(edges: list[tuple[int, int]], group_of: dict[int, int]) -> list[tuple[int, int]]:
    intra = [e for e in edges if group_of[e[0]] == group_of[e[1]]]
    bridges = [e for e in edges if group_of[e[0]] != group_of[e[1]]]
    intra.sort(key=lambda e: (group_of[e[0]], e[0], e[1]))
    bridges.sort(key=lambda e: (min(group_of[e[0]], group_of[e[1]]), max(group_of[e[0]], group_of[e[1]]), e[0], e[1]))
    return intra + bridges


def draw_centered_text(draw: ImageDraw.ImageDraw, text: str, y: int, fnt: ImageFont.FreeTypeFont, fill=BLACK) -> None:
    draw.text((W // 2, y), text, font=fnt, fill=fill, anchor="mm")


def draw_title_frame(draw: ImageDraw.ImageDraw, t: float) -> None:
    alpha = int(255 * (1 - ease((t - 5.2) / 1.5)))
    if alpha <= 0:
        return
    draw_centered_text(draw, TITLE, 810, FB[96], (BLACK[0], BLACK[1], BLACK[2], alpha))
    draw_centered_text(draw, SUBTITLE, 930, F[56], (GRAY[0], GRAY[1], GRAY[2], alpha))
    draw.line((900, 1045, 2940, 1045), fill=(BLUE[0], BLUE[1], BLUE[2], alpha), width=8)
    draw_centered_text(draw, "personas  ->  groups  ->  friendship bonds", 1145, FB[56], (BLUE[0], BLUE[1], BLUE[2], alpha))


def draw_header(draw: ImageDraw.ImageDraw, t: float) -> None:
    alpha = int(255 * ease((t - 5.5) / 1.5))
    if alpha <= 0:
        return
    draw_centered_text(draw, TITLE, 92, FB[64], (BLACK[0], BLACK[1], BLACK[2], alpha))
    draw_centered_text(draw, "Generated network sample: 50 personas, 105 friendship bonds", 164, F[42], (GRAY[0], GRAY[1], GRAY[2], alpha))


def draw_node(draw: ImageDraw.ImageDraw, n: int, xy: tuple[int, int], alpha: int = 255, highlight: bool = False) -> None:
    x, y = xy
    r = 42 if not highlight else 52
    if highlight:
        draw.ellipse((x - 72, y - 72, x + 72, y + 72), outline=(ORANGE[0], ORANGE[1], ORANGE[2], alpha), width=9)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(BLUE[0], BLUE[1], BLUE[2], alpha), outline=(BLUE[0], BLUE[1], BLUE[2], alpha), width=3)
    draw.text((x, y), str(n), font=FB[36], fill=(8, 20, 30, alpha), anchor="mm")


def draw_grid_card(draw: ImageDraw.ImageDraw, n: int, xy: tuple[int, int], alpha: int) -> None:
    x, y = xy
    w, h = 132, 86
    draw.rounded_rectangle((x - w // 2, y - h // 2, x + w // 2, y + h // 2), radius=16, fill=(BLUE[0], BLUE[1], BLUE[2], int(215 * alpha / 255)), outline=(BLUE[0], BLUE[1], BLUE[2], alpha), width=3)
    draw.text((x, y), str(n), font=FB[36], fill=(10, 24, 34, alpha), anchor="mm")


def draw_frame(
    frame: int,
    edges_ordered: list[tuple[int, int]],
    groups: list[list[int]],
    final_positions: dict[int, tuple[int, int]],
    group_of: dict[int, int],
) -> Image.Image:
    t = frame / FPS
    img = Image.new("RGBA", (W, H), (*WHITE, 255))
    draw = ImageDraw.Draw(img)

    draw_title_frame(draw, t)
    draw_header(draw, t)

    graph_alpha = int(255 * ease((t - 5.8) / 2.0))
    positions = positions_at(t, final_positions)
    grid_stage = t < 16.0

    # Edges form slowly, exactly like a network plot being drawn.
    edge_start = 17.0
    edge_end = 44.0
    edge_progress = max(0.0, min(1.0, (t - edge_start) / (edge_end - edge_start)))
    visible = int(len(edges_ordered) * edge_progress)

    if t >= edge_start:
        for a, b in edges_ordered[:visible]:
            x1, y1 = final_positions[a]
            x2, y2 = final_positions[b]
            color = BLACK if group_of[a] == group_of[b] else (50, 50, 50)
            draw.line((x1, y1, x2, y2), fill=color, width=5)

        if visible < len(edges_ordered):
            a, b = edges_ordered[visible]
            x1, y1 = final_positions[a]
            x2, y2 = final_positions[b]
            pulse = 0.5 + 0.5 * math.sin(frame * 0.6)
            draw.line((x1, y1, x2, y2), fill=ORANGE, width=int(9 + 4 * pulse))

    # Draw nodes over edges.
    for n in range(50):
        appear = int(255 * ease((t - 6.2 - n * 0.025) / 0.9))
        if appear <= 0:
            continue
        if grid_stage:
            draw_grid_card(draw, n, positions[n], min(appear, graph_alpha))
        else:
            highlight = False
            if t >= edge_start and visible < len(edges_ordered):
                highlight = n in edges_ordered[visible]
            draw_node(draw, n, final_positions[n], min(appear, graph_alpha), highlight)

    # Minimal counter, not clutter.
    if t >= edge_start and t < 45.0:
        count_text = f"{min(visible, len(edges_ordered))} / {len(edges_ordered)} friendships"
        draw.rounded_rectangle((130, 1900, 820, 2025), radius=28, fill=(255, 255, 255, 235), outline=(180, 180, 180), width=3)
        draw.text((180, 1930), count_text, font=FB[48], fill=BLACK)

    return img.convert("RGB")


def make_contact_sheet(video: Path, out: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    times = [3, 9, 15, 22, 34, 48]
    thumbs = []
    for sec in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, frame = cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            thumbs.append((sec, Image.fromarray(rgb).resize((640, 360), Image.Resampling.LANCZOS)))
    cap.release()

    sheet = Image.new("RGB", (640 * len(thumbs), 430), WHITE)
    draw = ImageDraw.Draw(sheet)
    for i, (sec, thumb) in enumerate(thumbs):
        x = i * 640
        sheet.paste(thumb, (x, 0))
        draw.text((x + 24, 380), f"{sec}s", font=FB[36], fill=BLACK)
    sheet.save(out, quality=92)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    personas = load_personas()
    edges = load_edges()

    graph = nx.Graph()
    graph.add_nodes_from(personas)
    graph.add_edges_from(edges)

    groups = build_groups(graph)
    final_positions, group_of = build_positions(graph, groups)
    edges_ordered = order_edges(edges, group_of)

    video = OUT / "reference_style_network_formation_4k.mp4"
    final_png = OUT / "reference_style_network_final_4k.png"
    contact = OUT / "reference_style_network_formation_contact_sheet.jpg"

    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer")

    total_frames = DURATION * FPS
    for idx in range(total_frames):
        frame = draw_frame(idx, edges_ordered, groups, final_positions, group_of)
        writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        if idx and idx % (FPS * 5) == 0:
            print(f"rendered {idx // FPS}s / {DURATION}s")
    writer.release()

    draw_frame((DURATION - 1) * FPS, edges_ordered, groups, final_positions, group_of).save(final_png)
    make_contact_sheet(video, contact)

    cap = cv2.VideoCapture(str(video))
    summary = {
        "video": str(video),
        "final_png": str(final_png),
        "contact_sheet": str(contact),
        "source_personas": str(PERSONA_FILE),
        "source_network": str(NETWORK_FILE),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        "nodes": len(personas),
        "edges": len(edges),
        "density": round(len(edges) / (50 * 49 / 2), 3),
        "detected_groups": [len(g) for g in groups],
    }
    cap.release()
    (OUT / "reference_style_network_formation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
