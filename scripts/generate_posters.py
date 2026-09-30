from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

Image.MAX_IMAGE_PIXELS = None


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "posters"
TEXT = ROOT / "text-files"
STATS = ROOT / "stats"
UA_LOGO = OUT / "uarizona_logo.webp"

# 42 x 56 inches at 300 DPI. The design grid keeps the previous 3600-unit
# height and adds horizontal room for the wider print format.
S = 14 / 3
W, H = int(round(2700 * S)), int(round(3600 * S))


def p(v: float) -> int:
    return int(round(v * S))


def box(x0, y0, x1, y1):
    return (p(x0), p(y0), p(x1), p(y1))


def rgb(hex_value: str) -> tuple[int, int, int]:
    hex_value = hex_value.strip("#")
    return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    font_dir = Path("C:/Windows/Fonts")
    candidates = [
        font_dir / name,
        font_dir / f"{name}.ttf",
        font_dir / f"{name}.otf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), p(size))
    return ImageFont.truetype(str(font_dir / "arial.ttf"), p(size))


F = {
    "display": font("bahnschrift.ttf", 72),
    "display_big": font("bahnschrift.ttf", 91),
    "headline": font("bahnschrift.ttf", 46),
    "section": font("bahnschrift.ttf", 30),
    "section_big": font("bahnschrift.ttf", 38),
    "label": font("segoeuib.ttf", 20),
    "body": font("segoeui.ttf", 18),
    "body_bold": font("segoeuib.ttf", 18),
    "small": font("segoeui.ttf", 13),
    "small_bold": font("segoeuib.ttf", 13),
    "tiny": font("segoeui.ttf", 10),
    "metric": font("bahnschrift.ttf", 52),
    "metric2": font("bahnschrift.ttf", 38),
    "serif_title": font("georgiab.ttf", 54),
    "serif_head": font("georgiab.ttf", 34),
    "serif_italic": font("georgiai.ttf", 22),
    "serif_quote": font("georgiab.ttf", 28),
}


def draw_ua_mark(draw: ImageDraw.ImageDraw, rect):
    """Draw a simple UA-inspired mark when no official logo asset is available."""
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 22, "#AB0520", outline="#FFFFFF", width=1.2)
    inner = (x0 + p(8), y0 + p(8), x0 + p(108), y1 - p(8))
    rounded(draw, inner, 16, "#FFFFFF", outline="#0C234B", width=1.2)
    draw.text((inner[0] + p(26), inner[1] + p(8)), "A", font=F["serif_title"], fill="#AB0520")
    draw.text((x0 + p(132), y0 + p(22)), "THE UNIVERSITY", font=F["small_bold"], fill="#FFFFFF")
    draw.text((x0 + p(132), y0 + p(48)), "OF ARIZONA", font=F["label"], fill="#FFFFFF")


def draw_ua_logo(img: Image.Image, draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    if not UA_LOGO.exists():
        draw_ua_mark(draw, rect)
        return

    rounded(draw, rect, 20, "#FFFFFF", outline="#AB0520", width=1.4)
    logo = Image.open(UA_LOGO).convert("RGBA")
    # Remove the large white padding in the downloaded official card image.
    logo_rgb = logo.convert("RGB")
    white = Image.new("RGB", logo_rgb.size, (255, 255, 255))
    bbox = ImageChops.difference(logo_rgb, white).getbbox()
    if bbox:
        logo = logo.crop(bbox)

    pad_x, pad_y = p(22), p(18)
    max_w = x1 - x0 - 2 * pad_x
    max_h = y1 - y0 - 2 * pad_y
    scale = min(max_w / logo.width, max_h / logo.height)
    size = (max(1, int(logo.width * scale)), max(1, int(logo.height * scale)))
    logo = logo.resize(size, Image.Resampling.LANCZOS)
    px = x0 + (x1 - x0 - size[0]) // 2
    py = y0 + (y1 - y0 - size[1]) // 2
    img.paste(logo, (px, py), logo)

DARK = {
    "bg0": "#06111F",
    "bg1": "#102A43",
    "panel": "#0B1828",
    "panel2": "#10243A",
    "cyan": "#2FE6FF",
    "blue": "#4F8CFF",
    "gold": "#FFD166",
    "red": "#FF4D6D",
    "green": "#72F2A7",
    "ink": "#F5FBFF",
    "muted": "#B9C7D8",
}

LIGHT = {
    "paper": "#F8F1E7",
    "ink": "#172033",
    "muted": "#58677B",
    "blue": "#145DA0",
    "cyan": "#00A9C7",
    "red": "#AB0520",
    "gold": "#D99A1E",
    "green": "#177E63",
    "card": "#FFFDF8",
    "line": "#D7DFE8",
}

POLITICS = {
    "Democrat": "#2F80ED",
    "Republican": "#E84855",
}

METHOD_FILES = {
    "Global": TEXT / "global_gpt-4.1-mini_culture_us_0.adj",
    "Local": TEXT / "local_gpt-4.1-mini_n5_culture_us_0.adj",
    "Sequential": TEXT / "sequential_gpt-4.1-mini_n5_culture_us_0.adj",
    "Iterative": TEXT / "iterative_gpt-4.1-mini_n5_culture_us_0.adj",
}


def draw_text_size(draw: ImageDraw.ImageDraw, text: str, fnt) -> tuple[int, int]:
    b = draw.textbbox((0, 0), text, font=fnt)
    return b[2] - b[0], b[3] - b[1]


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        words = para.split()
        if not words:
            lines.append("")
            continue
        current: list[str] = []
        for word in words:
            trial = " ".join(current + [word])
            if draw_text_size(draw, trial, fnt)[0] <= max_w or not current:
                current.append(word)
            else:
                lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
    return lines


def text(draw, xy, content, fnt, fill, max_w=None, gap=6):
    x, y = xy
    if max_w is None:
        draw.text((x, y), content, font=fnt, fill=fill)
        return y + draw_text_size(draw, content, fnt)[1]
    for line in wrap(draw, content, fnt, max_w):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += draw_text_size(draw, line or "X", fnt)[1] + p(gap)
    return y


def rounded(draw, rect, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(rect, radius=p(radius), fill=fill, outline=outline, width=p(width))


def gradient(size, top, bottom):
    w, h = size
    img = Image.new("RGB", (w, h), top)
    d = ImageDraw.Draw(img)
    a, b = rgb(top), rgb(bottom)
    for y in range(h):
        t = y / max(1, h - 1)
        col = tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))
        d.line((0, y, w, y), fill=col)
    return img


def add_noise(img: Image.Image, strength=14):
    rnd = random.Random(7)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    pix = layer.load()
    step = p(3)
    for y in range(0, img.height, step):
        for x in range(0, img.width, step):
            v = rnd.randint(0, strength)
            pix[x, y] = (255, 255, 255, v)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def load_personas():
    raw = json.loads((TEXT / "us_50_gpt4o_w_interests.json").read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def parse_adj(path: Path) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(50))
    row = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for token in line.split():
            try:
                target = int(token)
            except ValueError:
                continue
            if 0 <= target < 50 and target != row:
                graph.add_edge(row, target)
        row += 1
    return graph


def load_graphs():
    return {name: parse_adj(path) for name, path in METHOD_FILES.items()}


def normalize_positions(pos, rect, pad=28):
    x0, y0, x1, y1 = rect
    xs = [v[0] for v in pos.values()]
    ys = [v[1] for v in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    out = {}
    for node, (x, y) in pos.items():
        nx_ = (x - min_x) / max(1e-9, max_x - min_x)
        ny_ = (y - min_y) / max(1e-9, max_y - min_y)
        out[node] = (
            x0 + pad + nx_ * (x1 - x0 - 2 * pad),
            y0 + pad + ny_ * (y1 - y0 - 2 * pad),
        )
    return out


def graph_stats(graph: nx.Graph, personas):
    edges = list(graph.edges())
    density = nx.density(graph)
    if not edges:
        same_party = 0.0
    else:
        same = sum(
            1
            for a, b in edges
            if personas[a].get("political affiliation") == personas[b].get("political affiliation")
        )
        same_party = same / len(edges)
    comps = sorted(nx.connected_components(graph), key=len, reverse=True)
    lcc = len(comps[0]) / graph.number_of_nodes() if comps else 0
    return {"edges": len(edges), "density": density, "same_party": same_party, "lcc": lcc}


def draw_network(draw, graph, personas, rect, theme, title="", subtitle="", dark=True, seed=4):
    x0, y0, x1, y1 = rect
    bg = theme["panel"] if dark else "#FFFFFF"
    rounded(draw, rect, 24, bg, outline=theme.get("cyan", LIGHT["line"]), width=1.4)
    header_h = p(78)
    draw.rounded_rectangle((x0, y0, x1, y0 + header_h), radius=p(24), fill=theme.get("panel2", "#EEF4F8"))
    draw.rectangle((x0, y0 + p(34), x1, y0 + header_h), fill=theme.get("panel2", "#EEF4F8"))
    draw.text((x0 + p(28), y0 + p(18)), title, font=F["section"], fill=theme["cyan"] if dark else theme["blue"])
    if subtitle:
        draw.text((x0 + p(28), y0 + p(52)), subtitle, font=F["tiny"], fill=theme["muted"])

    inner = (x0 + p(28), y0 + header_h + p(24), x1 - p(28), y1 - p(28))
    pos = nx.spring_layout(graph, seed=seed, k=0.56, iterations=180)
    pos = normalize_positions(pos, inner, pad=p(18))

    edge_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ed = ImageDraw.Draw(edge_layer)
    for a, b in graph.edges():
        ax, ay = pos[a]
        bx, by = pos[b]
        same = personas[a]["political affiliation"] == personas[b]["political affiliation"]
        col = rgb(theme["cyan"] if same else theme["red"])
        ed.line((ax, ay, bx, by), fill=(*col, 118 if dark else 155), width=p(1.2))
    base = draw.im
    tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    tmp.paste(edge_layer)
    draw._image = Image.alpha_composite(draw._image.convert("RGBA"), tmp).convert("RGB") if hasattr(draw, "_image") else None

    # ImageDraw cannot mutate the caller image through the private path above in all versions,
    # so return the edge layer for the caller to composite when needed.
    for node, (x, y) in pos.items():
        party = personas[node].get("political affiliation", "")
        color = POLITICS.get(party, theme["gold"])
        age = personas[node].get("age", 35)
        r = p(5 + min(70, int(age)) / 18)
        draw.ellipse((x - r - p(2), y - r - p(2), x + r + p(2), y + r + p(2)), fill="#FFFFFF" if dark else "#0D1B2A")
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    s = graph_stats(graph, personas)
    stat = f"{s['edges']} ties | density {s['density']:.3f} | same-party {s['same_party']:.0%}"
    draw.text((x0 + p(28), y1 - p(50)), stat, font=F["tiny"], fill=theme["muted"])
    return edge_layer


def composite_edges(img, edge_layer):
    return Image.alpha_composite(img.convert("RGBA"), edge_layer).convert("RGB")


def draw_mini_network(img, draw, graph, personas, rect, theme, title, seed, dark=True):
    edge_layer = draw_network(draw, graph, personas, rect, theme, title, "", dark=dark, seed=seed)
    return composite_edges(img, edge_layer)


def draw_people_constellation(draw, personas, rect, theme, dark=True):
    x0, y0, x1, y1 = rect
    bg = theme["panel"] if dark else theme["card"]
    rounded(draw, rect, 28, bg, outline=theme["cyan"] if dark else theme["line"], width=1.2)
    draw.text((x0 + p(28), y0 + p(24)), "50 PERSONA ROSTER", font=F["section"], fill=theme["gold"] if dark else theme["red"])
    draw.text((x0 + p(28), y0 + p(62)), "dots are real personas; color encodes political affiliation", font=F["small"], fill=theme["muted"])
    cols = 10
    gx0, gy0 = x0 + p(60), y0 + p(120)
    cell = min((x1 - x0 - p(130)) / cols, (y1 - y0 - p(200)) / 5)
    for node in range(50):
        row, col = divmod(node, cols)
        person = personas[node]
        cx = gx0 + col * cell + cell / 2
        cy = gy0 + row * cell + cell / 2
        radius = p(13)
        color = POLITICS.get(person["political affiliation"], theme["cyan"])
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline="#FFFFFF" if dark else "#1B2430", width=p(1.5))
    counts = Counter(p["political affiliation"] for p in personas.values())
    draw.text((x0 + p(60), y1 - p(64)), f"Democrat {counts['Democrat']}     Republican {counts['Republican']}", font=F["label"], fill=theme["ink"] if dark else theme["ink"])


def draw_metric_card(draw, rect, eyebrow, value, body, accent, theme, dark=True):
    x0, y0, x1, y1 = rect
    bg = theme["panel"] if dark else theme["card"]
    rounded(draw, rect, 26, bg, outline=accent, width=1.5)
    draw.text((x0 + p(26), y0 + p(22)), eyebrow.upper(), font=F["label"], fill=accent)
    draw.text((x0 + p(26), y0 + p(60)), value, font=F["metric"], fill=accent)
    text(draw, (x0 + p(30), y0 + p(130)), body, F["body_bold"], theme["ink"], x1 - x0 - p(60), gap=5)


def draw_horizontal_bars(draw, rect, labels, values, title, accent, theme, dark=True, formatter=lambda v: f"{v:.2f}"):
    x0, y0, x1, y1 = rect
    bg = theme["panel"] if dark else theme["card"]
    rounded(draw, rect, 26, bg, outline=accent, width=1.3)
    draw.text((x0 + p(26), y0 + p(22)), title.upper(), font=F["label"], fill=accent)
    max_v = max(values) if values else 1
    top = y0 + p(76)
    usable = y1 - top - p(28)
    row_h = usable / max(1, len(labels))
    for i, (label, val) in enumerate(zip(labels, values)):
        yy = top + i * row_h
        draw.text((x0 + p(28), int(yy + row_h * 0.26)), label, font=F["small"], fill=theme["ink"])
        bx0 = x0 + p(170)
        bx1 = x1 - p(92)
        bw = int((bx1 - bx0) * val / max_v)
        draw.rounded_rectangle((bx0, int(yy + row_h * 0.24), bx1, int(yy + row_h * 0.72)), radius=p(10), fill="#243449" if dark else "#E9EEF4")
        draw.rounded_rectangle((bx0, int(yy + row_h * 0.24), bx0 + bw, int(yy + row_h * 0.72)), radius=p(10), fill=accent)
        draw.text((bx1 + p(12), int(yy + row_h * 0.22)), formatter(val), font=F["tiny"], fill=theme["muted"])


def draw_pipeline(draw, rect, theme, dark=True):
    x0, y0, x1, y1 = rect
    bg = theme["panel"] if dark else theme["card"]
    rounded(draw, rect, 30, bg, outline=theme["cyan"], width=1.3)
    draw.text((x0 + p(28), y0 + p(24)), "EXPERIMENT DESIGN", font=F["section"], fill=theme["cyan"])
    steps = ["50 personas", "culture/language prompt", "GPT model", "adjacency matrix", "homophily + topology"]
    cy = y0 + p(115)
    for i, step in enumerate(steps):
        cx = x0 + p(68)
        draw.ellipse((cx - p(24), cy - p(24), cx + p(24), cy + p(24)), fill=theme["gold"])
        draw.text((cx - p(8), cy - p(13)), str(i + 1), font=F["label"], fill="#08111F")
        draw.text((x0 + p(118), cy - p(18)), step, font=F["body_bold"], fill=theme["ink"])
        if i < len(steps) - 1:
            draw.line((cx, cy + p(28), cx, cy + p(72)), fill=theme["cyan"], width=p(2))
        cy += p(100)


def derived_metrics(graphs, personas):
    rows = []
    for name, graph in graphs.items():
        s = graph_stats(graph, personas)
        rows.append((name, s))
    return rows


def external_findings():
    model = pd.read_csv(STATS / "cultural_study" / "model_divergence.csv")
    model_avg = model.groupby("model_pair")["edge_distance"].mean().sort_values()
    lang = pd.read_csv(STATS / "language_study" / "language_summary.csv")
    lang_same = lang[(lang["table"] == "homophily") & (lang["metric_name"] == "same_ratio")]
    ranges = lang_same.groupby("demo")["_metric_value"].agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
    culture = pd.read_csv(STATS / "cultural_study" / "culture_summary.csv")
    cult_net = culture[(culture["table"] == "network") & (culture["metric_name"] == "prop_nodes_lcc")]
    lcc_range = cult_net["_metric_value"].max() - cult_net["_metric_value"].min()
    return {
        "closest_model": model_avg.index[0],
        "closest_dist": model_avg.iloc[0],
        "farthest_model": model_avg.index[-1],
        "farthest_dist": model_avg.iloc[-1],
        "language_demo": ranges.index[0],
        "language_range": ranges.iloc[0],
        "lcc_range": lcc_range,
    }


def collect_model_inventory():
    detect_order = ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-3.5-turbo", "gpt-4o", "gpt-4.1"]
    rows = []
    for path in TEXT.glob("*.adj"):
        name = path.stem
        method = next((m for m in ["global", "local", "sequential", "iterative"] if name.startswith(f"{m}_")), None)
        model = next((m for m in detect_order if m in name), None)
        if not method or not model:
            continue
        graph = parse_adj(path)
        rows.append(
            {
                "file": path.name,
                "method": method,
                "model": model,
                "edges": graph.number_of_edges(),
                "density": nx.density(graph),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["model"] = pd.Categorical(
            frame["model"],
            categories=["gpt-3.5-turbo", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"],
            ordered=True,
        )
        frame = frame.sort_values(["model", "method", "file"])
    return frame


def demographic_breakdown(personas):
    age_bins = Counter()
    for person in personas.values():
        age = int(person.get("age", 0))
        if age <= 24:
            age_bins["18-24"] += 1
        elif age <= 34:
            age_bins["25-34"] += 1
        elif age <= 44:
            age_bins["35-44"] += 1
        elif age <= 54:
            age_bins["45-54"] += 1
        else:
            age_bins["55+"] += 1
    return {
        "Gender": Counter(p.get("gender", "Unknown") for p in personas.values()),
        "Age": age_bins,
        "Race / Ethnicity": Counter(p.get("race/ethnicity", "Unknown") for p in personas.values()),
        "Religion": Counter(p.get("religion", "Unknown") for p in personas.values()),
        "Political": Counter(p.get("political affiliation", "Unknown") for p in personas.values()),
    }


def story_findings():
    findings = external_findings()
    inventory = collect_model_inventory()
    dominance = pd.read_csv(STATS / "cultural_study" / "demographic_dominance.csv")
    top_counts = dominance["top_demo"].value_counts().to_dict()
    language = pd.read_csv(STATS / "language_study" / "language_summary.csv")
    lang_ranges = (
        language[(language["table"] == "homophily") & (language["metric_name"] == "same_ratio")]
        .groupby("demo")["_metric_value"]
        .agg(lambda s: float(s.max() - s.min()))
        .sort_values(ascending=False)
    )
    old_models = ["gpt-3.5-turbo", "gpt-4o"]
    old_count = int(inventory[inventory["model"].isin(old_models)].shape[0])
    new_count = int(inventory[~inventory["model"].isin(old_models)].shape[0])
    return {
        **findings,
        "inventory": inventory,
        "dominance_top": top_counts,
        "lang_ranges": lang_ranges,
        "total_graphs": int(inventory.shape[0]),
        "old_count": old_count,
        "new_count": new_count,
    }


def verified_capstone_data():
    culture_md = (STATS / "cultural_study" / "research_answers.md").read_text(encoding="utf-8")
    method_md = (STATS / "method_study" / "research_answers.md").read_text(encoding="utf-8")
    language_md = (STATS / "language_study" / "research_answers.md").read_text(encoding="utf-8")
    refresh_md = (STATS / "capstone_verification_refresh.md").read_text(encoding="utf-8")
    culture_condition = pd.read_csv(STATS / "cultural_study" / "condition_summary.csv")
    method_summary = pd.read_csv(STATS / "method_study" / "method_summary.csv")
    dominance_counts = {
        "culture": pd.read_csv(STATS / "cultural_study" / "demographic_dominance.csv")["top_demo"].value_counts(),
        "language": pd.read_csv(STATS / "language_study" / "demographic_dominance.csv")["top_demo"].value_counts(),
        "method": pd.read_csv(STATS / "method_study" / "demographic_dominance.csv")["top_demo"].value_counts(),
    }
    model_stage_distances = []
    for label, folder in [
        ("Culture", "cultural_study"),
        ("Language", "language_study"),
        ("Method", "method_study"),
    ]:
        distances = (
            pd.read_csv(STATS / folder / "model_divergence.csv")
            .groupby("model_pair")["edge_distance"]
            .mean()
            .sort_values()
        )
        model_stage_distances.append(
            {
                "stage": label,
                "closest_pair": distances.index[0],
                "closest": float(distances.iloc[0]),
                "farthest_pair": distances.index[-1],
                "farthest": float(distances.iloc[-1]),
            }
        )

    def grab(pattern, text):
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise ValueError(f"Could not parse pattern: {pattern}")
        return match.groups()

    def to_float(value):
        return float(value.rstrip("."))

    culture_demo, culture_range = grab(
        r"largest culture-driven homophily shift appears on `([^`]+)` with mean same-ratio range ([0-9]+(?:\.[0-9]+)?)",
        culture_md,
    )
    culture_topology, culture_topology_range = grab(
        r"widest cross-culture spread is `([^`]+)` with range ([0-9]+(?:\.[0-9]+)?)",
        culture_md,
    )
    culture_top_demo, culture_top_count = grab(
        r"`([^`]+)` is the most frequent top-ranked homophily dimension across conditions \((\d+) conditions\)",
        culture_md,
    )
    closest_model, closest_dist = grab(
        r"most consistent pair is `([^`]+)` with average edge distance ([0-9]+(?:\.[0-9]+)?)",
        culture_md,
    )
    farthest_model, farthest_dist = grab(
        r"most divergent pair is `([^`]+)` with average edge distance ([0-9]+(?:\.[0-9]+)?)",
        culture_md,
    )

    method_highest, method_highest_density = grab(
        r"`([^`]+)` produced the highest average density \(([0-9]+(?:\.[0-9]+)?)\)",
        method_md,
    )
    method_lowest, method_lowest_density = grab(
        r"`([^`]+)` produced the lowest average density \(([0-9]+(?:\.[0-9]+)?)\)",
        method_md,
    )

    language_demo, language_range = grab(
        r"largest language-driven homophily shift appears on `([^`]+)` with range ([0-9]+(?:\.[0-9]+)?)",
        language_md,
    )
    language_topology, language_topology_range = grab(
        r"widest cross-language spread is `([^`]+)` with range ([0-9]+(?:\.[0-9]+)?)",
        language_md,
    )
    closest_lang_pair, closest_lang_dist = grab(
        r"closest language pair was `([^`]+)` \(([0-9]+(?:\.[0-9]+)?)\)",
        language_md,
    )
    farthest_lang_pair, farthest_lang_dist = grab(
        r"farthest language pair was `([^`]+)` \(([0-9]+(?:\.[0-9]+)?)\)",
        language_md,
    )
    language_top_demo, language_top_count = grab(
        r"`([^`]+)` was the most frequent top-ranked homophily dimension across Step 4 conditions \((\d+) conditions\)",
        language_md,
    )

    culture_verified = grab(r"cultural_study: (\d+/\d+) passed", refresh_md)[0]
    method_verified = grab(r"method_study: (\d+/\d+) passed", refresh_md)[0]
    language_verified = grab(r"language_study: (\d+/\d+) passed", refresh_md)[0]

    sequential_density = float(
        culture_condition[
            (culture_condition["table"] == "network")
            & (culture_condition["metric_name"] == "density")
            & (culture_condition["method"] == "sequential")
        ]["_metric_value"].mean()
    )
    gpt41_sequential_density = []
    for model in ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]:
        model_density = float(
            culture_condition[
                (culture_condition["table"] == "network")
                & (culture_condition["metric_name"] == "density")
                & (culture_condition["method"] == "sequential")
                & (culture_condition["model"] == model)
            ]["_metric_value"].mean()
        )
        gpt41_sequential_density.append((model, model_density))
    four_method_density = [("Global", 0.0), ("Local", 0.0), ("Sequential", sequential_density), ("Iterative", 0.0)]
    density_lookup = {
        row["method"]: float(row["_metric_value"])
        for _, row in method_summary[
            (method_summary["table"] == "network") & (method_summary["metric_name"] == "density")
        ].iterrows()
    }
    four_method_density = [
        ("Global", density_lookup["global"]),
        ("Local", density_lookup["local"]),
        ("Sequential", sequential_density),
        ("Iterative", density_lookup["iterative"]),
    ]

    return {
        "culture_demo": culture_demo,
        "culture_range": to_float(culture_range),
        "culture_topology": culture_topology,
        "culture_topology_range": to_float(culture_topology_range),
        "culture_top_demo": culture_top_demo,
        "culture_top_count": int(culture_top_count),
        "closest_model": closest_model,
        "closest_dist": to_float(closest_dist),
        "farthest_model": farthest_model,
        "farthest_dist": to_float(farthest_dist),
        "method_highest": method_highest,
        "method_highest_density": to_float(method_highest_density),
        "method_lowest": method_lowest,
        "method_lowest_density": to_float(method_lowest_density),
        "language_demo": language_demo,
        "language_range": to_float(language_range),
        "language_topology": language_topology,
        "language_topology_range": to_float(language_topology_range),
        "closest_lang_pair": closest_lang_pair,
        "closest_lang_dist": to_float(closest_lang_dist),
        "farthest_lang_pair": farthest_lang_pair,
        "farthest_lang_dist": to_float(farthest_lang_dist),
        "language_top_demo": language_top_demo,
        "language_top_count": int(language_top_count),
        "culture_verified": culture_verified,
        "method_verified": method_verified,
        "language_verified": language_verified,
        "four_method_density": four_method_density,
        "sequential_density": sequential_density,
        "gpt41_sequential_density": gpt41_sequential_density,
        "dominance_counts": {
            key: {str(k): int(v) for k, v in counts.to_dict().items()}
            for key, counts in dominance_counts.items()
        },
        "political_top_total": int(
            dominance_counts["culture"].get("political affiliation", 0)
            + dominance_counts["language"].get("political affiliation", 0)
            + dominance_counts["method"].get("political affiliation", 0)
        ),
        "dominance_total": int(sum(counts.sum() for counts in dominance_counts.values())),
        "model_stage_distances": model_stage_distances,
        "capstone_total": 24 + 72 + 96,
    }


def legacy_repo_data():
    model_rows = [
        ("GPT-3.5 Turbo", "sequential_gpt-3.5-turbo"),
        ("GPT-4o", "sequential_gpt-4o"),
        ("Llama 3.1 8B", "sequential_llama3.1-8b"),
        ("Llama 3.1 70B", "sequential_llama3.1-70b"),
        ("Gemma 2 9B", "sequential_gemma2-9b"),
        ("Gemma 2 27B", "sequential_gemma2-27b"),
    ]
    sequential_density = []
    lcc_values = []
    for label, folder in model_rows:
        df = pd.read_csv(STATS / folder / "network_metrics.csv")
        sequential_density.append(
            (label, float(df[df["metric_name"] == "density"]["_metric_value"].mean()))
        )
        lcc_values.append(
            (label, float(df[df["metric_name"] == "prop_nodes_lcc"]["_metric_value"].mean()))
        )
    gpt35_method_density = []
    for label, folder in [
        ("Global", "global_gpt-3.5-turbo"),
        ("Local", "local_gpt-3.5-turbo"),
        ("Sequential", "sequential_gpt-3.5-turbo"),
    ]:
        df = pd.read_csv(STATS / folder / "network_metrics.csv")
        gpt35_method_density.append(
            (label, float(df[df["metric_name"] == "density"]["_metric_value"].mean()))
        )
    return {
        "model_labels": [row[0] for row in model_rows],
        "sequential_density": sequential_density,
        "gpt35_method_density": gpt35_method_density,
        "lcc_values": lcc_values,
        "legacy_methods_note": "Legacy notebook cells load global/local/sequential GPT baselines and sequential open-model comparisons.",
    }


def draw_arrow(draw, start, end, fill, width=4, head=16):
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=fill, width=p(width))
    ang = math.atan2(y1 - y0, x1 - x0)
    left = (
        x1 - math.cos(ang - math.pi / 6) * p(head),
        y1 - math.sin(ang - math.pi / 6) * p(head),
    )
    right = (
        x1 - math.cos(ang + math.pi / 6) * p(head),
        y1 - math.sin(ang + math.pi / 6) * p(head),
    )
    draw.polygon([(x1, y1), left, right], fill=fill)


def draw_network_icon(draw, rect, node_fill, edge_fill):
    x0, y0, x1, y1 = rect
    pts = [
        (x0 + (x1 - x0) * 0.18, y0 + (y1 - y0) * 0.72),
        (x0 + (x1 - x0) * 0.36, y0 + (y1 - y0) * 0.34),
        (x0 + (x1 - x0) * 0.58, y0 + (y1 - y0) * 0.22),
        (x0 + (x1 - x0) * 0.78, y0 + (y1 - y0) * 0.56),
        (x0 + (x1 - x0) * 0.50, y0 + (y1 - y0) * 0.78),
    ]
    for a, b in [(0, 1), (1, 2), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4)]:
        draw.line((*pts[a], *pts[b]), fill=edge_fill, width=p(3))
    for px, py in pts:
        r = p(8)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=node_fill, outline="#FFFFFF", width=p(1))


def draw_method_process_card(img, draw, rect, title, accent, description, mode, graph, personas, seed):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 26, "#FFFDF8", outline=accent, width=1.4)
    draw.rectangle((x0, y0, x1, y0 + p(10)), fill=accent)
    draw.text((x0 + p(22), y0 + p(24)), title, font=F["section_big"], fill=accent)
    text(draw, (x0 + p(22), y0 + p(74)), description, F["body_bold"], LIGHT["ink"], x1 - x0 - p(44), gap=5)

    panel = (x0 + p(24), y0 + p(142), x1 - p(24), y0 + p(340))
    rounded(draw, panel, 18, "#F6F9FD", outline=LIGHT["line"], width=1.1)

    if mode == "global":
        for row in range(2):
            for col in range(3):
                cx = panel[0] + p(42) + col * p(28)
                cy = panel[1] + p(48) + row * p(28)
                draw.ellipse((cx - p(7), cy - p(7), cx + p(7), cy + p(7)), fill=accent)
        rounded(draw, (panel[0] + p(126), panel[1] + p(36), panel[0] + p(238), panel[1] + p(106)), 14, accent)
        draw.text((panel[0] + p(156), panel[1] + p(58)), "LLM", font=F["label"], fill="#FFFFFF")
        draw_arrow(draw, (panel[0] + p(102), panel[1] + p(68)), (panel[0] + p(126), panel[1] + p(68)), accent, width=3)
        draw_arrow(draw, (panel[0] + p(238), panel[1] + p(68)), (panel[0] + p(300), panel[1] + p(68)), accent, width=3)
        draw_network_icon(draw, (panel[0] + p(300), panel[1] + p(26), panel[0] + p(392), panel[1] + p(114)), accent, LIGHT["ink"])
        draw.text((panel[0] + p(20), panel[1] + p(132)), "one prompt sees the full roster", font=F["small"], fill=LIGHT["muted"])
    elif mode == "local":
        for i in range(3):
            cy = panel[1] + p(40) + i * p(40)
            draw.ellipse((panel[0] + p(18), cy, panel[0] + p(44), cy + p(26)), fill=accent)
            draw_arrow(draw, (panel[0] + p(54), cy + p(13)), (panel[0] + p(126), cy + p(13)), accent, width=3)
            for j in range(3):
                cx = panel[0] + p(150) + j * p(22)
                draw.ellipse((cx, cy + p(2), cx + p(16), cy + p(18)), fill="#86D5FF" if j != i % 3 else LIGHT["red"])
        draw.text((panel[0] + p(18), panel[1] + p(154)), "each persona chooses independently", font=F["small"], fill=LIGHT["muted"])
    elif mode == "sequential":
        for idx in range(3):
            cx = panel[0] + p(42) + idx * p(70)
            cy = panel[1] + p(62)
            draw.ellipse((cx - p(17), cy - p(17), cx + p(17), cy + p(17)), fill=accent)
            draw.text((cx - p(8), cy - p(12)), str(idx + 1), font=F["label"], fill="#FFFFFF")
            if idx < 2:
                draw_arrow(draw, (cx + p(18), cy), (cx + p(52), cy), accent, width=3)
        rounded(draw, (panel[0] + p(246), panel[1] + p(24), panel[0] + p(386), panel[1] + p(94)), 14, "#FFFFFF", outline=accent, width=1.2)
        draw.text((panel[0] + p(262), panel[1] + p(34)), "# friends", font=F["small"], fill=LIGHT["ink"])
        draw.text((panel[0] + p(262), panel[1] + p(58)), "friend IDs", font=F["small"], fill=LIGHT["ink"])
        draw.text((panel[0] + p(18), panel[1] + p(154)), "later personas see the current graph state", font=F["small"], fill=LIGHT["muted"])
    else:
        draw_network_icon(draw, (panel[0] + p(136), panel[1] + p(26), panel[0] + p(246), panel[1] + p(126)), accent, LIGHT["ink"])
        draw.arc((panel[0] + p(92), panel[1] + p(12), panel[0] + p(282), panel[1] + p(142)), start=210, end=20, fill=accent, width=p(4))
        draw.arc((panel[0] + p(100), panel[1] + p(20), panel[0] + p(290), panel[1] + p(150)), start=30, end=200, fill=LIGHT["gold"], width=p(4))
        draw.text((panel[0] + p(302), panel[1] + p(42)), "+ add", font=F["body_bold"], fill=accent)
        draw.text((panel[0] + p(302), panel[1] + p(78)), "- drop", font=F["body_bold"], fill=LIGHT["gold"])
        draw.text((panel[0] + p(18), panel[1] + p(166)), "the graph is revised across multiple rounds", font=F["small"], fill=LIGHT["muted"])

    network_rect = (x0 + p(24), y0 + p(370), x1 - p(24), y1 - p(24))
    img = draw_mini_network(img, draw, graph, personas, network_rect, LIGHT, "Representative output", seed=seed, dark=False)
    return img


def draw_legacy_capstone_panel(draw, rect, legacy, capstone):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 30, "#FEFBF4", outline=LIGHT["line"], width=1.3)
    draw.text((x0 + p(28), y0 + p(26)), "BASELINE REPO VS OUR STUDY", font=F["section_big"], fill=LIGHT["ink"])
    draw.text((x0 + p(28), y0 + p(74)), "We separate the older SNAP baseline models from the controlled GPT-4.1 study reported on this poster.", font=F["small"], fill=LIGHT["muted"])

    left = (x0 + p(28), y0 + p(120), x0 + p(620), y1 - p(28))
    right = (x0 + p(652), y0 + p(120), x1 - p(28), y1 - p(28))
    rounded(draw, left, 24, "#F6F9FD", outline=LIGHT["line"], width=1.2)
    rounded(draw, right, 24, "#FFF7F2", outline=LIGHT["line"], width=1.2)

    draw.text((left[0] + p(22), left[1] + p(18)), "SNAP BASE REPO", font=F["label"], fill=LIGHT["blue"])
    draw.text((left[0] + p(22), left[1] + p(48)), "Legacy model set", font=F["serif_head"], fill=LIGHT["ink"])
    text(draw, (left[0] + p(22), left[1] + p(96)), "Notebook loading cells use GPT global/local/sequential baselines and sequential open-model comparisons for GPT-4o, Llama, and Gemma.", F["body_bold"], LIGHT["ink"], p(510), gap=5)
    chip_x = left[0] + p(22)
    chip_y = left[1] + p(168)
    chip_palette = ["#20314B", "#1A6AA4", "#3B5B7A", "#58799C", "#5F7F91", "#82989E"]
    for idx, label in enumerate(legacy["model_labels"]):
        chip_x = draw_chip(draw, (chip_x, chip_y), label, chip_palette[idx], ink="#FFFFFF")
        if chip_x > left[2] - p(180):
            chip_x = left[0] + p(22)
            chip_y += p(46)

    draw.text((right[0] + p(22), right[1] + p(18)), "UA CAPSTONE STUDY", font=F["label"], fill=LIGHT["red"])
    draw.text((right[0] + p(22), right[1] + p(48)), "Experiment matrix", font=F["serif_head"], fill=LIGHT["ink"])
    right_text_w = right[2] - right[0] - p(44)
    text(draw, (right[0] + p(22), right[1] + p(96)), "Same roster. GPT-4.1 models. Culture, language, and method sweeps.", F["body_bold"], LIGHT["ink"], right_text_w, gap=5)
    chip_x = right[0] + p(22)
    for label, fill, ink in [
        ("gpt-4.1", LIGHT["red"], "#FFFFFF"),
        ("gpt-4.1-mini", LIGHT["cyan"], "#FFFFFF"),
        ("gpt-4.1-nano", LIGHT["gold"], LIGHT["ink"]),
    ]:
        chip_x = draw_chip(draw, (chip_x, right[1] + p(150)), label, fill, ink=ink)

    step_boxes = [
        ("Step 2", capstone["culture_verified"], "Sequential culture study", LIGHT["blue"]),
        ("Step 3", capstone["method_verified"], "Adds global, local, iterative", LIGHT["red"]),
        ("Step 4", capstone["language_verified"], "Language-only study", LIGHT["green"]),
    ]
    inner_x0 = right[0] + p(22)
    inner_x1 = right[2] - p(22)
    gap = p(10)
    step_w = int((inner_x1 - inner_x0 - 2 * gap) / 3)
    for idx, (step, verified, desc, accent) in enumerate(step_boxes):
        bx0 = inner_x0 + idx * (step_w + gap)
        bx1 = bx0 + step_w
        by0 = right[1] + p(216)
        by1 = by0 + p(144)
        rounded(draw, (bx0, by0, bx1, by1), 18, "#FFFFFF", outline=accent, width=1.2)
        draw.text((bx0 + p(12), by0 + p(14)), step.upper(), font=F["tiny"], fill=accent)
        draw.text((bx0 + p(12), by0 + p(46)), verified, font=F["metric2"], fill=accent)
        text(draw, (bx0 + p(12), by0 + p(98)), desc, F["tiny"], LIGHT["ink"], step_w - p(24), gap=3)
    text(draw, (right[0] + p(22), right[1] + p(374)), f"{capstone['capstone_total']} generated study graphs across all three stages.", F["body_bold"], LIGHT["ink"], right_text_w, gap=5)


def draw_chip(draw, xy, label, fill, ink="#FFFFFF", outline=None):
    x, y = xy
    tw, th = draw_text_size(draw, label, F["small"])
    rect = (x, y, x + tw + p(30), y + th + p(18))
    rounded(draw, rect, 18, fill, outline=outline, width=1.2)
    draw.text((x + p(15), y + p(8)), label, font=F["small"], fill=ink)
    return rect[2] + p(10)


def draw_group_bars(draw, rect, title, items, accent, theme, bg=None, tiny=False):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 22, bg or LIGHT["card"], outline=theme["line"], width=1.2)
    draw.text((x0 + p(18), y0 + p(16)), title.upper(), font=F["label"], fill=accent)
    max_v = max(v for _, v in items) if items else 1
    top = y0 + p(56)
    row_h = (y1 - top - p(18)) / max(1, len(items))
    label_font = F["tiny"] if tiny else F["small"]
    for idx, (label, value) in enumerate(items):
        yy = top + idx * row_h
        draw.text((x0 + p(18), int(yy + row_h * 0.18)), label, font=label_font, fill=theme["ink"])
        bx0 = x0 + p(140)
        bx1 = x1 - p(48)
        by0 = int(yy + row_h * 0.22)
        by1 = int(yy + row_h * 0.66)
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=p(8), fill="#E6ECF2")
        fill_w = int((bx1 - bx0) * value / max_v)
        draw.rounded_rectangle((bx0, by0, bx0 + fill_w, by1), radius=p(8), fill=accent)
        value_label = f"{value:.3f}" if isinstance(value, float) and value < 1 else f"{int(value)}"
        draw.text((bx1 - p(4), int(yy + row_h * 0.14)), value_label, font=F["tiny"], fill=theme["muted"], anchor="ra")


def draw_sequential_density_comparison(draw, rect, legacy_items, capstone_items):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 22, "#FFFDF9", outline=LIGHT["line"], width=1.2)
    draw.text((x0 + p(18), y0 + p(16)), "SEQUENTIAL MEAN DENSITY", font=F["label"], fill=LIGHT["blue"])
    draw.text((x0 + p(18), y0 + p(44)), "Baseline models and our GPT-4.1 family are shown separately.", font=F["tiny"], fill=LIGHT["muted"])

    rows = [("BASE SNAP", None, None, LIGHT["blue"])]
    rows.extend([("base", label, value, LIGHT["blue"]) for label, value in legacy_items])
    rows.append(("OUR STUDY", None, None, LIGHT["red"]))
    rows.extend([("study", label, value, LIGHT["red"] if "nano" not in label else LIGHT["gold"]) for label, value in capstone_items])

    max_v = 0.40
    top = y0 + p(82)
    row_h = (y1 - top - p(20)) / len(rows)
    label_x = x0 + p(18)
    bar_x0 = x0 + p(210)
    bar_x1 = x1 - p(74)
    value_x = x1 - p(22)
    for idx, (group, label, value, accent) in enumerate(rows):
        yy = top + idx * row_h
        if value is None:
            line_y = int(yy + row_h * 0.50)
            draw.text((label_x, int(yy + row_h * 0.22)), group, font=F["tiny"], fill=accent)
            draw.line((x0 + p(100), line_y, x1 - p(22), line_y), fill="#D8E2EC", width=p(1))
            continue
        draw.text((label_x, int(yy + row_h * 0.16)), label, font=F["tiny"], fill=LIGHT["ink"])
        bx0 = bar_x0
        bx1 = bar_x1
        by0 = int(yy + row_h * 0.18)
        by1 = int(yy + row_h * 0.66)
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=p(7), fill="#E6ECF2")
        fill_w = max(p(3), int((bx1 - bx0) * value / max_v))
        draw.rounded_rectangle((bx0, by0, bx0 + fill_w, by1), radius=p(7), fill=accent)
        draw.text((value_x, int(yy + row_h * 0.13)), f"{value:.3f}", font=F["tiny"], fill=LIGHT["muted"], anchor="ra")


def draw_capstone_evidence_panel(draw, rect, capstone):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 24, "#FFFDF9", outline=LIGHT["line"], width=1.2)
    draw.text((x0 + p(18), y0 + p(16)), "WHAT OUR NEW STUDY ADDS", font=F["label"], fill=LIGHT["red"])
    text(
        draw,
        (x0 + p(18), y0 + p(48)),
        "Not a density contest: the GPT-4.1 experiments test whether the same roster changes under culture, language, method, and model-size controls.",
        F["tiny"],
        LIGHT["muted"],
        x1 - x0 - p(36),
        gap=3,
    )

    card_gap = p(12)
    card_w = int((x1 - x0 - p(36) - 2 * card_gap) / 3)
    card_y0 = y0 + p(120)
    stat_cards = [
        ("192", "generated graphs", LIGHT["red"]),
        (f"{capstone['political_top_total']}/{capstone['dominance_total']}", "top-rank cases are politics", LIGHT["gold"]),
        (f"{capstone['language_range']:.3f}", "religion language shift", LIGHT["cyan"]),
    ]
    for idx, (value, label, accent) in enumerate(stat_cards):
        cx0 = x0 + p(18) + idx * (card_w + card_gap)
        rounded(draw, (cx0, card_y0, cx0 + card_w, card_y0 + p(132)), 18, "#FFFFFF", outline=accent, width=1.2)
        draw.text((cx0 + p(14), card_y0 + p(18)), value, font=F["metric2"], fill=accent)
        text(draw, (cx0 + p(14), card_y0 + p(70)), label, F["tiny"], LIGHT["ink"], card_w - p(28), gap=3)

    draw.text((x0 + p(18), y0 + p(286)), "MODEL SIZE SIGNAL", font=F["label"], fill=LIGHT["ink"])
    draw.text((x0 + p(18), y0 + p(316)), "Full GPT-4.1 and mini stay closest; nano diverges most.", font=F["tiny"], fill=LIGHT["muted"])
    max_dist = 0.16
    label_x = x0 + p(18)
    bar_x0 = x0 + p(120)
    bar_x1 = x1 - p(74)
    value_x = x1 - p(18)
    row_y = y0 + p(356)
    row_h = p(62)
    for idx, item in enumerate(capstone["model_stage_distances"]):
        yy = row_y + idx * row_h
        draw.text((label_x, yy + p(12)), item["stage"], font=F["tiny"], fill=LIGHT["ink"])
        for sub_idx, (value, accent) in enumerate([(item["closest"], LIGHT["cyan"]), (item["farthest"], LIGHT["red"])]):
            by0 = yy + p(6 + sub_idx * 24)
            by1 = by0 + p(13)
            draw.rounded_rectangle((bar_x0, by0, bar_x1, by1), radius=p(6), fill="#E6ECF2")
            fill_w = int((bar_x1 - bar_x0) * value / max_dist)
            draw.rounded_rectangle((bar_x0, by0, bar_x0 + fill_w, by1), radius=p(6), fill=accent)
        draw.text((value_x, yy + p(3)), f"{item['closest']:.3f}", font=F["tiny"], fill=LIGHT["cyan"], anchor="ra")
        draw.text((value_x, yy + p(27)), f"{item['farthest']:.3f}", font=F["tiny"], fill=LIGHT["red"], anchor="ra")

    rounded(draw, (x0 + p(18), y1 - p(138), x1 - p(18), y1 - p(18)), 20, LIGHT["ink"], outline=LIGHT["ink"], width=1)
    text(
        draw,
        (x0 + p(38), y1 - p(114)),
        "How to say it: the new models are useful because they expose controlled sensitivity. Mini tracks full GPT-4.1 closely, nano shifts farther, and prompt framing changes the social structure.",
        F["body_bold"],
        "#FFFFFF",
        x1 - x0 - p(76),
        gap=5,
    )


def draw_demographic_breakdown(draw, rect, personas):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 30, "#FFFDF9", outline=LIGHT["line"], width=1.3)
    draw.text((x0 + p(28), y0 + p(26)), "WHO ENTERS THE NETWORK?", font=F["section_big"], fill=LIGHT["ink"])
    draw.text((x0 + p(28), y0 + p(74)), "The roster stays fixed at 50 personas so prompt context, not the people, is what changes.", font=F["small"], fill=LIGHT["muted"])

    groups = demographic_breakdown(personas)
    cards = [
        ("Gender", groups["Gender"], LIGHT["cyan"]),
        ("Political", groups["Political"], LIGHT["red"]),
        ("Age", groups["Age"], LIGHT["gold"]),
        ("Religion", groups["Religion"], LIGHT["green"]),
        ("Race / Ethnicity", groups["Race / Ethnicity"], LIGHT["blue"]),
    ]
    col_w = (x1 - x0 - p(84)) // 2
    row_h = p(150)
    positions = [
        (x0 + p(28), y0 + p(122)),
        (x0 + p(28) + col_w + p(28), y0 + p(122)),
        (x0 + p(28), y0 + p(122) + row_h + p(24)),
        (x0 + p(28) + col_w + p(28), y0 + p(122) + row_h + p(24)),
        (x0 + p(28), y0 + p(122) + 2 * (row_h + p(24))),
    ]
    spans = [1, 1, 1, 1, 2]
    for (title, counter, accent), (px, py), span in zip(cards, positions, spans):
        width = col_w if span == 1 else 2 * col_w + p(28)
        items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        draw_group_bars(draw, (px, py, px + width, py + row_h), title, items, accent, LIGHT, bg="#FFFCF7", tiny=title == "Race / Ethnicity")


def draw_project_arc(draw, rect, findings):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 30, "#FEFBF4", outline=LIGHT["line"], width=1.3)
    draw.text((x0 + p(28), y0 + p(26)), "PROJECT ARC", font=F["section_big"], fill=LIGHT["ink"])
    draw.text((x0 + p(28), y0 + p(74)), "This poster should read as an evolution of the original repo, not a disconnected refresh.", font=F["small"], fill=LIGHT["muted"])

    left = (x0 + p(28), y0 + p(124), x0 + p(460), y0 + p(420))
    right = (x0 + p(635), y0 + p(124), x1 - p(28), y0 + p(420))
    rounded(draw, left, 24, "#F5F8FC", outline=LIGHT["line"], width=1.2)
    rounded(draw, right, 24, "#FFF7F2", outline=LIGHT["line"], width=1.2)

    draw.text((left[0] + p(24), left[1] + p(20)), "ORIGINAL REPO BASELINE", font=F["label"], fill=LIGHT["blue"])
    draw.text((left[0] + p(24), left[1] + p(56)), "gpt-3.5-turbo + gpt-4o", font=F["serif_head"], fill=LIGHT["ink"])
    text(draw, (left[0] + p(24), left[1] + p(106)), "Legacy baseline runs span three generation methods: global, local, and sequential.", F["body_bold"], LIGHT["ink"], p(370), gap=5)
    draw.text((left[0] + p(24), left[1] + p(212)), str(findings["old_count"]), font=F["metric"], fill=LIGHT["blue"])
    draw.text((left[0] + p(126), left[1] + p(232)), "graph files in workspace", font=F["small"], fill=LIGHT["muted"])

    draw.text((right[0] + p(24), right[1] + p(20)), "CAPSTONE EXTENSION", font=F["label"], fill=LIGHT["red"])
    draw.text((right[0] + p(24), right[1] + p(56)), "gpt-4.1, mini, nano", font=F["serif_head"], fill=LIGHT["ink"])
    text(draw, (right[0] + p(24), right[1] + p(106)), "Added the iterative method, cultural framing, and prompt-language variation while keeping the same 50-person roster.", F["body_bold"], LIGHT["ink"], p(500), gap=5)
    draw.text((right[0] + p(24), right[1] + p(212)), str(findings["new_count"]), font=F["metric"], fill=LIGHT["red"])
    draw.text((right[0] + p(126), right[1] + p(232)), "new study graph files", font=F["small"], fill=LIGHT["muted"])

    draw.line((left[2] + p(32), y0 + p(270), right[0] - p(32), y0 + p(270)), fill=LIGHT["gold"], width=p(4))
    draw.polygon(
        [
            (right[0] - p(36), y0 + p(252)),
            (right[0], y0 + p(270)),
            (right[0] - p(36), y0 + p(288)),
        ],
        fill=LIGHT["gold"],
    )

    draw.text((x0 + p(28), y1 - p(114)), "Full model coverage on the poster", font=F["label"], fill=LIGHT["ink"])
    chip_x = x0 + p(28)
    chip_y = y1 - p(76)
    for label, fill, ink in [
        ("gpt-3.5-turbo", LIGHT["ink"], "#FFFFFF"),
        ("gpt-4o", LIGHT["blue"], "#FFFFFF"),
        ("gpt-4.1", LIGHT["red"], "#FFFFFF"),
        ("gpt-4.1-mini", LIGHT["cyan"], "#FFFFFF"),
        ("gpt-4.1-nano", LIGHT["gold"], LIGHT["ink"]),
    ]:
        chip_x = draw_chip(draw, (chip_x, chip_y), label, fill, ink=ink)


def draw_model_density_panel(draw, rect, findings):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 30, "#FFFDF9", outline=LIGHT["line"], width=1.3)
    draw.text((x0 + p(28), y0 + p(24)), "MODEL FAMILY PATTERN", font=F["section_big"], fill=LIGHT["ink"])
    draw.text((x0 + p(28), y0 + p(72)), "Mean density by model family using the real `.adj` outputs available in this repo.", font=F["small"], fill=LIGHT["muted"])
    inventory = findings["inventory"]
    means = inventory.groupby(["model", "method"], observed=False)["density"].mean()
    global_items = []
    sequential_items = []
    for model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]:
        if (model, "global") in means.index:
            global_items.append((model, float(means[(model, "global")])))
        if (model, "sequential") in means.index:
            sequential_items.append((model, float(means[(model, "sequential")])))
    draw_group_bars(draw, (x0 + p(28), y0 + p(118), x1 - p(28), y0 + p(388)), "Global mean density", global_items, LIGHT["cyan"], LIGHT, bg="#FFFFFF", tiny=True)
    draw_group_bars(draw, (x0 + p(28), y0 + p(416), x1 - p(28), y0 + p(686)), "Sequential mean density", sequential_items, LIGHT["red"], LIGHT, bg="#FFFFFF", tiny=True)
    footer = f"Closest 4.1 pair: {findings['closest_model']} ({findings['closest_dist']:.3f}). Farthest pair: {findings['farthest_model']} ({findings['farthest_dist']:.3f})."
    text(draw, (x0 + p(28), y1 - p(96)), footer, F["body_bold"], LIGHT["ink"], x1 - x0 - p(56), gap=5)


def draw_insight_panel(draw, rect, findings):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 30, LIGHT["ink"], outline=LIGHT["ink"], width=1.3)
    draw.text((x0 + p(28), y0 + p(24)), "WHAT THE RESULTS SAY", font=F["section_big"], fill="#FFFFFF")
    draw.text((x0 + p(28), y0 + p(74)), "The network is not fixed. The prompt and model family steer it.", font=F["serif_italic"], fill="#DCE6F5")

    draw.text((x0 + p(28), y0 + p(136)), "10 / 12", font=F["metric"], fill=LIGHT["gold"])
    text(draw, (x0 + p(195), y0 + p(150)), "cultural conditions are led by political affiliation as the top same-ratio demographic.", F["body_bold"], "#FFFFFF", p(420), gap=5)

    draw.text((x0 + p(28), y0 + p(276)), f"{findings['lcc_range']:.3f}", font=F["metric2"], fill=LIGHT["cyan"])
    text(draw, (x0 + p(150), y0 + p(284)), "largest culture-driven spread in network topology appears in `prop_nodes_lcc`.", F["body_bold"], "#FFFFFF", p(470), gap=5)

    draw.text((x0 + p(28), y0 + p(394)), findings["language_demo"].title(), font=F["serif_head"], fill="#FFFFFF")
    text(draw, (x0 + p(28), y0 + p(444)), f"shows the largest prompt-language homophily shift ({findings['language_range']:.3f}), so translation is an experimental condition, not a cosmetic change.", F["body_bold"], "#FFFFFF", p(560), gap=5)

    rounded(draw, (x0 + p(28), y1 - p(170), x1 - p(28), y1 - p(34)), 22, "#132740", outline="#27486B", width=1.2)
    text(draw, (x0 + p(48), y1 - p(144)), "Presentation takeaway: synthetic societies can look plausible while still being deeply prompt-sensitive. That is the cautionary result worth telling from the stage.", F["serif_quote"], "#FFFFFF", x1 - x0 - p(96), gap=6)


def poster_story():
    personas = load_personas()
    capstone = verified_capstone_data()
    legacy = legacy_repo_data()
    graphs = load_graphs()

    img = Image.new("RGB", (W, H), LIGHT["paper"])
    draw = ImageDraw.Draw(img)

    hero = gradient((W, p(800)), "#11263A", "#153D59")
    img.paste(hero, (0, 0))
    draw = ImageDraw.Draw(img)

    random.seed(31)
    points = [(random.randint(p(40), W - p(40)), random.randint(p(40), p(760))) for _ in range(95)]
    for i, a in enumerate(points):
        for j in range(i + 1, min(len(points), i + 14)):
            b = points[j]
            if math.dist(a, b) < p(300) and (i + j) % 4 == 0:
                draw.line((a[0], a[1], b[0], b[1]), fill="#23526F", width=p(1))
    for idx, (px, py) in enumerate(points):
        col = "#57D8EB" if idx % 5 else "#F5BF4F"
        r = p(2.5 if idx % 5 else 3.2)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=col)

    draw.text((p(108), p(78)), "LLMs AS", font=F["display"], fill="#FFFFFF")
    draw.text((p(108), p(166)), "SOCIAL NETWORK", font=F["display_big"], fill="#FFFFFF")
    draw.text((p(108), p(272)), "GENERATORS", font=F["display_big"], fill=LIGHT["cyan"])
    draw_ua_logo(img, draw, box(1108, 104, 1528, 206))
    draw.text((p(112), p(392)), "How we extend the base SNAP repo", font=F["serif_italic"], fill="#E8F3FF")
    text(
        draw,
        (p(112), p(442)),
        "We build on the original SNAP repo, which compares GPT, Llama, and Gemma baselines. In our study, we hold the 50-person roster fixed and vary culture, prompt language, generation method, and GPT-4.1 model size to test what actually changes the network.",
        F["body_bold"],
        "#DDEBFA",
        p(1120),
        gap=6,
    )

    chip_x = p(112)
    chip_y = p(566)
    draw.text((p(112), p(540)), "Base repo models", font=F["label"], fill="#DDEBFA")
    for label, fill, ink in [
        ("gpt-3.5-turbo", "#223B58", "#FFFFFF"),
        ("gpt-4o", "#1B6CA8", "#FFFFFF"),
        ("llama3.1-8b", "#3B5B7A", "#FFFFFF"),
        ("llama3.1-70b", "#58799C", "#FFFFFF"),
        ("gemma2-9b", "#5F7F91", "#FFFFFF"),
        ("gemma2-27b", "#82989E", "#FFFFFF"),
    ]:
        chip_x = draw_chip(draw, (chip_x, chip_y), label, fill, ink=ink)
    chip_x = p(112)
    chip_y = p(620)
    draw.text((p(112), p(594)), "Capstone models", font=F["label"], fill="#DDEBFA")
    for label, fill, ink in [
        ("gpt-4.1", "#A92C48", "#FFFFFF"),
        ("gpt-4.1-mini", "#17AFC7", "#FFFFFF"),
        ("gpt-4.1-nano", "#E4B64D", LIGHT["ink"]),
    ]:
        chip_x = draw_chip(draw, (chip_x, chip_y), label, fill, ink=ink)

    draw.text(
        (p(112), p(660)),
        "Sai Hemanth Kilaru | Sriram Theerdh Manikyala | Raghav Upadhyay | Sri Sai Kumar Ramavath | Srivika Nunavathu",
        font=F["small_bold"],
        fill="#FFFFFF",
    )
    draw.text((p(112), p(698)), "University of Arizona | INFC 698 Capstone | Instructor: Dr. Dalal Alharthi | April 2026", font=F["small_bold"], fill="#DDEBFA")

    hero_theme = dict(DARK)
    hero_theme["panel"] = "#132740"
    hero_theme["panel2"] = "#1A3551"
    hero_graph = parse_adj(TEXT / "sequential_gpt-4.1-mini_n5_culture_us_0.adj")
    img = draw_mini_network(img, draw, hero_graph, personas, box(1580, 96, 2596, 734), hero_theme, "One Fixed Roster, Multiple Outcomes", seed=88, dark=True)
    draw = ImageDraw.Draw(img)
    rounded(draw, box(1614, 586, 2568, 708), 20, "#10263A", outline="#2C6789", width=1.2)
    draw.text((p(1646), p(610)), "50 personas", font=F["label"], fill=LIGHT["gold"])
    draw.text((p(1888), p(610)), "3 GPT-4.1 models", font=F["label"], fill=LIGHT["cyan"])
    draw.text((p(2180), p(610)), "4 methods", font=F["label"], fill="#FFFFFF")
    draw.text((p(1646), p(648)), "4 cultures", font=F["label"], fill="#FFFFFF")
    draw.text((p(1888), p(648)), "4 languages", font=F["label"], fill="#FFFFFF")
    draw.text((p(2180), p(648)), "192 generated graphs", font=F["label"], fill=LIGHT["gold"])

    draw_legacy_capstone_panel(draw, box(104, 860, 1308, 1464), legacy, capstone)
    draw_demographic_breakdown(draw, box(1350, 860, 2596, 1464), personas)

    draw.text((p(104), p(1560)), "FOUR METHODS, FOUR DIFFERENT GENERATION PROCESSES", font=F["section_big"], fill=LIGHT["ink"])
    draw.text((p(104), p(1606)), "These are not just four similar pictures. Each method gives the model different information and builds the graph in a different way.", font=F["small"], fill=LIGHT["muted"])
    method_descriptions = {
        "Global": "One prompt sees the whole roster and proposes the graph in a single pass.",
        "Local": "Each persona chooses friends separately without seeing the evolving network state.",
        "Sequential": "People are processed one by one, so later choices can react to the current graph state.",
        "Iterative": "A starting graph is revised through repeated add and drop rounds.",
    }
    mini_w = 600
    mini_gap = 34
    start_y = 1660
    for idx, (name, graph) in enumerate(graphs.items()):
        x = 104 + idx * (mini_w + mini_gap)
        rect = box(x, start_y, x + mini_w, start_y + 650)
        img = draw_method_process_card(
            img,
            draw,
            rect,
            name,
            [LIGHT["blue"], LIGHT["red"], LIGHT["gold"], LIGHT["green"]][idx],
            method_descriptions[name],
            name.lower(),
            graph,
            personas,
            140 + idx,
        )
        draw = ImageDraw.Draw(img)

    draw_capstone_evidence_panel(draw, box(104, 2388, 940, 3076), capstone)
    rounded(draw, box(976, 2388, 1796, 3076), 30, LIGHT["ink"], outline=LIGHT["ink"], width=1.3)
    draw.text((p(1004), p(2414)), "KEY STUDY FINDINGS", font=F["section_big"], fill="#FFFFFF")
    draw.text((p(1004), p(2468)), "Main empirical signals from the controlled capstone study.", font=F["small"], fill="#DCE6F5")
    draw.text((p(1004), p(2540)), capstone["culture_demo"].title(), font=F["serif_head"], fill=LIGHT["gold"])
    text(draw, (p(1004), p(2592)), f"is the largest cross-culture homophily shift ({capstone['culture_range']:.3f}) and the most frequent top-ranked dimension across culture conditions ({capstone['culture_top_count']} conditions).", F["body_bold"], "#FFFFFF", p(720), gap=5)
    draw.text((p(1004), p(2726)), capstone["language_demo"].title(), font=F["serif_head"], fill=LIGHT["cyan"])
    text(draw, (p(1004), p(2778)), f"is the largest prompt-language homophily shift ({capstone['language_range']:.3f}); {capstone['language_topology']} varies most across languages ({capstone['language_topology_range']:.3f}).", F["body_bold"], "#FFFFFF", p(720), gap=5)
    draw.text((p(1004), p(2914)), "Model Family", font=F["serif_head"], fill=LIGHT["green"])
    text(draw, (p(1004), p(2966)), f"GPT-4.1 and GPT-4.1-mini are the closest pair in culture, language, and method studies; GPT-4.1-nano is farthest. Method still matters: density ranges from {capstone['method_lowest']} {capstone['method_lowest_density']:.3f} to {capstone['method_highest']} {capstone['method_highest_density']:.3f}.", F["body_bold"], "#FFFFFF", p(720), gap=5)
    rounded(draw, box(1830, 2388, 2596, 3076), 30, "#FFFDF9", outline=LIGHT["line"], width=1.3)
    draw.text((p(1860), p(2414)), "RQ RESULTS", font=F["section_big"], fill=LIGHT["ink"])
    text(
        draw,
        (p(1860), p(2470)),
        "These answers follow the final four-RQ framing in the repo. The new GPT-4.1 models do not simply dominate; they expose controlled sensitivity to culture, language, method, and model size.",
        F["small"],
        LIGHT["muted"],
        p(680),
        gap=5,
    )

    proof_cards = [
        (
            "RQ1",
            "Culture with language fixed",
            f"Culture changes homophily and topology; the largest culture shift is {capstone['culture_demo']} ({capstone['culture_range']:.3f}).",
            LIGHT["cyan"],
        ),
        (
            "RQ2",
            "Dominant demographic dimensions",
            "Political affiliation usually dominates tie formation; global often elevates age.",
            LIGHT["red"],
        ),
        (
            "RQ3",
            "Model consistency or divergence",
            "GPT-4.1 and GPT-4.1-mini are closest; GPT-4.1-nano is farthest.",
            LIGHT["green"],
        ),
        (
            "RQ4",
            "Prompt language with culture fixed",
            f"Changing only language shifts the network; religion moves most ({capstone['language_range']:.3f}).",
            LIGHT["gold"],
        ),
    ]
    for idx, (tag, question, answer, accent) in enumerate(proof_cards):
        yy = 2542 + idx * 126
        rounded(draw, box(1860, yy, 2568, yy + 104), 18, "#FFFFFF", outline=LIGHT["line"], width=1.1)
        draw.text((p(1882), p(yy + 16)), tag, font=F["label"], fill=accent)
        text(draw, (p(1950), p(yy + 14)), question, F["body_bold"], LIGHT["ink"], p(570), gap=3)
        text(draw, (p(1882), p(yy + 54)), answer, F["small"], LIGHT["muted"], p(640), gap=3)

    rounded(draw, box(104, 3170, 2596, 3496), 34, "#071729", outline="#071729", width=1.4)
    draw.text((p(154), p(3212)), "CORE QUESTION", font=F["section"], fill=LIGHT["gold"])
    text(
        draw,
        (p(154), p(3264)),
        "When the same 50-person roster is fixed, how do culture, prompt language, generation method, and model choice reshape LLM-generated friendship networks?",
        F["serif_quote"],
        "#FFFFFF",
        p(900),
        gap=7,
    )
    draw.line((p(1160), p(3228), p(1160), p(3448)), fill="#27486B", width=p(2))
    draw.text((p(1220), p(3212)), "ANSWER", font=F["section"], fill=LIGHT["cyan"])
    answer = [
        f"Culture changes structure: political affiliation has the largest culture-driven homophily shift ({capstone['culture_range']:.3f}).",
        f"Tie formation is usually political; across analyses, politics is top-ranked in {capstone['political_top_total']}/{capstone['dominance_total']} cases, while global often elevates age.",
        "Models are not interchangeable: GPT-4.1-mini stays closest to GPT-4.1, while GPT-4.1-nano diverges farthest.",
        f"Prompt language matters too: with culture fixed, religion shifts most ({capstone['language_range']:.3f}).",
    ]
    for idx, line in enumerate(answer):
        yy = p(3260 + idx * 48)
        draw.text((p(1220), yy), f"{idx + 1}", font=F["label"], fill=LIGHT["gold"])
        text(draw, (p(1264), yy - p(4)), line, F["body_bold"], "#FFFFFF", p(1200), gap=3)
    draw.text(
        (p(154), p(3450)),
        "University of Arizona Capstone | INFC 698 | LLM-generated friendship networks under controlled culture, language, method, and model-size changes",
        font=F["small"],
        fill="#DDEBFA",
    )
    return img


def poster_dark():
    personas = load_personas()
    graphs = load_graphs()
    metrics = derived_metrics(graphs, personas)
    findings = external_findings()

    img = gradient((W, H), DARK["bg0"], DARK["bg1"])
    img = add_noise(img, 18)
    draw = ImageDraw.Draw(img)

    # Ambient network background.
    random.seed(17)
    points = [(random.randint(p(120), W - p(120)), random.randint(p(250), H - p(250))) for _ in range(120)]
    for i, a in enumerate(points):
        for j in range(i + 1, min(len(points), i + 18)):
            b = points[j]
            dist = math.dist(a, b)
            if dist < p(390) and (i * 11 + j * 3) % 5 == 0:
                draw.line((a[0], a[1], b[0], b[1]), fill="#153D59", width=p(0.8))
    for i, pt in enumerate(points):
        c = DARK["cyan"] if i % 7 else DARK["red"]
        r = p(2.4 if i % 7 else 3.8)
        draw.ellipse((pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r), fill=c)

    draw.text((p(118), p(90)), "LLMs AS", font=F["display"], fill=DARK["ink"])
    draw.text((p(118), p(168)), "SOCIAL NETWORK", font=F["display_big"], fill=DARK["cyan"])
    draw.text((p(118), p(272)), "GENERATORS", font=F["display_big"], fill=DARK["ink"])
    text(
        draw,
        (p(124), p(410)),
        "A capstone study of whether GPT models can simulate friendship networks, and what hidden assumptions shape the ties they produce.",
        F["body_bold"],
        DARK["muted"],
        p(1180),
        gap=7,
    )
    draw.text((p(124), p(550)), "University of Arizona | INFC 698 Capstone | April 2026", font=F["small"], fill="#D6E5F5")

    # Hero claim.
    rounded(draw, box(124, 660, 2276, 990), 34, "#081524", outline=DARK["gold"], width=1.6)
    draw.text((p(170), p(710)), "THE STORY", font=F["section"], fill=DARK["gold"])
    text(
        draw,
        (p(170), p(765)),
        "The same 50-person roster can lead to different social worlds. Culture, language, model size, and prompting method all change who becomes connected.",
        F["headline"],
        DARK["ink"],
        p(1900),
        gap=8,
    )

    # Main fresh network panels.
    panel_rects = [
        box(124, 1080, 1160, 1845),
        box(1240, 1080, 2276, 1845),
        box(124, 1915, 1160, 2680),
        box(1240, 1915, 2276, 2680),
    ]
    for idx, ((name, graph), rect) in enumerate(zip(graphs.items(), panel_rects)):
        img = draw_mini_network(img, draw, graph, personas, rect, DARK, name, seed=30 + idx, dark=True)
        draw = ImageDraw.Draw(img)

    # Bottom cards.
    draw_pipeline(draw, box(124, 2795, 650, 3445), DARK, dark=True)
    draw_people_constellation(draw, personas, box(700, 2795, 1335, 3445), DARK, dark=True)
    draw_metric_card(
        draw,
        box(1385, 2795, 2276, 3015),
        "Finding 1",
        "Politics",
        "Political affiliation is the strongest repeated homophily signal across the project summaries.",
        DARK["red"],
        DARK,
        dark=True,
    )
    draw_metric_card(
        draw,
        box(1385, 3055, 2276, 3275),
        "Finding 2",
        "Models",
        f"{findings['closest_model']} is closest; Nano is the most divergent family member.",
        DARK["blue"],
        DARK,
        dark=True,
    )
    draw_metric_card(
        draw,
        box(1385, 3315, 2276, 3535),
        "Finding 3",
        "Language",
        f"Prompt language shifts {findings['language_demo']} homophily the most.",
        DARK["green"],
        DARK,
        dark=True,
    )
    draw.text((p(124), p(3548)), "Nodes are colored by political affiliation. All four network panels are freshly rendered from project .adj files, not reused report images.", font=F["tiny"], fill=DARK["muted"])
    return img


def poster_light():
    personas = load_personas()
    graphs = load_graphs()
    metrics = derived_metrics(graphs, personas)
    findings = external_findings()

    img = Image.new("RGB", (W, H), LIGHT["paper"])
    draw = ImageDraw.Draw(img)

    # Editorial blocks.
    draw.rectangle(box(0, 0, 2400, 570), fill=LIGHT["ink"])
    draw.rectangle(box(0, 512, 2400, 570), fill=LIGHT["red"])
    draw.text((p(112), p(86)), "CAN GPT MODELS", font=F["display"], fill="#FFFFFF")
    draw.text((p(112), p(170)), "GENERATE SOCIETY?", font=F["display_big"], fill=LIGHT["cyan"])
    text(
        draw,
        (p(116), p(322)),
        "Fresh visual poster for LLMs as Social Network Generators: 50 personas, four generation methods, and measured homophily/topology outcomes.",
        F["body_bold"],
        "#DDEBFA",
        p(1500),
        gap=7,
    )
    draw.text((p(116), p(455)), "Sai Hemanth Kilaru | Sriram Theerdh Manikyala | Raghav Upadhyay | Sri Sai Kumar Ramavath | Srivika Nunavathu", font=F["small"], fill="#FFFFFF")

    # Big central single network.
    central = parse_adj(TEXT / "sequential_gpt-4.1-mini_n5_culture_us_0.adj")
    img = draw_mini_network(img, draw, central, personas, box(112, 675, 1480, 1850), LIGHT, "A Generated Friendship World", seed=90, dark=False)
    draw = ImageDraw.Draw(img)

    rounded(draw, box(1530, 675, 2288, 1850), 32, LIGHT["card"], outline=LIGHT["line"], width=1.4)
    draw.text((p(1580), p(735)), "WHAT THIS PROJECT PROVES", font=F["section"], fill=LIGHT["red"])
    proof = [
        ("01", "LLMs can produce structured social graphs from persona rosters."),
        ("02", "The generated graph changes when the cultural frame changes."),
        ("03", "Prompt language is a real experimental variable, not just translation."),
        ("04", "New GPT models do not uniformly dominate; Nano diverges more."),
        ("05", "The strongest caution is bias sensitivity, especially politics."),
    ]
    y = p(815)
    for number, body in proof:
        draw.text((p(1580), y), number, font=F["metric2"], fill=LIGHT["cyan"])
        text(draw, (p(1665), y + p(3)), body, F["body_bold"], LIGHT["ink"], p(530), gap=4)
        y += p(175)

    # Four method strip.
    draw.text((p(112), p(1965)), "FOUR METHODS, FOUR DIFFERENT SOCIAL SHAPES", font=F["headline"], fill=LIGHT["ink"])
    mini_w, mini_h = 520, 445
    start_x, start_y = 112, 2048
    for idx, (name, graph) in enumerate(graphs.items()):
        x = start_x + (idx % 4) * (mini_w + 40)
        rect = box(x, start_y, x + mini_w, start_y + mini_h)
        img = draw_mini_network(img, draw, graph, personas, rect, LIGHT, name, seed=130 + idx, dark=False)
        draw = ImageDraw.Draw(img)

    # Data-driven chart cards.
    method_names = [name for name, _ in metrics]
    densities = [m["density"] for _, m in metrics]
    same_party = [m["same_party"] for _, m in metrics]
    draw_horizontal_bars(draw, box(112, 2565, 760, 3035), method_names, densities, "Fresh density from four .adj files", LIGHT["cyan"], LIGHT, dark=False, formatter=lambda v: f"{v:.3f}")
    draw_horizontal_bars(draw, box(840, 2565, 1488, 3035), method_names, same_party, "Within-party edge share", LIGHT["red"], LIGHT, dark=False, formatter=lambda v: f"{v:.0%}")
    draw_people_constellation(draw, personas, box(1568, 2565, 2288, 3035), LIGHT, dark=False)

    # Final takeaways.
    rounded(draw, box(112, 3135, 2288, 3468), 34, LIGHT["ink"], outline=LIGHT["ink"], width=1.4)
    draw.text((p(170), p(3190)), "BOTTOM LINE", font=F["section"], fill=LIGHT["gold"])
    text(
        draw,
        (p(170), p(3250)),
        "This project does not claim that LLMs perfectly recreate human society. It shows that they can generate analyzable synthetic networks, and that their outputs must be audited because identity, prompt context, and model family shape the final graph.",
        F["headline"],
        "#FFFFFF",
        p(1960),
        gap=7,
    )
    draw.text((p(116), p(3525)), f"Verified signals: culture LCC range {findings['lcc_range']:.3f}; closest model pair {findings['closest_model']} ({findings['closest_dist']:.3f}); farthest {findings['farthest_model']} ({findings['farthest_dist']:.3f}).", font=F["small"], fill=LIGHT["muted"])
    return img


def draw_glow_network(img, draw, graph, personas, rect, seed=202):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 42, "#071729", outline="#2FE6FF", width=1.6)
    draw.rectangle((x0, y0, x1, y0 + p(122)), fill="#0D2740")
    draw.text((x0 + p(34), y0 + p(28)), "ONE ROSTER, MANY POSSIBLE SOCIETIES", font=F["section"], fill="#FFFFFF")
    draw.text((x0 + p(34), y0 + p(78)), "fresh network render from project .adj output", font=F["small"], fill="#AFC5D8")

    inner = (x0 + p(50), y0 + p(156), x1 - p(50), y1 - p(52))
    pos = nx.spring_layout(graph, seed=seed, k=0.54, iterations=220)
    pos = normalize_positions(pos, inner, pad=p(26))

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for a, b in graph.edges():
        ax, ay = pos[a]
        bx, by = pos[b]
        same = personas[a].get("political affiliation") == personas[b].get("political affiliation")
        color = rgb("#2FE6FF" if same else "#FF4D6D")
        gd.line((ax, ay, bx, by), fill=(*color, 70), width=p(1.2))
    glow = glow.filter(ImageFilter.GaussianBlur(p(1.2)))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    draw = ImageDraw.Draw(img)

    for a, b in graph.edges():
        ax, ay = pos[a]
        bx, by = pos[b]
        same = personas[a].get("political affiliation") == personas[b].get("political affiliation")
        draw.line((ax, ay, bx, by), fill="#23CFEA" if same else "#D83A59", width=p(1.0))
    for node, (cx, cy) in pos.items():
        party = personas[node].get("political affiliation")
        fill = POLITICS.get(party, "#FFD166")
        age = int(personas[node].get("age", 35))
        r = p(8 + min(age, 70) / 16)
        draw.ellipse((cx - r - p(4), cy - r - p(4), cx + r + p(4), cy + r + p(4)), fill="#FFFFFF")
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline="#0B1828", width=p(1))

    stats = graph_stats(graph, personas)
    footer = f"{stats['edges']} ties | density {stats['density']:.3f} | same-party edge share {stats['same_party']:.0%}"
    draw.text((x0 + p(34), y1 - p(44)), footer, font=F["small"], fill="#B9C7D8")
    return img


def draw_compact_model_coverage(draw, rect, capstone):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 30, "#FFFFFF", outline="#D7DFE8", width=1.2)
    draw.text((x0 + p(26), y0 + p(24)), "MODEL COVERAGE", font=F["section"], fill=LIGHT["ink"])
    draw.text((x0 + p(26), y0 + p(70)), "Base repo context and our extension are kept separate.", font=F["small"], fill=LIGHT["muted"])

    draw.text((x0 + p(26), y0 + p(126)), "BASE SNAP REPO", font=F["label"], fill=LIGHT["blue"])
    chip_x, chip_y = x0 + p(26), y0 + p(162)
    for label, fill in [
        ("gpt-3.5-turbo", "#223B58"),
        ("gpt-4o", "#1B6CA8"),
        ("llama3.1-8b", "#3B5B7A"),
        ("llama3.1-70b", "#58799C"),
        ("gemma2-9b", "#5F7F91"),
        ("gemma2-27b", "#82989E"),
    ]:
        chip_x = draw_chip(draw, (chip_x, chip_y), label, fill, ink="#FFFFFF")
        if chip_x > x1 - p(210):
            chip_x = x0 + p(26)
            chip_y += p(44)

    draw.line((x0 + p(26), y0 + p(282), x1 - p(26), y0 + p(282)), fill="#E4EAF1", width=p(2))
    draw.text((x0 + p(26), y0 + p(318)), "OUR GPT-4.1 STUDY", font=F["label"], fill=LIGHT["red"])
    chip_x = x0 + p(26)
    for label, fill, ink in [
        ("gpt-4.1", LIGHT["red"], "#FFFFFF"),
        ("gpt-4.1-mini", LIGHT["cyan"], "#FFFFFF"),
        ("gpt-4.1-nano", LIGHT["gold"], LIGHT["ink"]),
    ]:
        chip_x = draw_chip(draw, (chip_x, y0 + p(356)), label, fill, ink=ink)

    text(draw, (x0 + p(26), y0 + p(430)), f"{capstone['capstone_total']} verified generated graphs: culture {capstone['culture_verified']}, method {capstone['method_verified']}, language {capstone['language_verified']}.", F["body_bold"], LIGHT["ink"], x1 - x0 - p(52), gap=5)


def draw_research_question_card(draw, rect):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 30, "#0F1B2E", outline="#0F1B2E", width=1)
    draw.text((x0 + p(28), y0 + p(28)), "RESEARCH QUESTION", font=F["label"], fill=LIGHT["gold"])
    text(
        draw,
        (x0 + p(28), y0 + p(76)),
        "If the people stay fixed, what actually changes the social network: culture, language, generation method, or model size?",
        F["serif_quote"],
        "#FFFFFF",
        x1 - x0 - p(56),
        gap=7,
    )
    draw.text((x0 + p(28), y1 - p(66)), "Design: fixed 50-person roster -> controlled prompt/model sweeps -> homophily + topology metrics", font=F["small"], fill="#B9C7D8")


def draw_signal_card(draw, rect, number, title, subtitle, accent):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 28, "#FFFFFF", outline=accent, width=1.4)
    draw.text((x0 + p(22), y0 + p(20)), number, font=F["metric2"], fill=accent)
    num_w, _ = draw_text_size(draw, number, F["metric2"])
    title_x = min(x0 + p(250), x0 + p(22) + num_w + p(30))
    draw.text((title_x, y0 + p(28)), title, font=F["section"], fill=LIGHT["ink"])
    text(draw, (title_x, y0 + p(78)), subtitle, F["body_bold"], LIGHT["muted"], x1 - title_x - p(34), gap=5)


def draw_method_tiles(draw, rect, capstone):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 30, "#FFFFFF", outline=LIGHT["line"], width=1.2)
    draw.text((x0 + p(28), y0 + p(24)), "WHY THE FOUR METHODS ARE DIFFERENT", font=F["section"], fill=LIGHT["ink"])
    draw.text((x0 + p(28), y0 + p(68)), "Same roster, different information flow into the model.", font=F["small"], fill=LIGHT["muted"])
    methods = [
        ("Global", "one prompt sees all people", LIGHT["blue"], "global"),
        ("Local", "each person chooses alone", LIGHT["red"], "local"),
        ("Sequential", "later people see graph state", LIGHT["gold"], "sequential"),
        ("Iterative", "graph is repeatedly revised", LIGHT["green"], "iterative"),
    ]
    density = dict(capstone["four_method_density"])
    tile_w = int((x1 - x0 - p(80)) / 4)
    for idx, (name, desc, accent, mode) in enumerate(methods):
        tx0 = x0 + p(28) + idx * (tile_w + p(8))
        ty0 = y0 + p(124)
        tx1 = tx0 + tile_w
        ty1 = y1 - p(28)
        rounded(draw, (tx0, ty0, tx1, ty1), 22, "#F8FAFD", outline=accent, width=1.2)
        draw.text((tx0 + p(18), ty0 + p(18)), name, font=F["label"], fill=accent)
        text(draw, (tx0 + p(18), ty0 + p(52)), desc, F["small"], LIGHT["ink"], tile_w - p(36), gap=3)
        panel = (tx0 + p(18), ty0 + p(128), tx1 - p(18), ty0 + p(272))
        rounded(draw, panel, 16, "#FFFFFF", outline="#E4EAF1", width=1)
        if mode == "global":
            for row in range(2):
                for col in range(3):
                    cx = panel[0] + p(28) + col * p(26)
                    cy = panel[1] + p(36) + row * p(28)
                    draw.ellipse((cx - p(6), cy - p(6), cx + p(6), cy + p(6)), fill=accent)
            rounded(draw, (panel[0] + p(126), panel[1] + p(34), panel[0] + p(202), panel[1] + p(86)), 12, accent)
            draw.text((panel[0] + p(148), panel[1] + p(50)), "LLM", font=F["tiny"], fill="#FFFFFF")
            draw_arrow(draw, (panel[0] + p(92), panel[1] + p(60)), (panel[0] + p(126), panel[1] + p(60)), accent, width=2)
        elif mode == "local":
            for i in range(3):
                cy = panel[1] + p(28) + i * p(32)
                draw.ellipse((panel[0] + p(20), cy, panel[0] + p(40), cy + p(20)), fill=accent)
                draw_arrow(draw, (panel[0] + p(50), cy + p(10)), (panel[0] + p(116), cy + p(10)), accent, width=2)
                for j in range(3):
                    cx = panel[0] + p(136) + j * p(22)
                    draw.ellipse((cx, cy + p(2), cx + p(14), cy + p(16)), fill="#8EDCF0" if j != i % 3 else LIGHT["red"])
        elif mode == "sequential":
            for i in range(4):
                cx = panel[0] + p(32) + i * p(46)
                cy = panel[1] + p(64)
                draw.ellipse((cx - p(14), cy - p(14), cx + p(14), cy + p(14)), fill=accent)
                draw.text((cx - p(5), cy - p(9)), str(i + 1), font=F["tiny"], fill="#FFFFFF")
                if i < 3:
                    draw_arrow(draw, (cx + p(16), cy), (cx + p(32), cy), accent, width=2)
        else:
            draw_network_icon(draw, (panel[0] + p(58), panel[1] + p(24), panel[0] + p(178), panel[1] + p(120)), accent, LIGHT["ink"])
            draw.arc((panel[0] + p(42), panel[1] + p(12), panel[0] + p(194), panel[1] + p(132)), start=210, end=20, fill=accent, width=p(3))
            draw.text((panel[0] + p(184), panel[1] + p(48)), "+/-", font=F["label"], fill=accent)
        draw.text((tx0 + p(18), ty1 - p(44)), f"mean density {density[name]:.3f}", font=F["small"], fill=LIGHT["muted"])


def draw_verification_strip(draw, rect, capstone):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, 34, "#071729", outline="#071729", width=1)
    draw.text((x0 + p(34), y0 + p(30)), "THE 30-SECOND MESSAGE", font=F["section"], fill=LIGHT["gold"])
    columns = [
        ("1", f"One roster became many societies: {capstone['capstone_total']} verified graphs."),
        ("2", f"Politics dominated homophily: {capstone['political_top_total']}/{capstone['dominance_total']} top-rank cases."),
        ("3", f"Language and scale mattered: religion shifted {capstone['language_range']:.3f}; nano diverged."),
    ]
    col_w = int((x1 - x0 - p(92)) / 3)
    for idx, (num, line) in enumerate(columns):
        cx = x0 + p(34) + idx * (col_w + p(12))
        cy = y0 + p(88)
        draw.text((cx, cy), f"{num}.", font=F["section"], fill=LIGHT["cyan"])
        text(draw, (cx + p(54), cy + p(2)), line, F["body_bold"], "#FFFFFF", col_w - p(68), gap=5)


def poster_competition():
    personas = load_personas()
    capstone = verified_capstone_data()
    legacy = legacy_repo_data()
    graphs = load_graphs()

    img = Image.new("RGB", (W, H), "#F6EFE5")
    draw = ImageDraw.Draw(img)

    # Tall editorial background with a central data column.
    draw.rectangle(box(0, 0, 2400, 1020), fill="#071729")
    draw.polygon([box(0, 890, 2400, 1020)[0:2], (W, p(760)), (W, p(1020)), (0, p(1020))], fill="#AB0520")
    random.seed(55)
    for _ in range(135):
        x = random.randint(p(40), W - p(40))
        y = random.randint(p(40), p(920))
        c = "#2FE6FF" if random.random() > 0.18 else "#FFD166"
        r = p(random.choice([2, 2.5, 3.2]))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=c)
    for _ in range(70):
        x0 = random.randint(p(40), W - p(40))
        y0 = random.randint(p(40), p(920))
        x1 = x0 + random.randint(-p(220), p(220))
        y1 = y0 + random.randint(-p(130), p(130))
        draw.line((x0, y0, x1, y1), fill="#143B59", width=p(1))

    draw.text((p(96), p(74)), "LLMs AS", font=F["display"], fill="#FFFFFF")
    draw.text((p(96), p(166)), "SOCIAL NETWORK", font=F["display_big"], fill="#FFFFFF")
    draw.text((p(96), p(276)), "GENERATORS", font=F["display_big"], fill="#2FE6FF")
    draw.text((p(100), p(400)), "Same people. Different prompts. Different societies.", font=F["serif_head"], fill="#FFD166")
    text(
        draw,
        (p(100), p(468)),
        "We extend the original SNAP repository from baseline model comparison into a controlled GPT-4.1-family sensitivity study.",
        F["body_bold"],
        "#DDEBFA",
        p(1120),
        gap=6,
    )
    draw.text((p(100), p(635)), "Sai Hemanth Kilaru | Sriram Theerdh Manikyala | Raghav Upadhyay | Sri Sai Kumar Ramavath | Srivika Nunavathu", font=F["small"], fill="#FFFFFF")
    draw.text((p(100), p(676)), "University of Arizona | INFC 698 Capstone | Instructor: Dr. Dalal Alharthi | April 2026", font=F["small"], fill="#DDEBFA")

    rounded(draw, box(96, 742, 1240, 904), 24, "#0E2A43", outline="#2FE6FF", width=1.2)
    draw.text((p(128), p(776)), "Fixed roster", font=F["label"], fill="#FFD166")
    draw.text((p(314), p(776)), "50 personas", font=F["label"], fill="#FFFFFF")
    draw.text((p(128), p(830)), "Controlled outputs", font=F["label"], fill="#FFD166")
    draw.text((p(370), p(830)), "192 verified generated graphs", font=F["label"], fill="#FFFFFF")

    hero_graph = parse_adj(TEXT / "sequential_gpt-4.1-mini_n5_culture_us_0.adj")
    img = draw_glow_network(img, draw, hero_graph, personas, box(1370, 84, 2296, 890), seed=161)
    draw = ImageDraw.Draw(img)

    draw_research_question_card(draw, box(104, 1100, 914, 1474))
    draw_compact_model_coverage(draw, box(954, 1100, 2296, 1474), capstone)

    draw.text((p(104), p(1580)), "THE RESULT STORY", font=F["headline"], fill=LIGHT["ink"])
    draw.text((p(104), p(1632)), "The newer GPT-4.1 models are not judged by making denser graphs. They are useful because the controlled study reveals which factors steer the network.", font=F["small"], fill=LIGHT["muted"])
    claim_y = 1700
    draw_signal_card(draw, box(104, claim_y, 790, claim_y + 210), f"{capstone['political_top_total']}/{capstone['dominance_total']}", "Politics dominates", "Political affiliation is the strongest recurring top-ranked homophily signal across the final analyses.", LIGHT["red"])
    draw_signal_card(draw, box(858, claim_y, 1544, claim_y + 210), f"{capstone['language_range']:.3f}", "Language matters", "Changing prompt language shifts religion homophily most strongly while the roster stays fixed.", LIGHT["cyan"])
    draw_signal_card(draw, box(1610, claim_y, 2296, claim_y + 210), "0.086", "Mini tracks full", "GPT-4.1 and GPT-4.1-mini are closest in the culture study; nano diverges farther.", LIGHT["green"])

    rounded(draw, box(104, 1990, 2296, 2356), 30, "#FFFFFF", outline=LIGHT["line"], width=1.2)
    draw.text((p(132), p(2024)), "CONTROLLED EXPERIMENT MATRIX", font=F["section"], fill=LIGHT["ink"])
    stages = [
        ("Step 2", capstone["culture_verified"], "culture", "Brazil, India, Japan, US", LIGHT["blue"]),
        ("Step 3", capstone["method_verified"], "method", "global, local, sequential, iterative", LIGHT["red"]),
        ("Step 4", capstone["language_verified"], "language", "English, Hindi, Japanese, Spanish", LIGHT["green"]),
    ]
    sx = 132
    for idx, (step, verified, title, desc, accent) in enumerate(stages):
        x0 = p(sx + idx * 724)
        y0 = p(2092)
        rounded(draw, (x0, y0, x0 + p(650), y0 + p(206)), 22, "#F8FAFD", outline=accent, width=1.3)
        draw.text((x0 + p(22), y0 + p(20)), step.upper(), font=F["tiny"], fill=accent)
        draw.text((x0 + p(22), y0 + p(54)), verified, font=F["metric2"], fill=accent)
        draw.text((x0 + p(214), y0 + p(52)), title.upper(), font=F["label"], fill=LIGHT["ink"])
        text(draw, (x0 + p(214), y0 + p(92)), desc, F["body_bold"], LIGHT["muted"], p(390), gap=5)

    draw_method_tiles(draw, box(104, 2440, 2296, 2918), capstone)

    rounded(draw, box(104, 3002, 1120, 3338), 30, "#FFFFFF", outline=LIGHT["line"], width=1.2)
    draw.text((p(132), p(3032)), "NUMBERS TO DEFEND", font=F["section"], fill=LIGHT["ink"])
    evidence = [
        ("Culture", f"{capstone['culture_demo']} homophily range {capstone['culture_range']:.3f}; topology spread in {capstone['culture_topology']} is {capstone['culture_topology_range']:.3f}."),
        ("Language", f"{capstone['language_demo']} homophily range {capstone['language_range']:.3f}; {capstone['language_top_demo']} top-ranked in {capstone['language_top_count']} conditions."),
        ("Model", f"Closest pair: {capstone['closest_model']} ({capstone['closest_dist']:.3f}); farthest pair: {capstone['farthest_model']} ({capstone['farthest_dist']:.3f})."),
    ]
    yy = 3094
    for label, body in evidence:
        draw.text((p(132), p(yy)), label, font=F["label"], fill=LIGHT["red"])
        text(draw, (p(260), p(yy - 2)), body, F["body_bold"], LIGHT["ink"], p(800), gap=4)
        yy += 74

    rounded(draw, box(1160, 3002, 2296, 3338), 30, "#FFFFFF", outline=LIGHT["line"], width=1.2)
    draw.text((p(1190), p(3032)), "WHO IS IN THE ROSTER?", font=F["section"], fill=LIGHT["ink"])
    groups = demographic_breakdown(personas)
    tiny_items = [
        ("Gender", groups["Gender"], LIGHT["cyan"]),
        ("Politics", groups["Political"], LIGHT["red"]),
        ("Religion", groups["Religion"], LIGHT["green"]),
    ]
    cx = 1190
    for title, counter, accent in tiny_items:
        draw.text((p(cx), p(3092)), title, font=F["label"], fill=accent)
        top = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:3]
        y = 3132
        for name, value in top:
            draw.text((p(cx), p(y)), f"{name}: {value}", font=F["small"], fill=LIGHT["ink"])
            y += 34
        cx += 350

    draw_verification_strip(draw, box(104, 3418, 2296, 3578), capstone)
    return img


def save(name: str, img: Image.Image):
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    img.save(png, dpi=(300, 300), optimize=True)
    img.save(pdf, "PDF", resolution=300.0)
    return png, pdf


def main():
    # Remove poster variants the user explicitly rejected.
    for old_name in [
        "llm_networks_award_dark_poster.png",
        "llm_networks_award_dark_poster.pdf",
        "llm_networks_academic_poster.png",
        "llm_networks_academic_poster.pdf",
        "llm_networks_visual_poster.png",
        "llm_networks_visual_poster.pdf",
    ]:
        (OUT / old_name).unlink(missing_ok=True)

    outputs = []
    outputs.extend(save("llm_networks_editorial_light_poster", poster_light()))
    outputs.extend(save("llm_networks_editorial_story_poster", poster_story()))
    for path in outputs:
        if path.suffix.lower() == ".png":
            im = Image.open(path)
            print(f"Wrote {path} {im.size[0]}x{im.size[1]}")
        else:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
