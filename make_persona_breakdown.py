import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


INPUT_PATH = Path("text-files/us_50_gpt4o_w_interests.json")
OUTPUT_DIR = Path("plots/persona_breakdown")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_personas(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_categories(personas, key):
    values = [personas[pid][key] for pid in personas]
    return Counter(values)


def plot_categorical(ax, counts, title):
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [label for label, _ in items]
    values = [value for _, value in items]
    bars = ax.barh(labels, values, color=sns.color_palette("pastel", len(labels)))
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_xlim(0, max(values) + 2)
    ax.tick_params(axis="y", labelsize=11)
    for bar, value in zip(bars, values):
        ax.text(value + 0.1, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9)


def plot_age(ax, personas):
    ages = [int(personas[pid]["age"]) for pid in personas]
    bins = list(range(0, 100, 10)) + [200]
    age_bins = pd.cut(ages, bins=bins, right=False, include_lowest=True)
    counts = Counter(age_bins)
    ordered_labels = [f"[{b}, {b+10})" for b in range(0, 90, 10)] + ["[90, 200)"]
    values = [counts.get(pd.Interval(left=b, right=b + 10, closed='left'), 0) for b in range(0, 90, 10)] + [counts.get(pd.Interval(left=90, right=200, closed='left'), 0)]
    ax.barh(ordered_labels, values, color="#8da0cb")
    ax.set_title("Age distribution")
    ax.set_xlabel("Count")
    ax.set_ylabel("Age bin")
    ax.set_xlim(0, max(values) + 2)
    ax.tick_params(axis="y", labelsize=10)
    for y, value in enumerate(values):
        ax.text(value + 0.1, y, str(value), va="center", fontsize=9)


def main():
    personas = load_personas(INPUT_PATH)

    breakdowns = {
        "Gender": count_categories(personas, "gender"),
        "Race / ethnicity": count_categories(personas, "race/ethnicity"),
        "Religion": count_categories(personas, "religion"),
        "Political affiliation": count_categories(personas, "political affiliation"),
    }

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plot_categorical(axes[0], breakdowns["Gender"], "Gender")
    plot_age(axes[1], personas)
    plot_categorical(axes[2], breakdowns["Race / ethnicity"], "Race / ethnicity")
    plot_categorical(axes[3], breakdowns["Religion"], "Religion")
    plot_categorical(axes[4], breakdowns["Political affiliation"], "Political affiliation")

    axes[5].axis("off")
    axes[5].text(
        0.0,
        0.8,
        "Roster size: 50 personas\n\nEach panel is computed directly from\ntext-files/us_50_gpt4o_w_interests.json.\n\nAge is shown as 10-year bins for readability.",
        fontsize=14,
        va="top",
    )

    fig.suptitle("Demographic breakdown of the 50-person roster", fontsize=22, y=1.02)
    plt.tight_layout()

    out_png = OUTPUT_DIR / "persona_breakdown.png"
    out_csv = OUTPUT_DIR / "persona_breakdown_counts.csv"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")

    rows = []
    for dim, counts in breakdowns.items():
        for category, count in counts.items():
            rows.append({"dimension": dim, "category": category, "count": count})
    age_bins = pd.cut([int(personas[pid]["age"]) for pid in personas], bins=list(range(0, 100, 10)) + [200], right=False, include_lowest=True)
    for interval, count in Counter(age_bins).items():
        rows.append({"dimension": "Age", "category": str(interval), "count": count})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved chart to {out_png}")
    print(f"Saved counts to {out_csv}")


if __name__ == "__main__":
    main()
