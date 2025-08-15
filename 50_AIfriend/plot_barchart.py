import argparse
import json
import os
import re
from datetime import datetime

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})$")


def load_summary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_group(data: Dict, group_name: Optional[str] = None) -> Dict:
    if group_name:
        return data.get(group_name, {})
    if len(data) == 1:
        return next(iter(data.values()))
    return next(iter(data.values()))


def extract_date_from_name(name: str):
    m = DATE_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    except Exception:
        return None


def sort_llms(llm_dict: Dict) -> List[str]:
    def key(item):
        name = item[0]
        date = extract_date_from_name(name)
        return (date or datetime(1, 1, 1).date(), name)

    return [name for name, _ in sorted(llm_dict.items(), key=key)]


def collect_criteria(llm_dict: Dict, llm_order: List[str]) -> Dict[str, List[float]]:
    criteria = set()
    for v in llm_dict.values():
        criteria.update(v.get("criteria_stats", {}).keys())

    data = {}
    for c in sorted(criteria):
        vals = [llm_dict.get(llm, {}).get("criteria_stats", {}).get(c, 0.0) for llm in llm_order]
        data[c] = vals
    return data


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "_", s)[:200]


def get_provider(name: str) -> str:
    n = name.lower()
    if n.startswith("claude") or "claude-" in n:
        return "Anthropic"
    if n.startswith("openai/") or n.startswith("gpt") or "o4-mini" in n:
        return "OpenAI"
    return "Other"


COLOR_MAP = {"Anthropic": "#d4a37f", "OpenAI": "#1f77b4", "Other": "#888888"}


def _add_watermark():
    fig = plt.gcf()
    fig.text(0.99, 0.01, "https://github.com/cpldcpu/llmbenchmark",
             ha="right", va="bottom", fontsize=8, color="#888888", alpha=0.7)


def plot_and_save(criteria_data: Dict[str, List[float]], llm_order: List[str], out_dir: str, prefix: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    providers = [get_provider(n) for n in llm_order]
    colors_per_llm = [COLOR_MAP.get(p, "#888888") for p in providers]

    for criterion, values in criteria_data.items():
        try:
            plt.figure(figsize=(max(6, len(llm_order) * 0.6), 4.5))
            bars = plt.bar(range(len(llm_order)), values, color=colors_per_llm)
            plt.xticks(range(len(llm_order)), llm_order, rotation=45, ha="right")

            max_val = max(values) if values else 1.0
            top = max(1.0, max_val)
            margin = max(0.06, 0.12 * top)
            ymax = top + margin
            plt.ylim(0, ymax)
            plt.ylabel("stat")
            plt.title(criterion)

            ann_offset = 0.02 * (top + margin)
            ann_top_limit = ymax - 0.01 * (top + margin)
            for bar, val in zip(bars, values):
                ann_y = min(val + ann_offset, ann_top_limit)
                plt.text(bar.get_x() + bar.get_width() / 2, ann_y, f"{val:.2f}", ha='center', va='bottom', fontsize=8)

            unique_providers = []
            for p in providers:
                if p not in unique_providers:
                    unique_providers.append(p)
            legend_handles = [Patch(color=COLOR_MAP.get(p, "#888888"), label=p) for p in unique_providers]
            if legend_handles:
                plt.legend(handles=legend_handles, loc='upper left')

            fname = sanitize_filename((prefix + "__" if prefix else "") + criterion) + ".png"
            path = os.path.join(out_dir, fname)
            _add_watermark()
            plt.tight_layout()
            plt.savefig(path, dpi=200)
            plt.close()
            print(f"Wrote {path}")
        except Exception:
            import traceback
            print(f"Error plotting bar for criterion {criterion}")
            traceback.print_exc()


def plot_scatter_by_provider(criteria_data: Dict[str, List[float]], llm_dict: Dict, llm_order: List[str], out_dir: str, prefix: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    for criterion in criteria_data.keys():
        try:
            points = []
            for llm in llm_order:
                stat = llm_dict.get(llm, {}).get("criteria_stats", {}).get(criterion, None)
                date = extract_date_from_name(llm)
                provider = get_provider(llm)
                if stat is None or date is None:
                    continue
                points.append((date, stat, provider, llm))

            if not points:
                continue

            grouped = {}
            for date, stat, provider, llm in points:
                grouped.setdefault(provider, []).append((date, stat, llm))

            plt.figure(figsize=(8, 4.5))
            ax = plt.gca()
            for provider, items in grouped.items():
                items.sort(key=lambda x: x[0])
                dates = [d for d, s, l in items]
                stats = [s for d, s, l in items]
                ax.plot(dates, stats, marker='o', markersize=10, linewidth=2, color=COLOR_MAP.get(provider, '#888888'), label=provider)

            ax.set_ylim(0, 1.05)
            ax.set_ylabel('stat')
            ax.set_xlabel('date')
            ax.set_title(criterion + ' — scatter by provider')
            ax.legend(loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45, ha='right')
            _add_watermark()
            fname = sanitize_filename((prefix + "__" if prefix else "") + criterion) + "_scatter.png"
            path = os.path.join(out_dir, fname)
            plt.tight_layout()
            plt.savefig(path, dpi=200)
            plt.close()
            print(f"Wrote {path}")
        except Exception:
            import traceback
            print(f"Error plotting scatter for criterion {criterion}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Plot bar charts and scatter plots per criterion from evaluation summary JSON.")
    parser.add_argument("input", help="Path to evaluation_summary.json")
    parser.add_argument("--group", help="Top-level group to use (e.g. 'friend')", default=None)
    parser.add_argument("--out", help="Output directory for plots", default=None)
    args = parser.parse_args()

    summary = load_summary(args.input)
    llm_dict = extract_group(summary, args.group)
    if not llm_dict:
        print("No LLM entries found in the provided group/file.")
        return

    llm_order = sort_llms(llm_dict)
    criteria_data = collect_criteria(llm_dict, llm_order)

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.input)), "plots")
    prefix = args.group if args.group else (next(iter(summary.keys())) if len(summary) == 1 else None)

    plot_and_save(criteria_data, llm_order, out_dir, prefix=prefix)
    plot_scatter_by_provider(criteria_data, llm_dict, llm_order, out_dir, prefix=prefix)


if __name__ == "__main__":
    main()
