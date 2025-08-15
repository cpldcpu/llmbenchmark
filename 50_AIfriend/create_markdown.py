#!/usr/bin/env python3
"""Generate a Markdown summary from output_queries.json.

Usage:
  python create_markdown.py               # reads output_queries.json -> output_queries.md
  python create_markdown.py -i in.json -o out.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import re
import datetime
import random


def sanitize_cell(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    # Replace pipe to avoid breaking markdown tables
    s = s.replace("|", "\\|")
    # Convert multiple lines to HTML <br> so cells render reasonably in GH-flavored Markdown
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n\n", "<br><br>")
    s = s.replace("\n", "<br>")
    return s


def sanitize_blockquote(text: Any) -> str:
    """Return text formatted as a Markdown blockquote (lines prefixed with '> ')."""
    if text is None:
        return ""
    s = str(text)
    # ensure CRLF -> LF
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")
    return "\n".join(
        "> " + line if line.strip() != "" else ">" for line in lines
    )


def join_list_field(field: Any) -> str:
    # field is often a list of strings; join with visual separator if multiple
    if field is None:
        return ""
    if isinstance(field, list):
        # Randomly select at most 2 items from the list
        if len(field) <= 2:
            selected_items = field
        else:
            selected_items = random.sample(field, 2)
        return "\n\n---\n\n".join(str(x) for x in selected_items)
    return str(field)


def generate_markdown(data: Dict[str, Any]) -> str:
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise ValueError("input JSON must contain a top-level 'results' list")

    # Group entries by prompt_id preserving order of first appearance
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in results:
        pid = entry.get("prompt_id", "<no_prompt_id>")
        groups.setdefault(pid, []).append(entry)

    parts: List[str] = []
    parts.append("# Output queries\n")

    # Table of contents / index
    parts.append("## Index\n")
    for pid in groups.keys():
        parts.append(f"- [{pid}](#{pid})")
    parts.append("")

    for prompt_id, entries in groups.items():
        # explicit anchor so links work reliably
        parts.append(f"<a id=\"{prompt_id}\"></a>")
        parts.append(f"## {prompt_id}\n")

        # Show the prompt as a Markdown blockquote before the tables
        prompt_text = entries[0].get("prompt", "")
        if prompt_text:
            parts.append(sanitize_blockquote(prompt_text))
            parts.append("")

        # First table: LLM | Output (omit Thinking to save space)
        parts.append("| LLM | Output |")
        parts.append("|---|---|")

        def sort_key(entry: Dict[str, Any]) -> datetime.datetime:
            # Try to extract a YYYY-MM-DD from the model/llm string (take last match)
            model_str = str(entry.get("llm") or entry.get("provider") or "")
            matches = re.findall(r"\d{4}-\d{2}-\d{2}", model_str)
            if matches:
                date_str = matches[-1]
                try:
                    return datetime.datetime.strptime(date_str, "%Y-%m-%d")
                except Exception:
                    pass
            # fallback to timestamp field if present
            ts = entry.get("timestamp")
            if ts:
                try:
                    return datetime.datetime.fromisoformat(ts)
                except Exception:
                    try:
                        # sometimes timestamp may have Z
                        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        pass
            # final fallback: minimal date so it sorts first
            return datetime.datetime.min

        sorted_entries = sorted(entries, key=sort_key)
        for e in sorted_entries:
            llm = sanitize_cell(e.get("llm") or e.get("provider") or "")
            out = join_list_field(e.get("output"))
            out = sanitize_cell(out)
            parts.append(f"| {llm} | {out} |")
        parts.append("\n---\n")

    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description="Create a markdown summary from output_queries.json")
    p.add_argument("-i", "--input", default="output_queries.json", help="input JSON file")
    p.add_argument("-o", "--output", default="output_queries.md", help="output Markdown file")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    data = json.loads(inp.read_text(encoding="utf-8"))
    md = generate_markdown(data)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
