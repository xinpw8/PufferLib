from pathlib import Path
from aggregate_chess_stats import aggregate_stats, guess_user_id


def build_mermaid():
    stats = aggregate_stats(guess_user_id())
    per = stats["per_user_stats"]
    # Sort by games played
    top = sorted(per.items(), key=lambda x: x[1]["games"], reverse=True)[:20]
    total_games_all = sum(v["games"] for v in per.values())

    lines = ["graph LR", "  classDef user fill:#e8f4ff,stroke:#2680eb,stroke-width:1px;"]
    prev = None
    for idx, (user, s) in enumerate(top, 1):
        games = s["games"]
        w, l, t = s["win"], s["loss"], s["tie"]
        winpct = (w / (w + l)) * 100 if (w + l) else 0
        node_name = f"u{idx}"
        label = f"{user}\\nG:{games} W:{w} L:{l} Win%:{winpct:.1f}%"
        lines.append(f"  {node_name}[\"{label}\"]::user")
        if prev:
            lines.append(f"  {prev} --> {node_name}")
        prev = node_name

    # Lump others
    other_games = total_games_all - sum(s["games"] for _, s in top)
    lines.append(f"  others[\"Other Users\\nG:{other_games}\"]::user")
    if prev:
        lines.append(f"  {prev} --> others")

    return "\n".join(lines)


if __name__ == "__main__":
    print(build_mermaid()) 