import re
import json
from pathlib import Path
from collections import Counter, defaultdict
import argparse

EMAIL_DIR = Path(__file__).with_name("chess_emails")

HEADER_PATTERN = re.compile(r"^\[(\w+) \"(.*)\"\]")
MOVE_NUMBER_PATTERN = re.compile(r"(\d+)\.")


def parse_email(path: Path):
    """Parse PGN headers and moves from an email text file."""
    headers = {}
    moves_lines = []
    started_headers = False
    in_headers = False  # we will switch to True when we hit first '['
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("["):
                # Header line
                in_headers = True
                started_headers = True
                m = HEADER_PATTERN.match(line)
                if m:
                    tag, value = m.groups()
                    headers[tag] = value
                continue

            if in_headers:
                # Inside headers but current line not starting with '['
                if line.strip() == "":
                    # Blank line ends header section
                    in_headers = False
                continue  # skip any other lines inside headers (shouldn't happen)

            # After headers -> moves (skip blank lines)
            if started_headers and line.strip() != "":
                moves_lines.append(line)
    moves_text = " ".join(moves_lines)

    # Remove comments {...}
    moves_no_comments = re.sub(r"\{[^}]*\}", "", moves_text)
    # Remove result token at end (e.g., 1-0)
    moves_no_comments = re.sub(r"\s(1-0|0-1|1/2-1/2)\s*$", "", moves_no_comments.strip())
    # Remove move numbers like "12." or "12..."
    moves_clean = re.sub(r"\d+\.\.\.\s?|\d+\.\s?", "", moves_no_comments)
    tokens = moves_clean.split()

    half_moves = len(tokens)

    return headers, half_moves


def aggregate_stats(user_id: str):
    paths = sorted(EMAIL_DIR.glob("*.txt"))
    total_games = 0
    total_half_moves = 0
    timecontrol_counter = Counter()
    mode_counter = Counter()
    result_counter = Counter({"win": 0, "loss": 0, "tie": 0, "other": 0})
    unique_users = set()
    user_stats = defaultdict(lambda: {"games": 0, "win": 0, "loss": 0, "tie": 0, "elo_sum": 0, "elo_count": 0})
    opponent_stats = defaultdict(lambda: {
        "games": 0,
        "win": 0,   # wins for *user_id*
        "loss": 0,  # losses for *user_id*
        "tie": 0,
        "opp_elo_sum": 0,
        "opp_elo_count": 0,
        "user_elo_sum": 0,
        "user_elo_count": 0,
    })

    for path in paths:
        headers, moves = parse_email(path)
        if not headers:
            continue

        total_games += 1
        total_half_moves += moves

        tc = headers.get("TimeControl", "Unknown")
        timecontrol_counter[tc] += 1

        mode = headers.get("Mode", "Unknown")
        mode_counter[mode] += 1

        result = headers.get("Result", None)
        white = headers.get("White", "")
        black = headers.get("Black", "")
        white_elo = headers.get("WhiteElo")
        black_elo = headers.get("BlackElo")

        if white:
            unique_users.add(white)
            user_stats[white]["games"] += 1
            if white_elo and white_elo.isdigit():
                user_stats[white]["elo_sum"] += int(white_elo)
                user_stats[white]["elo_count"] += 1
        if black:
            unique_users.add(black)
            user_stats[black]["games"] += 1
            if black_elo and black_elo.isdigit():
                user_stats[black]["elo_sum"] += int(black_elo)
                user_stats[black]["elo_count"] += 1

        # Populate opponent-specific stats relative to user_id
        if user_id in (white, black):
            # Determine opponent name and elos
            if user_id == white:
                opponent = black
                user_elo_val = white_elo
                opp_elo_val = black_elo
                outcome_for_user = result
            else:
                opponent = white
                user_elo_val = black_elo
                opp_elo_val = white_elo
                # invert result perspective when user is black
                if result == "1-0":
                    outcome_for_user = "0-1"  # user lost
                elif result == "0-1":
                    outcome_for_user = "1-0"  # user won
                else:
                    outcome_for_user = result  # tie

            st = opponent_stats[opponent]
            st["games"] += 1

            if outcome_for_user == "1-0":
                st["win"] += 1
            elif outcome_for_user == "0-1":
                st["loss"] += 1
            else:
                st["tie"] += 1

            if opp_elo_val and opp_elo_val.isdigit():
                st["opp_elo_sum"] += int(opp_elo_val)
                st["opp_elo_count"] += 1
            if user_elo_val and user_elo_val.isdigit():
                st["user_elo_sum"] += int(user_elo_val)
                st["user_elo_count"] += 1

        # Update per-user win/loss/tie stats
        if result:
            if result == "1/2-1/2":
                user_stats[white]["tie"] += 1
                user_stats[black]["tie"] += 1

                if user_id in (white, black):
                    result_counter["tie"] += 1
                else:
                    result_counter["other"] += 1
            elif result == "1-0":
                user_stats[white]["win"] += 1
                user_stats[black]["loss"] += 1

                if user_id == white:
                    result_counter["win"] += 1
                elif user_id == black:
                    result_counter["loss"] += 1
                else:
                    result_counter["other"] += 1
            elif result == "0-1":
                user_stats[black]["win"] += 1
                user_stats[white]["loss"] += 1

                if user_id == black:
                    result_counter["win"] += 1
                elif user_id == white:
                    result_counter["loss"] += 1
                else:
                    result_counter["other"] += 1
            else:
                result_counter["other"] += 1

    stats = {
        "total_games": total_games,
        "total_half_moves": total_half_moves,
        "average_half_moves_per_game": round(total_half_moves / total_games, 2) if total_games else 0,
        "timecontrol_distribution": dict(timecontrol_counter),
        "mode_distribution": dict(mode_counter),
        "results_for_user": dict(result_counter),
        "user_id": user_id,
        "unique_user_ids": sorted(unique_users),
        "per_user_stats": user_stats,  # this is defaultdict but will be converted below
    }

    # finalize avg elo
    processed = {}
    for user, v in user_stats.items():
        avg_elo = v["elo_sum"] / v["elo_count"] if v["elo_count"] else None
        out = dict(v)
        out["avg_elo"] = avg_elo
        processed[user] = out

    stats["per_user_stats"] = processed

    # finalise opponent stats with averages
    opp_processed = {}
    for opp, v in opponent_stats.items():
        opp_avg = v["opp_elo_sum"] / v["opp_elo_count"] if v["opp_elo_count"] else None
        user_avg = v["user_elo_sum"] / v["user_elo_count"] if v["user_elo_count"] else None
        out = dict(v)
        out["opp_avg_elo"] = opp_avg
        out["user_avg_elo"] = user_avg
        opp_processed[opp] = out

    stats["opponent_stats"] = opp_processed

    return stats


def guess_user_id():
    """Guess the user ID by counting the most frequent name across headers."""
    name_counter = Counter()
    for path in EMAIL_DIR.glob("*.txt"):
        headers, _ = parse_email(path)
        if headers:
            white = headers.get("White")
            black = headers.get("Black")
            if white:
                name_counter[white] += 1
            if black:
                name_counter[black] += 1
    if not name_counter:
        return None
    user_id, _ = name_counter.most_common(1)[0]
    return user_id


def main():
    parser = argparse.ArgumentParser(description="Aggregate stats for Chess Games emails.")
    parser.add_argument("--user", "-u", dest="user_id", help="User ID to compute win/loss/tie stats for. If omitted, guessed from data.")

    args = parser.parse_args()
    user_id = args.user_id or guess_user_id()
    if user_id is None:
        parser.error("Unable to determine user ID automatically. Please specify with --user.")

    stats = aggregate_stats(user_id)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main() 