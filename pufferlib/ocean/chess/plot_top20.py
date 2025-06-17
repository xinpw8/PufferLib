import matplotlib.pyplot as plt
from aggregate_chess_stats import aggregate_stats, guess_user_id
import numpy as np


def main():
    stats = aggregate_stats(guess_user_id())
    per = stats["opponent_stats"]
    # list of tuples (opponent, data)
    top = sorted(per.items(), key=lambda x: x[1]["games"], reverse=True)[:20]

    users   = [u for u,_ in top]
    games   = np.array([d["games"] for _,d in top])
    tk_wins = np.array([d["win"] for _,d in top])  # TQK wins
    tk_losses = np.array([d["loss"] for _,d in top])
    opp_avg_elo  = np.array([d["opp_avg_elo"] if d["opp_avg_elo"] is not None else np.nan for _,d in top])
    tk_avg_elo   = np.array([d["user_avg_elo"] if d["user_avg_elo"] is not None else np.nan for _,d in top])

    tk_win_pct = tk_wins / (tk_wins + tk_losses)

    x = np.arange(len(users))
    fig, ax1 = plt.subplots(figsize=(14,7))

    # Bar plot for games played; colored by win percentage
    cmap = plt.cm.get_cmap('RdYlGn')
    colors = cmap(tk_win_pct)
    ax1.set_yscale('log')
    bars = ax1.bar(x, games, color=colors, edgecolor='black')
    ax1.set_ylabel('Games Played')
    ax1.set_xticks(x)
    ax1.set_xticklabels(users, rotation=45, ha='right')
    ax1.set_title('Top 20 Opponents – Games Played, Win% (bar color), Avg Elo (line)')

    # Annotate bars with win pct
    for rect, pct in zip(bars, tk_win_pct):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2, height*1.05, f'{pct*100:.1f}%',
                  ha='center', va='bottom', fontsize=8)

    # Second axis for average Elo
    ax2 = ax1.twinx()
    ax2.plot(x, opp_avg_elo, color='red', marker='o', linewidth=2, label='Opponent Avg Elo')
    ax2.plot(x, tk_avg_elo, color='blue', marker='x', linewidth=2, label='TQK Avg Elo')
    ax2.set_ylabel('Average Elo')
    ax2.set_ylim(bottom=min(opp_avg_elo[np.isfinite(opp_avg_elo)])*0.9 if np.isfinite(opp_avg_elo).any() else 0)

    # Legend for bar color mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[ax1, ax2])
    cbar.set_label('TheQuadKnight Win %')

    ax2.legend(loc='upper left')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main() 