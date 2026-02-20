'''Heads-up play/eval tooling for PokeBattle with Showdown replay export.

Modes:
  - human-vs-bot:   play as p1 against built-in bot (random/heuristic/mcts)
  - human-vs-policy: play as p2 against a trained policy (policy controls p1)
  - policy-vs-bot: evaluate a trained policy against built-in bot
'''

from __future__ import annotations

import argparse
import glob
import os
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import pufferlib.pytorch
from pufferlib.ocean.poke_battle.poke_battle import PokeBattle


OBS_SIZE = 140
NUM_ACTIONS = 10
REPLAY_JS = "https://play.pokemonshowdown.com/js/replay-embed.js"
MODE_NORMAL = 0
MODE_P1_FORCE_SWITCH = 1
MODE_P2_FORCE_SWITCH = 2
MODE_BOTH_FORCE_SWITCH = 3


@dataclass
class EpisodeStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    episodes: int = 0

    def update(self, reward: float):
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.draws += 1
        self.episodes += 1

    def summary(self) -> str:
        if self.episodes == 0:
            return "no episodes"
        wr = 100.0 * self.wins / self.episodes
        lr = 100.0 * self.losses / self.episodes
        dr = 100.0 * self.draws / self.episodes
        return (
            f"wins={self.wins}/{self.episodes} ({wr:.1f}%), "
            f"losses={self.losses}/{self.episodes} ({lr:.1f}%), "
            f"draws={self.draws}/{self.episodes} ({dr:.1f}%)"
        )


class ShowdownReplay:
    def __init__(self, p1_name: str, p2_name: str):
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.lines: list[str] = []

    @staticmethod
    def _hp_status(mon: dict) -> str:
        hp = int(mon["hp"])
        max_hp = int(mon["max_hp"])
        if hp <= 0:
            return "0 fnt"

        status = mon.get("status_name", "")
        if status:
            return f"{hp}/{max_hp} {status}"
        return f"{hp}/{max_hp}"

    @staticmethod
    def _ident(side: str, mon: dict, active_idx: int, idx: int) -> str:
        species = mon.get("species", "Unknown")
        if idx == active_idx:
            return f"{side}a: {species}"
        return f"{side}: {species}"

    @staticmethod
    def _action_name(player: dict, action: int) -> tuple[str, str]:
        active_idx = int(player["active_idx"])
        active = player["team"][active_idx]
        if 0 <= action < 4:
            move = active["moves"][action]["name"]
            return "move", move
        if 4 <= action < 10:
            target_idx = action - 4
            target = player["team"][target_idx]["species"]
            return "switch", target
        return "move", "Struggle"

    def start(self, state: dict):
        p1_team = state["p1"]["team"]
        p2_team = state["p2"]["team"]
        self.lines.extend([
            f"|player|p1|{self.p1_name}|169",
            f"|player|p2|{self.p2_name}|169",
            "|teamsize|p1|6",
            "|teamsize|p2|6",
            "|gen|1",
            "|tier|[Gen 1] OU",
            "|gametype|singles",
        ])
        for mon in p1_team:
            species = mon.get("species", "Unknown")
            self.lines.append(f"|poke|p1|{species}, L100")
        for mon in p2_team:
            species = mon.get("species", "Unknown")
            self.lines.append(f"|poke|p2|{species}, L100")
        self.lines.append("|start")

        p1_idx = int(state["p1"]["active_idx"])
        p2_idx = int(state["p2"]["active_idx"])
        p1_mon = p1_team[p1_idx]
        p2_mon = p2_team[p2_idx]
        self.lines.append(
            f"|switch|p1a: {p1_mon['species']}|{p1_mon['species']}, L100|{self._hp_status(p1_mon)}"
        )
        self.lines.append(
            f"|switch|p2a: {p2_mon['species']}|{p2_mon['species']}, L100|{self._hp_status(p2_mon)}"
        )

    @classmethod
    def turn_lines_from_state(cls, pre: dict, post: dict) -> list[str]:
        return cls("p1", "p2").turn_lines(pre, post)

    def turn_lines(self, pre: dict, post: dict) -> list[str]:
        lines: list[str] = []
        turn = int(pre["turn"]) + 1
        lines.append(f"|turn|{turn}")

        p1_action = int(post.get("last_p1_action", 0))
        p2_action = int(post.get("last_p2_action", 0))
        p1_kind, p1_name = self._action_name(pre["p1"], p1_action)
        p2_kind, p2_name = self._action_name(pre["p2"], p2_action)

        pre_p1_idx = int(pre["p1"]["active_idx"])
        pre_p2_idx = int(pre["p2"]["active_idx"])
        pre_p1 = pre["p1"]["team"][pre_p1_idx]
        pre_p2 = pre["p2"]["team"][pre_p2_idx]

        if p1_kind == "move":
            lines.append(
                f"|move|p1a: {pre_p1['species']}|{p1_name}|p2a: {pre_p2['species']}"
            )
        if p2_kind == "move":
            lines.append(
                f"|move|p2a: {pre_p2['species']}|{p2_name}|p1a: {pre_p1['species']}"
            )

        post_p1_idx = int(post["p1"]["active_idx"])
        post_p2_idx = int(post["p2"]["active_idx"])
        if p1_kind == "switch" or pre_p1_idx != post_p1_idx:
            mon = post["p1"]["team"][post_p1_idx]
            lines.append(
                f"|switch|p1a: {mon['species']}|{mon['species']}, L100|{self._hp_status(mon)}"
            )
        if p2_kind == "switch" or pre_p2_idx != post_p2_idx:
            mon = post["p2"]["team"][post_p2_idx]
            lines.append(
                f"|switch|p2a: {mon['species']}|{mon['species']}, L100|{self._hp_status(mon)}"
            )

        lines.extend(self._state_diff_lines(pre, post, "p1"))
        lines.extend(self._state_diff_lines(pre, post, "p2"))
        return lines

    def add_turn(self, pre: dict, post: dict):
        self.lines.extend(self.turn_lines(pre, post))

    def _state_diff_lines(self, pre: dict, post: dict, side: str) -> list[str]:
        lines: list[str] = []
        pre_side = pre[side]
        post_side = post[side]
        pre_active = int(pre_side["active_idx"])
        post_active = int(post_side["active_idx"])

        for idx, (pre_mon, post_mon) in enumerate(zip(pre_side["team"], post_side["team"])):
            pre_alive = bool(pre_mon.get("is_alive", 0))
            post_alive = bool(post_mon.get("is_alive", 0))
            ident = self._ident(side, post_mon, post_active, idx)

            pre_hp = int(pre_mon.get("hp", 0))
            post_hp = int(post_mon.get("hp", 0))
            if post_hp < pre_hp:
                lines.append(f"|-damage|{ident}|{self._hp_status(post_mon)}")
            elif post_hp > pre_hp:
                lines.append(f"|-heal|{ident}|{self._hp_status(post_mon)}")

            pre_status = pre_mon.get("status_name", "")
            post_status = post_mon.get("status_name", "")
            if pre_status != post_status:
                if pre_status and not post_status:
                    lines.append(f"|-curestatus|{ident}|{pre_status}")
                elif post_status:
                    lines.append(f"|-status|{ident}|{post_status}")

            if pre_alive and not post_alive:
                faint_ident = self._ident(side, pre_mon, pre_active, idx)
                lines.append(f"|faint|{faint_ident}")

        return lines

    def finish(self, reward: float):
        if reward > 0:
            self.lines.append(f"|win|{self.p1_name}")
        elif reward < 0:
            self.lines.append(f"|win|{self.p2_name}")
        else:
            self.lines.append("|tie")

    def to_log(self) -> str:
        return "\n".join(self.lines)

    def write(self, out_stem: Path) -> tuple[Path, Path]:
        log_path = out_stem.with_suffix(".log")
        html_path = out_stem.with_suffix(".html")
        log_text = self.to_log()
        log_path.write_text(log_text + "\n", encoding="utf-8")
        html = (
            "<!doctype html>\n"
            "<html><head><meta charset=\"utf-8\">"
            "<title>PokeBattle Replay</title>"
            f"<script src=\"{REPLAY_JS}\"></script>"
            "</head><body>\n"
            "<script type=\"text/plain\" class=\"battle-log-data\">\n"
            f"{log_text}\n"
            "</script>\n"
            "</body></html>\n"
        )
        html_path.write_text(html, encoding="utf-8")
        return log_path, html_path


def _latest_checkpoint() -> str | None:
    root = Path(__file__).resolve().parents[3]
    pattern = str(root / "experiments" / "puffer_poke_battle_*" / "model_puffer_poke_battle_*.pt")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _load_policy(model_path: str, env: PokeBattle, device: str):
    from pufferlib.ocean.torch import PokeBattle as PokeBattlePolicy, PokeBattleLSTM

    policy = PokeBattlePolicy(env, hidden_size=256)
    policy = PokeBattleLSTM(env, policy, input_size=256, hidden_size=256)
    policy = policy.to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def _action_mask(obs: np.ndarray, for_opponent_in_selfplay: bool = False) -> np.ndarray:
    if obs.shape[-1] < OBS_SIZE:
        raise ValueError(f"Observation too small: {obs.shape}")
    if for_opponent_in_selfplay:
        return obs[OBS_SIZE + 130:OBS_SIZE + 140] > 0.5
    return obs[130:140] > 0.5


def _needs_switch(mode: int, player_side: int) -> bool:
    if player_side == 0:
        return mode in (MODE_P1_FORCE_SWITCH, MODE_BOTH_FORCE_SWITCH)
    return mode in (MODE_P2_FORCE_SWITCH, MODE_BOTH_FORCE_SWITCH)


def _phase_text(mode: int, player_side: int) -> str:
    if mode == MODE_NORMAL:
        return "normal turn"
    if _needs_switch(mode, player_side):
        return "forced switch: choose a replacement"
    return "waiting: opponent must choose a replacement"


def _has_usable_move(player: dict) -> bool:
    active = player["team"][int(player["active_idx"])]
    for move in active["moves"]:
        if int(move.get("id", 0)) > 0 and int(move.get("pp", 0)) > 0:
            return True
    return False


def _is_pass_only(mask: np.ndarray, mode: int, player_side: int) -> bool:
    valid = set(np.flatnonzero(mask).tolist())
    return mode != MODE_NORMAL and not _needs_switch(mode, player_side) and valid == {0}


def _render_turn_log(pre: dict, post: dict):
    for line in ShowdownReplay.turn_lines_from_state(pre, post):
        print(line)


def _format_player(player: dict, label: str) -> str:
    active_idx = int(player["active_idx"])
    active = player["team"][active_idx]
    parts = [
        f"{label} active: {active['species']} "
        f"HP {active['hp']}/{active['max_hp']} "
        f"{active.get('status_name', '')}".strip(),
        f"Side: reflect={player['has_reflect']} light_screen={player['has_light_screen']}",
        "Team:",
    ]
    for i, mon in enumerate(player["team"]):
        alive = "alive" if mon["is_alive"] else "fainted"
        marker = "*" if i == active_idx else " "
        parts.append(
            f"  {marker}{i}: {mon['species']:<10} hp={mon['hp']:>3}/{mon['max_hp']:<3} "
            f"status={mon.get('status_name', '-') or '-':<3} {alive}"
        )
    return "\n".join(parts)


def _format_opponent(player: dict, label: str) -> str:
    active_idx = int(player["active_idx"])
    active = player["team"][active_idx]
    status = active.get("status_name", "") or "-"
    parts = [
        f"{label} active: {active['species']} "
        f"HP {active['hp']}/{active['max_hp']} "
        f"status={status}",
        "Opponent team:",
    ]
    for i, mon in enumerate(player["team"]):
        alive = "alive" if mon["is_alive"] else "fainted"
        marker = "*" if i == active_idx else " "
        mon_status = mon.get("status_name", "") or "-"
        parts.append(
            f"  {marker}{i}: {mon['species']:<10} hp={mon['hp']:>3}/{mon['max_hp']:<3} "
            f"status={mon_status:<3} {alive}"
        )
    return "\n".join(parts)


def _action_menu(player: dict, mask: np.ndarray, battle_mode: int, player_side: int) -> list[str]:
    active = player["team"][int(player["active_idx"])]
    pass_only = _is_pass_only(mask, battle_mode, player_side)
    must_switch = (battle_mode != MODE_NORMAL) and _needs_switch(battle_mode, player_side)
    no_usable_moves = (battle_mode == MODE_NORMAL) and (not _has_usable_move(player))
    is_recharging = bool(player.get("is_recharging", 0))
    labels: list[str] = []
    for action in range(NUM_ACTIONS):
        if action < 4:
            if action == 0 and pass_only:
                labels.append(f"{action}: pass (waiting for opponent switch)")
            elif pass_only:
                move = active["moves"][action]["name"]
                labels.append(f"{action}: move {move} (disabled: opponent switch)")
            elif must_switch:
                move = active["moves"][action]["name"]
                labels.append(f"{action}: move {move} (disabled: forced switch)")
            elif action == 0 and is_recharging:
                labels.append(f"{action}: pass (recharge turn)")
            elif action == 0 and no_usable_moves:
                labels.append(f"{action}: move Struggle")
            else:
                move = active["moves"][action]["name"]
                labels.append(f"{action}: move {move}")
        else:
            idx = action - 4
            mon = player["team"][idx]["species"]
            if pass_only:
                labels.append(f"{action}: switch {mon} (disabled: opponent switch)")
            else:
                labels.append(f"{action}: switch {mon}")
    valid = set(np.flatnonzero(mask).tolist())
    return [f"{line} {'[valid]' if int(line.split(':', 1)[0]) in valid else ''}".rstrip() for line in labels]


def _prompt_action(
        player: dict,
        mask: np.ndarray,
        role_name: str,
        battle_mode: int,
        player_side: int,
        opponent: dict | None = None,
        opponent_label: str = "Opponent") -> int:
    valid = set(np.flatnonzero(mask).tolist())
    if not valid:
        raise RuntimeError("No valid actions available.")

    if opponent is not None:
        print(_format_opponent(opponent, opponent_label))
    print(_format_player(player, role_name))
    print(f"Phase: {_phase_text(battle_mode, player_side)}")
    print("Actions:")
    for line in _action_menu(player, mask, battle_mode, player_side):
        print(f"  {line}")

    if _is_pass_only(mask, battle_mode, player_side):
        print(f"{role_name}: auto-pass while opponent performs forced switch.")
        return 0

    while True:
        try:
            raw = input(f"{role_name} action ({sorted(valid)}): ").strip().lower()
        except EOFError as exc:
            raise KeyboardInterrupt from exc
        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        try:
            action = int(raw)
        except ValueError:
            print("Enter an integer action id.")
            continue
        if action not in valid:
            print("Illegal action for current state.")
            continue
        return action


def _replay_path(base_dir: Path, mode: str, episode_idx: int) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return base_dir / f"poke_{mode}_{ts}_ep{episode_idx:04d}"


def run_human_vs_bot(args):
    base_seed = int(args.seed) if args.seed is not None else (time.time_ns() & 0x7FFFFFFF)
    env = PokeBattle(
        num_envs=1,
        selfplay=0,
        bot_mode=args.bot_mode,
        mcts_iterations=args.mcts_iterations,
        mcts_depth=args.mcts_depth,
        seed=base_seed,
        auto_reset=0,
    )

    replay_dir = Path(args.replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)
    stats = EpisodeStats()

    try:
        for ep in range(1, args.episodes + 1):
            env.reset(seed=base_seed + ep)
            state = env.get_state(0)
            replay = ShowdownReplay("Human", f"Bot-{args.bot_mode}") if args.save_replay else None
            if replay:
                replay.start(state)

            print(f"\n=== Episode {ep} / {args.episodes} ===")
            try:
                while True:
                    mask = _action_mask(env.observations[0])
                    action = _prompt_action(
                        state["p1"], mask, "You",
                        battle_mode=int(state["mode"]), player_side=0,
                        opponent=state["p2"], opponent_label="Bot")
                    pre = state
                    env.step(np.asarray([action], dtype=np.int32))
                    state = env.get_state(0)
                    _render_turn_log(pre, state)
                    if replay:
                        replay.add_turn(pre, state)

                    if bool(env.terminals[0]):
                        reward = float(env.rewards[0])
                        stats.update(reward)
                        if reward > 0:
                            print("Result: You won.")
                        elif reward < 0:
                            print("Result: You lost.")
                        else:
                            print("Result: Draw.")

                        if replay:
                            replay.finish(reward)
                            stem = _replay_path(replay_dir, "human_vs_bot", ep)
                            log_path, html_path = replay.write(stem)
                            print(f"Replay saved: {log_path}")
                            print(f"Replay viewer: {html_path}")
                            if args.open_replay:
                                webbrowser.open(html_path.resolve().as_uri())
                        break
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break

            print(f"Running stats: {stats.summary()}")
    finally:
        env.close()


def run_human_vs_policy(args):
    base_seed = int(args.seed) if args.seed is not None else (time.time_ns() & 0x7FFFFFFF)
    model_path = args.model_path or _latest_checkpoint()
    if not model_path:
        raise RuntimeError("No checkpoint found. Provide --model-path.")

    device = args.device
    env = PokeBattle(
        num_envs=1,
        selfplay=1,
        bot_mode=0,
        seed=base_seed,
        auto_reset=0,
    )
    policy = _load_policy(model_path, env, device)

    replay_dir = Path(args.replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)
    stats = EpisodeStats()

    lstm_h = torch.zeros(1, policy.hidden_size, device=device)
    lstm_c = torch.zeros(1, policy.hidden_size, device=device)

    try:
        for ep in range(1, args.episodes + 1):
            env.reset(seed=base_seed + ep)
            state = env.get_state(0)
            replay = ShowdownReplay("Policy", "Human") if args.save_replay else None
            if replay:
                replay.start(state)

            print(f"\n=== Episode {ep} / {args.episodes} ===")
            try:
                while True:
                    obs = env.observations[0]
                    learner_obs = obs[:OBS_SIZE]
                    human_mask = _action_mask(obs, for_opponent_in_selfplay=True)
                    human_action = _prompt_action(
                        state["p2"], human_mask, "You (p2)",
                        battle_mode=int(state["mode"]), player_side=1,
                        opponent=state["p1"], opponent_label="Policy")

                    obs_t = torch.as_tensor(learner_obs, device=device).unsqueeze(0)
                    net_state = dict(lstm_h=lstm_h, lstm_c=lstm_c)
                    with torch.no_grad():
                        logits, _ = policy.forward_eval(obs_t, net_state)
                        pol_action, _, _ = pufferlib.pytorch.sample_logits(logits)
                    lstm_h = net_state.get("lstm_h", lstm_h)
                    lstm_c = net_state.get("lstm_c", lstm_c)

                    pre = state
                    env.step(np.asarray([int(pol_action.item()), human_action], dtype=np.int32))
                    state = env.get_state(0)
                    _render_turn_log(pre, state)
                    if replay:
                        replay.add_turn(pre, state)

                    if bool(env.terminals[0]):
                        learner_reward = float(env.rewards[0])
                        human_reward = -learner_reward
                        stats.update(human_reward)
                        if human_reward > 0:
                            print("Result: You won.")
                        elif human_reward < 0:
                            print("Result: You lost.")
                        else:
                            print("Result: Draw.")

                        if replay:
                            replay.finish(learner_reward)
                            stem = _replay_path(replay_dir, "human_vs_policy", ep)
                            log_path, html_path = replay.write(stem)
                            print(f"Replay saved: {log_path}")
                            print(f"Replay viewer: {html_path}")
                            if args.open_replay:
                                webbrowser.open(html_path.resolve().as_uri())
                        break
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break

            print(f"Running stats: {stats.summary()}")
    finally:
        env.close()


def run_policy_vs_bot(args):
    base_seed = int(args.seed) if args.seed is not None else (time.time_ns() & 0x7FFFFFFF)
    model_path = args.model_path or _latest_checkpoint()
    if not model_path:
        raise RuntimeError("No checkpoint found. Provide --model-path.")

    device = args.device
    env = PokeBattle(
        num_envs=1,
        selfplay=0,
        bot_mode=args.bot_mode,
        mcts_iterations=args.mcts_iterations,
        mcts_depth=args.mcts_depth,
        seed=base_seed,
        auto_reset=0,
    )
    policy = _load_policy(model_path, env, device)

    replay_dir = Path(args.replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)
    stats = EpisodeStats()
    lstm_h = torch.zeros(1, policy.hidden_size, device=device)
    lstm_c = torch.zeros(1, policy.hidden_size, device=device)

    try:
        start = time.time()
        total_steps = 0
        for ep in range(1, args.episodes + 1):
            env.reset(seed=base_seed + ep)
            state = env.get_state(0)
            replay = ShowdownReplay("Policy", f"Bot-{args.bot_mode}") if args.save_replay else None
            if replay:
                replay.start(state)

            while True:
                obs_t = torch.as_tensor(env.observations, device=device)
                net_state = dict(lstm_h=lstm_h, lstm_c=lstm_c)
                with torch.no_grad():
                    logits, _ = policy.forward_eval(obs_t, net_state)
                    pol_action, _, _ = pufferlib.pytorch.sample_logits(logits)
                lstm_h = net_state.get("lstm_h", lstm_h)
                lstm_c = net_state.get("lstm_c", lstm_c)

                pre = state
                env.step(np.asarray([int(pol_action.item())], dtype=np.int32))
                total_steps += 1
                state = env.get_state(0)
                if replay:
                    replay.add_turn(pre, state)

                if bool(env.terminals[0]):
                    reward = float(env.rewards[0])
                    stats.update(reward)
                    if replay:
                        replay.finish(reward)
                        stem = _replay_path(replay_dir, "policy_vs_bot", ep)
                        replay.write(stem)
                    lstm_h.zero_()
                    lstm_c.zero_()
                    break

        elapsed = time.time() - start
        print(f"Checkpoint: {model_path}")
        print(f"Mode: policy-vs-bot (bot_mode={args.bot_mode})")
        print(stats.summary())
        if elapsed > 0:
            print(f"Episodes/s: {stats.episodes / elapsed:.2f}, Env steps/s: {total_steps / elapsed:.1f}")
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Heads-up tools for PokeBattle.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["human-vs-bot", "human-vs-policy", "policy-vs-bot"],
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bot-mode", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--mcts-iterations", type=int, default=128)
    parser.add_argument("--mcts-depth", type=int, default=5)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-replay", action="store_true")
    parser.add_argument("--open-replay", action="store_true")
    parser.add_argument("--replay-dir", type=str, default="experiments/poke_replays")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "human-vs-bot":
        run_human_vs_bot(args)
    elif args.mode == "human-vs-policy":
        run_human_vs_policy(args)
    elif args.mode == "policy-vs-bot":
        run_policy_vs_bot(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
