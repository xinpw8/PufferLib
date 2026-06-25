"""Persistent league state and batch Elo fitting for policy sweeps."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from copy import deepcopy
from typing import Any, Callable

import numpy as np


STATE_VERSION = 1
ANCHOR_ID = "random"
LOGISTIC_TO_ELO = 400.0 / math.log(10.0)


ARCH_SWEEP_KEYS = {'hidden_size', 'num_layers', 'policy.hidden_size', 'policy.num_layers'}


def validate_no_arch_sweep_keys(sweep_config: dict[str, Any]) -> None:
    sweep_only = str(sweep_config.get('sweep_only', ''))
    sweep_only_tokens = {
        p.strip().replace('/', '.').lower()
        for p in sweep_only.split(',')
        if p.strip()
    }
    if sweep_only_tokens & ARCH_SWEEP_KEYS:
        raise ValueError('league sweeps require fixed policy.hidden_size and policy.num_layers')


def pair_key(a_id: str, b_id: str) -> str:
    a, b = sorted((a_id, b_id))
    return f"{a}||{b}"


def make_state(sweep_id: str, arch: dict[str, int] | None = None,
        config: dict[str, Any] | None = None) -> dict[str, Any]:
    now = time.time()
    return {
        "version": STATE_VERSION,
        "sweep_id": sweep_id,
        "created_at": now,
        "updated_at": now,
        "anchor_id": ANCHOR_ID,
        "arch": deepcopy(arch or {}),
        "config": deepcopy(config or {}),
        "players": [],
        "matches": [],
        "ratings": {},
    }


@contextlib.contextmanager
def _state_lock(path: str):
    lock_path = f"{path}.lock"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(lock_path, "w") as lock_file:
        try:
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except ImportError:
            pass
        try:
            yield
        finally:
            try:
                import fcntl
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except ImportError:
                pass


def read_state(path: str, retries: int = 3, delay: float = 0.05) -> dict[str, Any] | None:
    for attempt in range(retries):
        try:
            with open(path) as f:
                state = json.load(f)
            if int(state.get("version", 0)) != STATE_VERSION:
                raise RuntimeError(f"Unsupported league state version in {path}")
            return state
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            if attempt + 1 >= retries:
                raise
            time.sleep(delay)
    return None


def _write_state(path: str, state: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state["updated_at"] = time.time()
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def update_state(path: str, mutate: Callable[[dict[str, Any]], Any],
        default_state: dict[str, Any] | None = None) -> Any:
    with _state_lock(path):
        state = read_state(path)
        if state is None:
            if default_state is None:
                raise FileNotFoundError(path)
            state = deepcopy(default_state)
        result = mutate(state)
        _write_state(path, state)
        return result


def load_or_create(path: str, sweep_id: str, arch: dict[str, int] | None = None,
        config: dict[str, Any] | None = None) -> dict[str, Any]:
    default = make_state(sweep_id, arch=arch, config=config)

    def _noop(state):
        return deepcopy(state)

    return update_state(path, _noop, default_state=default)


def _players_by_id(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {player["id"]: player for player in state.get("players", [])}


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode('latin1')
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def ensure_anchor(path: str, checkpoint_path: str, arch: dict[str, int],
        hypers: dict[str, Any] | None = None) -> dict[str, Any]:
    def _mutate(state):
        players = _players_by_id(state)
        anchor = players.get(ANCHOR_ID)
        if anchor is None:
            anchor = {
                "id": ANCHOR_ID,
                "run_id": ANCHOR_ID,
                "kind": "anchor",
                "checkpoint_path": checkpoint_path,
                "hypers": deepcopy(hypers or {}),
                "cost": 0.0,
                "elo": 0.0,
                "games": 0,
                "matches": 0,
                "score_sum": 0.0,
                "created_at": time.time(),
            }
            state.setdefault("players", []).append(anchor)
        else:
            anchor.update({
                "checkpoint_path": checkpoint_path,
                "hypers": deepcopy(hypers or anchor.get("hypers", {})),
                "elo": 0.0,
            })
        state["arch"] = deepcopy(arch)
        recompute_ratings_in_state(state)
        return deepcopy(anchor)

    return update_state(path, _mutate)


def register_player(path: str, run_id: str, checkpoint_path: str,
        hypers: dict[str, Any], cost: float, arch: dict[str, int] | None = None) -> dict[str, Any]:
    def _mutate(state):
        players = _players_by_id(state)
        player = players.get(run_id)
        row = {
            "id": run_id,
            "run_id": run_id,
            "kind": "policy",
            "checkpoint_path": checkpoint_path,
            "hypers": _jsonable(deepcopy(hypers)),
            "cost": float(cost),
            "created_at": time.time(),
        }
        if arch is not None:
            row["arch"] = deepcopy(arch)
        if player is None:
            row.update({
                "elo": 0.0,
                "games": 0,
                "matches": 0,
                "score_sum": 0.0,
            })
            state.setdefault("players", []).append(row)
            player = row
        else:
            player.update(row)
        recompute_ratings_in_state(state)
        return deepcopy(player)

    return update_state(path, _mutate)


def opponent_pool(state: dict[str, Any]) -> list[dict[str, Any]]:
    players = []
    for player in state.get("players", []):
        checkpoint_path = player.get("checkpoint_path")
        if not checkpoint_path:
            continue
        if player.get("kind") not in ("anchor", "policy", "random"):
            continue
        players.append({
            "id": player["id"],
            "run_id": player.get("run_id", player["id"]),
            "path": checkpoint_path,
            "elo": float(player.get("elo", 0.0)),
            "kind": player.get("kind", "policy"),
        })
    return players


def run_id_scores(state: dict[str, Any]) -> dict[str, float]:
    scores = {}
    for player in state.get("players", []):
        if player.get("kind") != "policy":
            continue
        run_id = player.get("run_id") or player.get("id")
        if run_id:
            scores[run_id] = float(player.get("elo", 0.0))
    return scores


def _match_edges(matches: list[dict[str, Any]]) -> list[tuple[str, str, float, float]]:
    edges = []
    for match in matches:
        games = float(match.get("games", 0.0))
        if games <= 0:
            continue
        a_id = match.get("a") or match.get("a_id")
        b_id = match.get("b") or match.get("b_id")
        if not a_id or not b_id or a_id == b_id:
            continue
        score = float(match.get("a_score_rate", match.get("a_score", 0.5)))
        score = min(max(score, 1e-6), 1.0 - 1e-6)
        edges.append((a_id, b_id, score, games))
    return edges


def recompute_ratings(players: list[dict[str, Any]],
        matches: list[dict[str, Any]], anchor_id: str = ANCHOR_ID) -> dict[str, float]:
    ids = [player["id"] for player in players]
    if not ids:
        return {}

    variable_ids = [pid for pid in ids if pid != anchor_id]
    var_index = {pid: idx for idx, pid in enumerate(variable_ids)}
    edges = _match_edges(matches)
    ratings = {pid: 0.0 for pid in ids}
    if not edges or not variable_ids:
        ratings[anchor_id] = 0.0
        return ratings

    def beta(beta_vec: np.ndarray, pid: str) -> float:
        if pid == anchor_id:
            return 0.0
        idx = var_index.get(pid)
        return 0.0 if idx is None else float(beta_vec[idx])

    def objective(beta_vec: np.ndarray) -> float:
        loss = 0.0
        for a_id, b_id, score, games in edges:
            z = np.clip(beta(beta_vec, a_id) - beta(beta_vec, b_id), -30.0, 30.0)
            prob = 1.0 / (1.0 + math.exp(-z))
            loss -= games * (score * math.log(prob) + (1.0 - score) * math.log(1.0 - prob))
        # Keeps disconnected components finite and centered without moving the anchor.
        loss += 1e-3 * float(np.dot(beta_vec, beta_vec))
        return loss

    try:
        from scipy.optimize import minimize
        result = minimize(objective, np.zeros(len(variable_ids)), method="L-BFGS-B")
        beta_vec = result.x if result.success or result.x is not None else np.zeros(len(variable_ids))
    except Exception:
        beta_vec = np.zeros(len(variable_ids), dtype=np.float64)
        lr = 0.01 / max(sum(edge[3] for edge in edges), 1.0)
        for _ in range(1000):
            grad = np.zeros_like(beta_vec)
            for a_id, b_id, score, games in edges:
                z = np.clip(beta(beta_vec, a_id) - beta(beta_vec, b_id), -30.0, 30.0)
                prob = 1.0 / (1.0 + math.exp(-z))
                delta = games * (prob - score)
                if a_id in var_index:
                    grad[var_index[a_id]] += delta
                if b_id in var_index:
                    grad[var_index[b_id]] -= delta
            grad += 2e-3 * beta_vec
            beta_vec -= lr * grad

    for pid in variable_ids:
        ratings[pid] = float(beta_vec[var_index[pid]] * LOGISTIC_TO_ELO)
    ratings[anchor_id] = 0.0
    return ratings


def recompute_ratings_in_state(state: dict[str, Any]) -> None:
    players = state.setdefault("players", [])
    matches = state.setdefault("matches", [])
    ratings = recompute_ratings(players, matches, state.get("anchor_id", ANCHOR_ID))
    stats = {
        player["id"]: {"games": 0, "matches": 0, "score_sum": 0.0}
        for player in players
    }
    for match in matches:
        games = int(match.get("games", 0))
        if games <= 0:
            continue
        a_id = match.get("a") or match.get("a_id")
        b_id = match.get("b") or match.get("b_id")
        if a_id not in stats or b_id not in stats:
            continue
        score = float(match.get("a_score_rate", match.get("a_score", 0.5)))
        stats[a_id]["games"] += games
        stats[b_id]["games"] += games
        stats[a_id]["matches"] += 1
        stats[b_id]["matches"] += 1
        stats[a_id]["score_sum"] += score * games
        stats[b_id]["score_sum"] += (1.0 - score) * games

    for player in players:
        pid = player["id"]
        player["elo"] = float(ratings.get(pid, 0.0))
        player.update(stats.get(pid, {"games": 0, "matches": 0, "score_sum": 0.0}))
    state["ratings"] = ratings


def record_match(path: str, a_id: str, b_id: str, games: int,
        a_score_rate: float, draw_rate: float = 0.0) -> dict[str, float]:
    def _mutate(state):
        state.setdefault("matches", []).append({
            "a": a_id,
            "b": b_id,
            "games": int(games),
            "a_score_rate": float(a_score_rate),
            "draw_rate": float(draw_rate),
            "time": time.time(),
        })
        recompute_ratings_in_state(state)
        return deepcopy(state.get("ratings", {}))

    return update_state(path, _mutate)


def pair_counts(matches: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in matches:
        a_id = match.get("a") or match.get("a_id")
        b_id = match.get("b") or match.get("b_id")
        if not a_id or not b_id:
            continue
        key = pair_key(a_id, b_id)
        counts[key] = counts.get(key, 0) + 1
    return counts


def choose_match_pair(state: dict[str, Any], rng: random.Random | None = None,
        anchor_prob: float = 0.12, temperature: float = 150.0,
        neighbor_window: int = 12) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = rng or random
    players = opponent_pool(state)
    policies = [p for p in players if p.get("kind") == "policy"]
    anchor = next((p for p in players if p["id"] == state.get("anchor_id", ANCHOR_ID)), None)
    if not policies:
        raise RuntimeError("Need at least one completed policy for league matches")

    by_id = {p["id"]: p for p in state.get("players", [])}
    counts = pair_counts(state.get("matches", []))

    def matches(pid: str) -> int:
        return int(by_id.get(pid, {}).get("matches", 0))

    if anchor is not None and rng.random() < anchor_prob:
        weights = [1.0 / math.sqrt(matches(p["id"]) + 1.0) for p in policies]
        return anchor, rng.choices(policies, weights=weights, k=1)[0]

    if len(policies) == 1:
        if anchor is None:
            raise RuntimeError("Need at least two policies or an anchor for league matches")
        return policies[0], anchor

    weights = [1.0 / math.sqrt(matches(p["id"]) + 1.0) for p in policies]
    a = rng.choices(policies, weights=weights, k=1)[0]
    candidates = [p for p in policies if p["id"] != a["id"]]
    candidates.sort(key=lambda p: abs(float(p.get("elo", 0.0)) - float(a.get("elo", 0.0))))
    candidates = candidates[:max(1, int(neighbor_window))]
    cand_weights = []
    for b in candidates:
        dist = abs(float(a.get("elo", 0.0)) - float(b.get("elo", 0.0)))
        dist_w = math.exp(-dist / max(float(temperature), 1e-6))
        count_w = 1.0 / math.sqrt(counts.get(pair_key(a["id"], b["id"]), 0) + 1.0)
        cand_weights.append(dist_w * count_w)
    return a, rng.choices(candidates, weights=cand_weights, k=1)[0]
