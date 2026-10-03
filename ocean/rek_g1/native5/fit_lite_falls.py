#!/usr/bin/env python3
"""Fit a rek.lite_falls.v1 model from rek.lite_fall_dataset.v1 aggregates.

The dataset logger (lite_fall_dataset.cu) runs the physical MuJoCo + SONIC
runtime and aggregates, on the GPU, one cell per
(own move bin, opponent move bin, distance bin, closing bin, struck):
upright exposure ticks and fall onsets. Because the hazard model is a logistic
function of one-hot cell features, those sufficient statistics fit it exactly
as row-level data would, at any dataset size.

  fit   DATASET.json NEW_MODEL.json [--holdout HOLDOUT.json]
  smoke NEW_MODEL.json   synthetic, hand-written model for pipeline tests only
  check MODEL.json       validate a model file against the runtime contract

numpy is the only dependency. Nothing here runs during training.
"""
import argparse
import hashlib
import json
import math
import sys

import numpy as np

SCHEMA = "rek.lite_falls.v1"
DATASET_SCHEMA = "rek.lite_fall_dataset.v1"
MOVES, PHASE_BINS = 17, 4
MOVE_BINS = MOVES * PHASE_BINS + 3
IDLE_BIN, TRANSLATE_BIN, YAW_BIN = MOVES * PHASE_BINS, MOVES * PHASE_BINS + 1, MOVES * PHASE_BINS + 2
DISTANCE_BINS, CLOSING_BINS, STRUCK = 5, 3, 2
CLASSES, OUTCOMES, QUANTILES = 3, 3, 9
RECOVER, FALLEN_QUICK, STUCK = 0, 1, 2
KICK_MOVES = (6, 7, 8, 9)
LEVELS = np.linspace(0.0, 1.0, QUANTILES)
CELL_SHAPE = (MOVE_BINS, MOVE_BINS, DISTANCE_BINS, CLOSING_BINS, STRUCK)


def move_class(bin_index):
    if bin_index >= IDLE_BIN:
        return 0
    return 2 if bin_index // PHASE_BINS in KICK_MOVES else 1


def require(ok, message):
    if not ok:
        raise SystemExit(f"fit_lite_falls: {message}")


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_dataset(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    require(data.get("schema") == DATASET_SCHEMA, f"{path}: schema")
    dims = data["dimensions"]
    require([dims["move_bins"], dims["move_bins"], dims["distance_bins"],
             dims["closing_bins"], dims["struck"]] == list(CELL_SHAPE), f"{path}: dimensions")
    exposures = np.asarray(data["exposures"], dtype=np.float64)
    onsets = np.asarray(data["onsets"], dtype=np.float64)
    size = int(np.prod(CELL_SHAPE))
    require(exposures.shape == (size,) and onsets.shape == (size,), f"{path}: cell arrays")
    require(np.all(exposures >= 0) and np.all(onsets >= 0) and np.all(onsets <= exposures),
            f"{path}: onsets must not exceed exposures")
    data["_exposures"], data["_onsets"] = exposures, onsets
    data["_sha256"] = sha256_file(path)
    return data


def cell_indices():
    """Feature indices for every cell, in C order of CELL_SHAPE."""
    own, opp, dist, close, struck = np.indices(CELL_SHAPE).reshape(5, -1)
    return own, opp, dist, close, struck


# Parameter layout: bias, own, opponent, distance, closing, own x distance,
# opponent x distance, struck.
OFFSETS = {}
_cursor = 0
for _name, _count in (("bias", 1), ("own", MOVE_BINS), ("opponent", MOVE_BINS),
                      ("distance", DISTANCE_BINS), ("closing", CLOSING_BINS),
                      ("own_distance", MOVE_BINS * DISTANCE_BINS),
                      ("opponent_distance", MOVE_BINS * DISTANCE_BINS), ("struck", 1)):
    OFFSETS[_name] = (_cursor, _count)
    _cursor += _count
PARAMETERS = _cursor


def design(own, opp, dist, close, struck):
    """Column index of each active one-hot feature, shape (cells, 7)."""
    o = OFFSETS
    columns = [
        np.full_like(own, o["bias"][0]),
        o["own"][0] + own,
        o["opponent"][0] + opp,
        o["distance"][0] + dist,
        o["closing"][0] + close,
        o["own_distance"][0] + own * DISTANCE_BINS + dist,
        o["opponent_distance"][0] + opp * DISTANCE_BINS + dist,
    ]
    return np.stack(columns, axis=1), struck.astype(np.float64)


def logits(weights, columns, struck):
    return weights[columns].sum(axis=1) + struck * weights[OFFSETS["struck"][0]]


def objective(weights, columns, struck, exposures, onsets, l2):
    z = logits(weights, columns, struck)
    # Binomial log likelihood with n = exposures, k = onsets.
    log1pexp = np.logaddexp(0.0, z)
    nll = float(np.sum(exposures * log1pexp - onsets * z))
    probability = np.exp(z - log1pexp)
    residual = exposures * probability - onsets
    gradient = np.zeros(PARAMETERS)
    for k in range(columns.shape[1]):
        gradient += np.bincount(columns[:, k], weights=residual, minlength=PARAMETERS)
    gradient[OFFSETS["struck"][0]] += float(np.sum(residual * struck))
    penalty = weights.copy()
    penalty[OFFSETS["bias"][0]] = 0.0  # The intercept is not shrunk.
    return nll + 0.5 * l2 * float(penalty @ penalty), gradient + l2 * penalty


def lbfgs(function, x, iterations=500, memory=12, tolerance=1e-7):
    value, gradient = function(x)
    s_list, y_list = [], []
    for _ in range(iterations):
        q = gradient.copy()
        alphas = []
        for s, y in reversed(list(zip(s_list, y_list))):
            rho = 1.0 / float(y @ s)
            alpha = rho * float(s @ q)
            alphas.append((rho, alpha))
            q -= alpha * y
        if y_list:
            q *= float(s_list[-1] @ y_list[-1]) / float(y_list[-1] @ y_list[-1])
        for (s, y), (rho, alpha) in zip(zip(s_list, y_list), reversed(alphas)):
            beta = rho * float(y @ q)
            q += (alpha - beta) * s
        direction = -q
        step = 1.0
        slope = float(gradient @ direction)
        if slope >= 0:
            direction, slope = -gradient, -float(gradient @ gradient)
        while True:
            candidate = x + step * direction
            new_value, new_gradient = function(candidate)
            if new_value <= value + 1e-4 * step * slope or step < 1e-12:
                break
            step *= 0.5
        s, y = candidate - x, new_gradient - gradient
        if float(s @ y) > 1e-12:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > memory:
                s_list.pop(0)
                y_list.pop(0)
        converged = abs(value - new_value) <= tolerance * max(1.0, abs(value))
        x, value, gradient = candidate, new_value, new_gradient
        if converged:
            break
    return x, value


def evaluate(weights, dataset):
    own, opp, dist, close, struck = cell_indices()
    columns, struck_values = design(own, opp, dist, close, struck)
    exposures, onsets = dataset["_exposures"], dataset["_onsets"]
    used = exposures > 0
    z = logits(weights, columns[used], struck_values[used])
    probability = 1.0 / (1.0 + np.exp(-z))
    total = float(exposures[used].sum())
    events = float(onsets[used].sum())
    log_loss = float(np.sum(exposures[used] * np.logaddexp(0.0, z) - onsets[used] * z)) / max(total, 1.0)
    base_rate = events / max(total, 1.0)
    base = 0.0 if base_rate in (0.0, 1.0) else -(base_rate * math.log(base_rate) + (1 - base_rate) * math.log(1 - base_rate))
    expected = exposures[used] * probability
    order = np.argsort(probability)
    cumulative = np.cumsum(exposures[used][order])
    deciles = []
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = order[(cumulative > low * total) & (cumulative <= high * total + 1e-9)]
        deciles.append({"exposure_ticks": float(exposures[used][selected].sum()),
                        "expected_onsets": float(expected[selected].sum()),
                        "observed_onsets": float(onsets[used][selected].sum())})
    return {"exposure_ticks": total, "onsets": events, "log_loss_per_tick": log_loss,
            "constant_rate_log_loss_per_tick": base,
            "expected_onsets": float(expected.sum()), "calibration_deciles": deciles}


def kaplan_meier_quantiles(event_ticks, censored_ticks, maximum):
    """Quantiles of a delay with right-censoring; unreached levels use the maximum."""
    events = np.bincount(np.asarray(event_ticks, dtype=np.int64), minlength=maximum + 1)[: maximum + 1]
    censored = np.bincount(np.asarray(censored_ticks, dtype=np.int64), minlength=maximum + 1)[: maximum + 1]
    at_risk = np.cumsum((events + censored)[::-1])[::-1].astype(np.float64)
    survival = np.cumprod(np.where(at_risk > 0, 1.0 - events / np.maximum(at_risk, 1.0), 1.0))
    cdf = 1.0 - survival
    out = []
    for level in LEVELS:
        if level <= 0.0:
            nonzero = np.nonzero(events)[0]
            out.append(float(nonzero[0]) if nonzero.size else 1.0)
            continue
        reached = np.nonzero(cdf >= level - 1e-12)[0]
        out.append(float(reached[0]) if reached.size else float(maximum))
    return np.maximum.accumulate(np.maximum(out, 1.0)).tolist()


def histogram_values(histogram):
    histogram = np.asarray(histogram, dtype=np.int64)
    return np.repeat(np.arange(histogram.size), histogram)


def fit(args):
    train = load_dataset(args.dataset)
    holdout = load_dataset(args.holdout) if args.holdout else None
    if holdout is not None:
        require(holdout["distance_edges_m"] == train["distance_edges_m"]
                and holdout["closing_edges_m_s"] == train["closing_edges_m_s"], "holdout bin edges differ")
    own, opp, dist, close, struck = cell_indices()
    columns, struck_values = design(own, opp, dist, close, struck)
    exposures, onsets = train["_exposures"], train["_onsets"]
    used = exposures > 0
    require(onsets.sum() > 0, "training dataset contains no fall onsets")
    columns, struck_values = columns[used], struck_values[used]
    exposures, onsets = exposures[used], onsets[used]
    x = np.zeros(PARAMETERS)
    rate = onsets.sum() / exposures.sum()
    x[OFFSETS["bias"][0]] = math.log(rate / (1 - rate))
    weights, value = lbfgs(lambda w: objective(w, columns, struck_values, exposures, onsets, args.l2),
                           x, iterations=args.iterations)

    outcome = train["outcomes"]
    maximum = int(outcome["max_delay_ticks"])
    threshold = int(args.stuck_threshold_ticks)
    probabilities, recover_all, quick_all, stuck_all, censored_all = [], [], [], [], []
    for c in range(CLASSES):
        recover = histogram_values(outcome["recover_delay_histogram"][c])
        fallen = histogram_values(outcome["fallen_delay_histogram"][c])
        censored = histogram_values(outcome["censored_delay_histogram"][c])
        quick, stuck = fallen[fallen <= threshold], fallen[fallen > threshold]
        counts = np.array([recover.size, quick.size, stuck.size + censored.size], dtype=np.float64)
        probabilities.append(((counts + args.outcome_prior) / (counts.sum() + OUTCOMES * args.outcome_prior)).tolist())
        recover_all.append(recover)
        quick_all.append(quick)
        stuck_all.append(stuck)
        censored_all.append(censored)
    recover_all, quick_all = np.concatenate(recover_all), np.concatenate(quick_all)
    stuck_all, censored_all = np.concatenate(stuck_all), np.concatenate(censored_all)
    quantiles = [
        kaplan_meier_quantiles(recover_all, [], maximum) if recover_all.size else [1.0] * QUANTILES,
        kaplan_meier_quantiles(quick_all, [], maximum) if quick_all.size else [1.0] * QUANTILES,
        kaplan_meier_quantiles(stuck_all, censored_all, maximum) if stuck_all.size + censored_all.size
        else [float(threshold)] * QUANTILES,
    ]
    obs = train.get("observation_means", {})
    model = model_document(
        model_id=args.model_id or hashlib.sha256((train["_sha256"] + json.dumps(vars(args), sort_keys=True,
                                                   default=str)).encode()).hexdigest(),
        distance_edges=train["distance_edges_m"], closing_edges=train["closing_edges_m_s"],
        weights=weights, outcome_probability=probabilities, delay_quantiles=quantiles,
        observation={
            "falling_tilt_degrees": float(obs.get("falling_tilt_degrees", 60.0)),
            "fallen_tilt_degrees": float(obs.get("fallen_tilt_degrees", 92.0)),
            "falling_height_ratio": float(obs.get("falling_height_ratio", 0.5)),
            "fallen_height_ratio": float(obs.get("fallen_height_ratio", 0.12)),
        },
        provenance={
            "synthetic": False,
            "training_dataset": {"path": args.dataset, "sha256": train["_sha256"],
                                 "dataset_id": train.get("dataset_id"), "provenance": train.get("provenance")},
            "holdout_dataset": None if holdout is None else {
                "path": args.holdout, "sha256": holdout["_sha256"], "dataset_id": holdout.get("dataset_id")},
            "l2": args.l2, "iterations": args.iterations, "stuck_threshold_ticks": threshold,
            "outcome_prior": args.outcome_prior, "objective": value,
            "training_metrics": evaluate(weights, train),
            "holdout_metrics": None if holdout is None else evaluate(weights, holdout),
            "outcome_counts_by_class": outcome.get("counts_by_class"),
        })
    write_model(args.output, model)
    metrics = model["provenance"]
    print(json.dumps({"model_id": model["model_id"], "training_log_loss": metrics["training_metrics"]["log_loss_per_tick"],
                      "training_constant_rate_log_loss": metrics["training_metrics"]["constant_rate_log_loss_per_tick"],
                      "holdout_log_loss": None if holdout is None else metrics["holdout_metrics"]["log_loss_per_tick"],
                      "outcome_probability": probabilities, "delay_quantiles_ticks": quantiles}))


def model_document(model_id, distance_edges, closing_edges, weights, outcome_probability,
                   delay_quantiles, observation, provenance):
    def part(name):
        start, count = OFFSETS[name]
        return [float(v) for v in weights[start:start + count]]

    def table(name):
        values = part(name)
        return [values[i * DISTANCE_BINS:(i + 1) * DISTANCE_BINS] for i in range(MOVE_BINS)]

    return {
        "schema": SCHEMA, "model_id": model_id,
        "phase_bins": PHASE_BINS, "move_bins": MOVE_BINS, "distance_bins": DISTANCE_BINS,
        "closing_bins": CLOSING_BINS, "quantiles": QUANTILES,
        "distance_edges_m": [float(v) for v in distance_edges],
        "closing_edges_m_s": [float(v) for v in closing_edges],
        "bias": part("bias")[0], "own_move": part("own"), "opponent_move": part("opponent"),
        "distance": part("distance"), "closing": part("closing"),
        "own_move_distance": table("own_distance"), "opponent_move_distance": table("opponent_distance"),
        "struck": part("struck")[0],
        "outcome_probability": [[float(p) for p in row] for row in outcome_probability],
        "delay_quantiles_ticks": [[float(q) for q in row] for row in delay_quantiles],
        "observation": observation, "provenance": provenance,
    }


def write_model(path, model):
    check_model(model)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(model, handle, indent=1)
        handle.write("\n")


def check_model(model):
    require(model["schema"] == SCHEMA, "schema")
    require(len(model["own_move"]) == MOVE_BINS and len(model["opponent_move"]) == MOVE_BINS, "move bins")
    require(len(model["own_move_distance"]) == MOVE_BINS and all(len(r) == DISTANCE_BINS for r in model["own_move_distance"]), "own x distance")
    require(len(model["opponent_move_distance"]) == MOVE_BINS, "opponent x distance")
    require(len(model["distance_edges_m"]) == DISTANCE_BINS - 1 and len(model["closing_edges_m_s"]) == CLOSING_BINS - 1, "edges")
    for row in model["outcome_probability"]:
        require(len(row) == OUTCOMES and abs(sum(row) - 1) <= 1e-6 and min(row) >= 0, "outcome probabilities")
    for row in model["delay_quantiles_ticks"]:
        require(len(row) == QUANTILES and all(b >= a for a, b in zip(row, row[1:])) and row[0] >= 0, "delay quantiles")
    flat = [model["bias"], model["struck"], *model["own_move"], *model["opponent_move"],
            *model["distance"], *model["closing"]]
    flat += [v for r in model["own_move_distance"] for v in r] + [v for r in model["opponent_move_distance"] for v in r]
    require(all(math.isfinite(v) for v in flat), "nonfinite weights")


def smoke(args):
    """Hand-written synthetic model. It exercises falls, stuck downs, counts,
    knockouts and spawn resets in the lite runtime. It is not fitted to any
    physics and must never back a training-quality claim."""
    weights = np.zeros(PARAMETERS)
    weights[OFFSETS["bias"][0]] = -11.0
    close_bins = (0, 1)  # under the first two distance edges
    for move in KICK_MOVES:
        for phase in (1, 2):  # single-support middle of the kick
            bin_index = move * PHASE_BINS + phase
            for d in close_bins:
                weights[OFFSETS["own_distance"][0] + bin_index * DISTANCE_BINS + d] = 5.5
                weights[OFFSETS["opponent_distance"][0] + bin_index * DISTANCE_BINS + d] = 3.0
    weights[OFFSETS["struck"][0]] = 4.0
    model = model_document(
        model_id="synthetic-smoke-v1",
        distance_edges=[0.55, 0.85, 1.2, 1.8], closing_edges=[-0.5, 0.5],
        weights=weights,
        outcome_probability=[[0.85, 0.12, 0.03], [0.6, 0.3, 0.1], [0.4, 0.45, 0.15]],
        delay_quantiles=[[8, 12, 16, 20, 25, 30, 38, 48, 60],
                         [12, 16, 20, 24, 28, 33, 40, 50, 70],
                         [80, 110, 140, 170, 200, 240, 290, 360, 500]],
        observation={"falling_tilt_degrees": 60.0, "fallen_tilt_degrees": 92.0,
                     "falling_height_ratio": 0.5, "fallen_height_ratio": 0.12},
        provenance={"synthetic": True, "fitted": False,
                    "purpose": "pipeline and referee smoke tests only; not physics"})
    write_model(args.output, model)
    print(json.dumps({"model_id": model["model_id"], "synthetic": True}))


def check(args):
    with open(args.model, encoding="utf-8") as handle:
        check_model(json.load(handle))
    print(json.dumps({"ok": True, "model": args.model, "sha256": sha256_file(args.model)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit")
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--holdout")
    p.add_argument("--model-id")
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--stuck-threshold-ticks", type=int, default=75)
    p.add_argument("--outcome-prior", type=float, default=1.0)
    p.set_defaults(run=fit)
    p = sub.add_parser("smoke")
    p.add_argument("output")
    p.set_defaults(run=smoke)
    p = sub.add_parser("check")
    p.add_argument("model")
    p.set_defaults(run=check)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    sys.exit(main())
