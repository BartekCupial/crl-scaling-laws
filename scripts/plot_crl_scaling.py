#!/usr/bin/env python3
"""Download CRL histories through wandb-cache and make no-fit scaling plots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from wandb_cache import WandbRunCache  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = "ideas-ncbr/crl_scaling_laws"
DEFAULT_GROUP_PREFIX = "humanoid_crl_v2"
DEFAULT_METRIC = "training/critic_loss"
DEFAULT_CACHE_DIR = REPO_ROOT / ".wandb_cache"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures" / "scaling_laws"
DEFAULT_FLOP_PROFILE = REPO_ROOT / "configs" / "crl_scaling_humanoid_v2_flops.json"
DEFAULT_CHECKPOINTS = (1_000_000, 10_000_000, 30_000_000, 100_000_000)
DEFAULT_ISOFLOP_BUDGETS = (
    8_000_000_000_000_000,
    12_000_000_000_000_000,
    20_000_000_000_000_000,
    30_000_000_000_000_000,
    50_000_000_000_000_000,
    80_000_000_000_000_000,
    120_000_000_000_000_000,
    200_000_000_000_000_000,
    300_000_000_000_000_000,
    500_000_000_000_000_000,
    800_000_000_000_000_000,
    1_200_000_000_000_000_000,
    1_600_000_000_000_000_000,
    2_000_000_000_000_000_000,
    3_000_000_000_000_000_000,
)
CONFIG_KEYS = (
    "model_family",
    "use_simba",
    "critic_network_width",
    "critic_depth",
    "actor_network_width",
    "actor_depth",
    "seed",
    "total_env_steps",
    "total_trainable_params",
    "batch_size",
    "num_sgd_batches_per_training_step",
    "num_envs",
    "unroll_length",
    "min_replay_size",
)
PLOT_STAGES = ("fit", "heldout")
RECOGNIZED_STAGES = ("smoke", *PLOT_STAGES)
STAGE_MARKERS = {"fit": "s", "heldout": "D", "unknown": "x"}
RUN_STATUS_MARKERS = {"finished": "o", "ongoing": "^", "other": "X"}
RUN_STATUS_LABELS = {"finished": "finished", "ongoing": "ongoing", "other": "other"}
RUN_STATUS_ORDER = ("finished", "ongoing", "other")


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def parse_str_list(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def experiment_groups(prefix: str, seeds: Sequence[int]) -> list[str]:
    return [f"{prefix}_{stage}_s{seed}" for stage in PLOT_STAGES for seed in seeds]


def expected_simba_params(width: int, depth: int) -> int:
    """Exact count used by the Humanoid SimBa grid, including actor and alpha."""
    return (
        728 * width
        + 163
        + (17 * depth - 16) * width**2
        + (13 * depth - 10) * width
    )


def fetch_history(
    *,
    project: str,
    group_prefix: str,
    seeds: Sequence[int],
    metric: str,
    cache_dir: Path,
    refresh: bool,
    samples: int,
    max_workers: int,
) -> pd.DataFrame:
    groups = experiment_groups(group_prefix, seeds)
    filters = {"group": {"$in": groups}}
    cache = WandbRunCache(
        project=project,
        cache=f"{group_prefix}_scaling_history_no_smoke",
        cache_dir=cache_dir,
    )
    return cache.history_dataframe(
        filters=filters,
        graphql_filters=filters,
        refresh_cache=refresh,
        keys=["training/envsteps", "training/walltime", metric],
        x_axis="training/envsteps",
        samples=samples,
        max_workers=max_workers,
        config_keys=CONFIG_KEYS,
    )


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def normalize_history(raw: pd.DataFrame, metric: str, group_prefix: str) -> pd.DataFrame:
    required = {"run_id", "run_name", "run_group", "training/envsteps", metric}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"W&B history is missing required columns: {', '.join(missing)}")

    frame = raw.copy()
    frame["env_steps"] = _numeric_column(frame, "training/envsteps")
    frame["loss"] = _numeric_column(frame, metric)
    frame["width"] = _numeric_column(frame, "config.critic_network_width")
    frame["depth"] = _numeric_column(frame, "config.critic_depth")
    frame["seed"] = _numeric_column(frame, "config.seed")
    frame["planned_env_steps"] = _numeric_column(frame, "config.total_env_steps")
    frame["training_walltime"] = _numeric_column(frame, "training/walltime")
    frame["batch_size"] = _numeric_column(frame, "config.batch_size")
    frame["sgd_batches_per_training_step"] = _numeric_column(
        frame, "config.num_sgd_batches_per_training_step"
    )
    frame["num_envs"] = _numeric_column(frame, "config.num_envs")
    frame["unroll_length"] = _numeric_column(frame, "config.unroll_length")
    frame["min_replay_size"] = _numeric_column(frame, "config.min_replay_size")
    frame["run_state"] = (
        frame.get("run_state", pd.Series("unknown", index=frame.index))
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )
    frame["run_status"] = np.select(
        [frame["run_state"].eq("finished"), frame["run_state"].eq("running")],
        ["finished", "ongoing"],
        default="other",
    )
    frame["trainable_params"] = _numeric_column(
        frame, "config.total_trainable_params"
    ).astype(float)

    missing_params = frame["trainable_params"].isna()
    can_reconstruct = missing_params & frame["width"].notna() & frame["depth"].notna()
    if can_reconstruct.any():
        frame.loc[can_reconstruct, "trainable_params"] = [
            expected_simba_params(int(width), int(depth))
            for width, depth in frame.loc[
                can_reconstruct, ["width", "depth"]
            ].itertuples(index=False)
        ]

    recognized_stages = "|".join(RECOGNIZED_STAGES)
    stage_pattern = rf"^{group_prefix}_({recognized_stages})_s\d+$"
    frame["stage"] = frame["run_group"].astype(str).str.extract(stage_pattern, expand=False)
    frame["stage"] = frame["stage"].fillna("unknown")
    frame = frame.loc[frame["stage"].isin(PLOT_STAGES)].copy()
    frame["model_family"] = (
        frame.get("config.model_family", pd.Series("simba", index=frame.index))
        .fillna("simba")
        .astype(str)
    )

    valid = (
        frame["env_steps"].gt(0)
        & frame["loss"].gt(0)
        & frame["trainable_params"].gt(0)
        & frame["width"].gt(0)
        & frame["depth"].gt(0)
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError(f"No positive finite observations are available for log plotting {metric!r}")

    frame["progress"] = frame["env_steps"] / frame["planned_env_steps"]
    frame["architecture"] = [
        f"{family}-w{int(width)}-d{int(depth)}"
        for family, width, depth in frame[["model_family", "width", "depth"]].itertuples(index=False)
    ]
    frame = frame.sort_values(["run_id", "env_steps"]).drop_duplicates(
        ["run_id", "env_steps"], keep="last"
    )
    frame["is_latest"] = frame["env_steps"].eq(frame.groupby("run_id")["env_steps"].transform("max"))
    return frame.reset_index(drop=True)


def filter_run_variant(history: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Select legacy or extended runs from their unique W&B run names."""
    if variant == "all":
        return history.copy()
    is_extended = history["run_name"].astype(str).str.contains(
        r"_ext_v2$", regex=True, na=False
    )
    selected = history.loc[is_extended if variant == "extended" else ~is_extended].copy()
    if selected.empty:
        raise ValueError(f"No {variant} runs are available after filtering")
    return selected.reset_index(drop=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_flop_profile(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported FLOP profile schema in {path}")
    expected_hashes = {
        "profiler_sha256": REPO_ROOT / "scripts" / "profile_crl_flops.py",
        "train_sha256": REPO_ROOT / "train.py",
        "simba_sha256": REPO_ROOT / "simba.py",
    }
    stale = [
        source.name
        for key, source in expected_hashes.items()
        if document.get(key) != _file_sha256(source)
    ]
    if stale:
        names = ", ".join(stale)
        raise ValueError(
            f"FLOP profile {path} is stale for {names}; rerun "
            "`uv run --no-sync python scripts/profile_crl_flops.py`"
        )
    return document


def add_flop_estimates(
    history: pd.DataFrame, profile: dict[str, Any]
) -> pd.DataFrame:
    """Add cumulative XLA-estimated model-training FLOPs to every observation."""
    frame = history.copy()
    required_positive = (
        "batch_size",
        "sgd_batches_per_training_step",
        "num_envs",
        "unroll_length",
        "min_replay_size",
    )
    invalid = [
        name
        for name in required_positive
        if name not in frame or frame[name].isna().any() or not frame[name].gt(0).all()
    ]
    if invalid:
        raise ValueError(
            "Cannot estimate FLOPs because these positive run settings are missing: "
            + ", ".join(invalid)
        )
    profile_batch_size = int(profile["batch_size"])
    observed_batch_sizes = {int(value) for value in frame["batch_size"].unique()}
    if observed_batch_sizes != {profile_batch_size}:
        raise ValueError(
            f"FLOP profile uses batch size {profile_batch_size}, but runs use "
            f"{sorted(observed_batch_sizes)}"
        )

    profiles = {
        (int(item["width"]), int(item["depth"])): item
        for item in profile["profiles"]
    }
    architecture_keys = [
        (int(width), int(depth))
        for width, depth in frame[["width", "depth"]].itertuples(index=False)
    ]
    missing = sorted(set(architecture_keys) - set(profiles))
    if missing:
        raise ValueError(f"FLOP profile is missing architectures: {missing}")

    frame["sgd_update_flops"] = [
        profiles[key]["sgd_update_flops"] for key in architecture_keys
    ]
    frame["sgd_update_transcendentals"] = [
        profiles[key]["sgd_update_transcendentals"] for key in architecture_keys
    ]
    frame["rollout_flops_per_env_step"] = [
        profiles[key]["rollout_flops_per_env_step"] for key in architecture_keys
    ]
    frame["rollout_transcendentals_per_env_step"] = [
        profiles[key]["rollout_transcendentals_per_env_step"]
        for key in architecture_keys
    ]

    env_steps_per_actor_step = frame["num_envs"] * frame["unroll_length"]
    prefill_actor_steps = np.ceil(
        frame["min_replay_size"] / frame["unroll_length"]
    )
    actual_prefill_env_steps = prefill_actor_steps * env_steps_per_actor_step
    training_actor_steps = np.floor(
        np.maximum(0.0, frame["env_steps"] - actual_prefill_env_steps)
        / env_steps_per_actor_step
        + 1e-9
    )
    frame["gradient_updates"] = (
        training_actor_steps * frame["sgd_batches_per_training_step"]
    ).astype("int64")
    frame["estimated_training_flops"] = (
        frame["gradient_updates"] * frame["sgd_update_flops"]
        + frame["env_steps"] * frame["rollout_flops_per_env_step"]
    )
    frame["estimated_training_transcendentals"] = (
        frame["gradient_updates"] * frame["sgd_update_transcendentals"]
        + frame["env_steps"] * frame["rollout_transcendentals_per_env_step"]
    )
    frame["estimated_flops_per_second"] = (
        frame["estimated_training_flops"] / frame["training_walltime"]
    ).where(frame["training_walltime"].gt(0))
    return frame


def select_checkpoint_observations(
    history: pd.DataFrame,
    checkpoints: Sequence[int],
    relative_tolerance: float = 0.20,
) -> pd.DataFrame:
    """Select the closest observed row for each reached run/checkpoint pair."""
    selected: list[pd.Series] = []
    for _run_id, run in history.groupby("run_id", sort=False):
        max_steps = float(run["env_steps"].max())
        for checkpoint in checkpoints:
            if max_steps < checkpoint * (1.0 - relative_tolerance):
                continue
            distances = (run["env_steps"] - checkpoint).abs()
            row = run.loc[distances.idxmin()].copy()
            if abs(float(row["env_steps"]) - checkpoint) > checkpoint * relative_tolerance:
                continue
            row["checkpoint_env_steps"] = int(checkpoint)
            selected.append(row)
    if not selected:
        return history.iloc[0:0].assign(checkpoint_env_steps=pd.Series(dtype="int64"))
    return pd.DataFrame(selected).reset_index(drop=True)


def interpolate_isoflop_observations(
    history: pd.DataFrame, budgets: Sequence[int]
) -> pd.DataFrame:
    """Interpolate each run in log-compute/log-loss space without extrapolation."""
    selected: list[pd.Series] = []
    for _run_id, run in history.groupby("run_id", sort=False):
        run = (
            run.sort_values("estimated_training_flops")
            .drop_duplicates("estimated_training_flops", keep="last")
            .reset_index(drop=True)
        )
        compute = run["estimated_training_flops"].to_numpy(dtype=float)
        losses = run["loss"].to_numpy(dtype=float)
        env_steps = run["env_steps"].to_numpy(dtype=float)
        for budget in budgets:
            budget = float(budget)
            if budget < compute[0] or budget > compute[-1]:
                continue
            right = int(np.searchsorted(compute, budget, side="left"))
            if right < len(compute) and compute[right] == budget:
                left = right
                fraction = 0.0
                loss = losses[right]
                budget_env_steps = env_steps[right]
            else:
                left = max(0, right - 1)
                if right >= len(compute):
                    continue
                log_left = np.log(compute[left])
                log_right = np.log(compute[right])
                fraction = (np.log(budget) - log_left) / (log_right - log_left)
                loss = np.exp(
                    np.log(losses[left])
                    + fraction * (np.log(losses[right]) - np.log(losses[left]))
                )
                budget_env_steps = np.exp(
                    np.log(env_steps[left])
                    + fraction * (np.log(env_steps[right]) - np.log(env_steps[left]))
                )
            row = run.iloc[-1].copy()
            row["loss"] = loss
            row["estimated_training_flops"] = budget
            row["isoflop_budget"] = int(budget)
            row["isoflop_env_steps"] = budget_env_steps
            row["interpolation_left_flops"] = compute[left]
            row["interpolation_right_flops"] = compute[right]
            row["interpolation_fraction"] = fraction
            row["is_interpolated"] = left != right
            selected.append(row)
    if not selected:
        return history.iloc[0:0].assign(
            isoflop_budget=pd.Series(dtype="int64"),
            isoflop_env_steps=pd.Series(dtype="float64"),
        )
    return pd.DataFrame(selected).reset_index(drop=True)


def select_budget_minimum_observations(
    history: pd.DataFrame, budgets: Sequence[int]
) -> pd.DataFrame:
    """Select the lowest observed loss from each run up to each compute budget.

    Each run is treated as a truncated training run at every budget it reaches.
    Values other than the budget come from the checkpoint attaining that minimum,
    so loss, samples, and actual compute remain a coherent observation.
    """
    selected: list[pd.Series] = []
    for _run_id, run in history.groupby("run_id", sort=False):
        run = run.sort_values("estimated_training_flops")
        for budget in budgets:
            eligible = run.loc[run["estimated_training_flops"].le(float(budget))]
            if eligible.empty or float(run["estimated_training_flops"].max()) < budget:
                continue
            row = eligible.loc[eligible["loss"].idxmin()].copy()
            row["training_budget_flops"] = int(budget)
            row["best_loss_observation_flops"] = row["estimated_training_flops"]
            row["best_loss_observation_env_steps"] = row["env_steps"]
            selected.append(row)
    if not selected:
        return history.iloc[0:0].assign(
            training_budget_flops=pd.Series(dtype="int64"),
            best_loss_observation_flops=pd.Series(dtype="float64"),
            best_loss_observation_env_steps=pd.Series(dtype="float64"),
        )
    candidates = pd.DataFrame(selected).reset_index(drop=True)
    # Extended runs supersede the old 100M run for the same scientific cell as
    # soon as they reach a budget.  This avoids double-counting a seed while
    # retaining old observations until the replacement has enough compute.
    cell_keys = [
        "stage", "model_family", "width", "depth", "seed",
        "training_budget_flops",
    ]
    candidates = candidates.sort_values(
        [*cell_keys, "planned_env_steps", "loss"],
        ascending=[True] * len(cell_keys) + [False, True],
    )
    return candidates.drop_duplicates(cell_keys, keep="first").reset_index(drop=True)


def select_budget_optimal_observations(
    budget_minima: pd.DataFrame,
) -> pd.DataFrame:
    """Keep the single lowest-loss run at every compute budget."""
    if budget_minima.empty:
        return budget_minima.copy()
    indices = budget_minima.groupby("training_budget_flops")["loss"].idxmin()
    return (
        budget_minima.loc[indices]
        .sort_values("training_budget_flops")
        .reset_index(drop=True)
    )


def _log_norm(values: pd.Series) -> LogNorm:
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum == maximum:
        minimum *= 0.9
        maximum *= 1.1
    return LogNorm(vmin=minimum, vmax=maximum)


def _format_count(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:g}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"


def _format_flops(value: float) -> str:
    if value >= 1e18:
        return f"{value / 1e18:g} EFLOPs"
    if value >= 1e15:
        return f"{value / 1e15:g} PFLOPs"
    if value >= 1e12:
        return f"{value / 1e12:g} TFLOPs"
    return f"{value:g} FLOPs"


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str | None = None) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.18, linewidth=0.6)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        fig.savefig(output_dir / f"{stem}.{file_format}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_status_handles(frame: pd.DataFrame) -> list[Line2D]:
    present = set(frame["run_status"])
    return [
        Line2D(
            [], [], linestyle="none", marker=RUN_STATUS_MARKERS[status],
            markerfacecolor="0.6", markeredgecolor="black", markersize=7,
            label=RUN_STATUS_LABELS[status],
        )
        for status in RUN_STATUS_ORDER
        if status in present
    ]


def _scatter_status_endpoints(
    ax: plt.Axes,
    rows: pd.DataFrame,
    x_column: str,
    color_column: str,
    cmap: Any,
    norm: LogNorm,
) -> None:
    for status in RUN_STATUS_ORDER:
        status_rows = rows.loc[rows["run_status"].eq(status)]
        if status_rows.empty:
            continue
        ax.scatter(
            status_rows[x_column], status_rows["loss"],
            c=status_rows[color_column], cmap=cmap, norm=norm,
            marker=RUN_STATUS_MARKERS[status], edgecolors="black",
            linewidths=0.55, s=54 if status == "ongoing" else 42, zorder=3,
        )


def _scatter_status_xy_endpoints(
    ax: plt.Axes,
    rows: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str,
    cmap: Any,
    norm: LogNorm,
) -> None:
    """Draw latest points while preserving the shared run-state marker language."""
    for status in RUN_STATUS_ORDER:
        status_rows = rows.loc[rows["run_status"].eq(status)]
        if status_rows.empty:
            continue
        ax.scatter(
            status_rows[x_column], status_rows[y_column],
            c=status_rows[color_column], cmap=cmap, norm=norm,
            marker=RUN_STATUS_MARKERS[status], edgecolors="black",
            linewidths=0.55, s=54 if status == "ongoing" else 42, zorder=3,
        )


def latest_architecture_observations(history: pd.DataFrame) -> pd.DataFrame:
    """Keep the furthest-progressed run for each architecture and seed."""
    keys = ["model_family", "width", "depth", "seed"]
    return (
        history.sort_values("env_steps")
        .groupby(keys, dropna=False, sort=False)
        .tail(1)
        .sort_values("trainable_params")
        .reset_index(drop=True)
    )


def plot_parameters(
    history: pd.DataFrame,
    metric: str,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    latest = latest_architecture_observations(history)
    step_norm = _log_norm(latest["env_steps"])
    step_cmap = plt.get_cmap("plasma")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for stage, stage_rows in latest.groupby("stage"):
        for status in RUN_STATUS_ORDER:
            rows = stage_rows.loc[stage_rows["run_status"].eq(status)]
            if rows.empty:
                continue
            marker = (
                STAGE_MARKERS.get(stage, "x")
                if status == "finished"
                else RUN_STATUS_MARKERS[status]
            )
            ax.scatter(
                rows["trainable_params"], rows["loss"], c=rows["env_steps"],
                cmap=step_cmap, norm=step_norm, marker=marker,
                s=58 if status == "ongoing" else 48, alpha=0.82,
                edgecolors="black", linewidths=0.35,
            )
    _style_axis(ax, "Trainable parameters", metric)
    ax.set_title("CRL loss vs model size — latest observation per architecture")
    legend_handles = [
        Line2D(
            [], [], linestyle="none", marker=STAGE_MARKERS.get(stage, "x"),
            markerfacecolor="0.6", markeredgecolor="black", markersize=7,
            label=f"{stage} (finished)",
        )
        for stage in sorted(latest.loc[latest["run_status"].eq("finished"), "stage"].unique())
    ]
    legend_handles.extend(
        handle
        for handle in _run_status_handles(latest)
        if handle.get_label() != "finished"
    )
    ax.legend(handles=legend_handles, title="Run status", frameon=False, loc="best")
    fig.colorbar(ScalarMappable(norm=step_norm, cmap=step_cmap), ax=ax, label="Environment steps")
    fig.tight_layout()
    _save_figure(fig, output_dir, "loss_vs_parameters", formats)


def plot_dataset_trajectories(
    history: pd.DataFrame,
    metric: str,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    latest = history.loc[history["is_latest"]]
    param_norm = _log_norm(history["trainable_params"])
    param_cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for _run_id, run in history.groupby("run_id", sort=False):
        params = float(run["trainable_params"].iloc[0])
        color = param_cmap(param_norm(params))
        ax.plot(run["env_steps"], run["loss"], color=color, alpha=0.65, linewidth=1.2)
        ax.scatter(run["env_steps"], run["loss"], color=color, alpha=0.62, s=13, linewidths=0)
    _scatter_status_endpoints(
        ax, latest, "env_steps", "trainable_params", param_cmap, param_norm
    )
    _style_axis(ax, "Dataset size (environment steps)", metric)
    ax.set_title("CRL loss vs dataset size — observed trajectories only")
    fig.colorbar(ScalarMappable(norm=param_norm, cmap=param_cmap), ax=ax, label="Trainable parameters")
    ax.legend(handles=_run_status_handles(latest), title="Run state", frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir, "loss_vs_dataset_size", formats)


def plot_compute_trajectories(
    history: pd.DataFrame,
    metric: str,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    latest = history.loc[history["is_latest"]]
    norm = _log_norm(history["trainable_params"])
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for _run_id, run in history.groupby("run_id", sort=False):
        color = cmap(norm(float(run["trainable_params"].iloc[0])))
        ax.plot(
            run["estimated_training_flops"], run["loss"],
            color=color, alpha=0.65, linewidth=1.2,
        )
        ax.scatter(
            run["estimated_training_flops"], run["loss"],
            color=color, alpha=0.62, s=13, linewidths=0,
        )
    _scatter_status_endpoints(
        ax, latest, "estimated_training_flops", "trainable_params", cmap, norm
    )
    _style_axis(ax, "Estimated model-training compute (FLOPs)", metric)
    ax.set_title("CRL loss vs XLA-estimated compute — observed trajectories only")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Trainable parameters")
    ax.legend(handles=_run_status_handles(latest), title="Run state", frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir, "loss_vs_compute", formats)


def plot_parameters_vs_compute(
    history: pd.DataFrame,
    metric: str,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """Plot model size against cumulative compute at every observed checkpoint."""
    latest = history.loc[history["is_latest"]]
    loss_norm = _log_norm(history["loss"])
    loss_cmap = plt.get_cmap("magma_r")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for _run_id, run in history.groupby("run_id", sort=False):
        ax.plot(
            run["estimated_training_flops"], run["trainable_params"],
            color="0.72", alpha=0.45, linewidth=0.8, zorder=0,
        )
    ax.scatter(
        history["estimated_training_flops"], history["trainable_params"],
        c=history["loss"], cmap=loss_cmap, norm=loss_norm,
        alpha=0.58, s=13, linewidths=0, zorder=1,
    )
    _scatter_status_xy_endpoints(
        ax, latest, "estimated_training_flops", "trainable_params",
        "loss", loss_cmap, loss_norm,
    )
    _style_axis(
        ax, "Estimated model-training compute (FLOPs)", "Trainable parameters"
    )
    ax.set_title("Model size vs XLA-estimated compute — observed checkpoints")
    fig.colorbar(ScalarMappable(norm=loss_norm, cmap=loss_cmap), ax=ax, label=metric)
    ax.legend(handles=_run_status_handles(latest), title="Run state", frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir, "parameters_vs_compute", formats)


def plot_samples_vs_compute(
    history: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """Plot collected environment samples against cumulative training compute."""
    latest = history.loc[history["is_latest"]]
    param_norm = _log_norm(history["trainable_params"])
    param_cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for _run_id, run in history.groupby("run_id", sort=False):
        color = param_cmap(param_norm(float(run["trainable_params"].iloc[0])))
        ax.plot(
            run["estimated_training_flops"], run["env_steps"],
            color=color, alpha=0.65, linewidth=1.2,
        )
        ax.scatter(
            run["estimated_training_flops"], run["env_steps"],
            color=color, alpha=0.62, s=13, linewidths=0,
        )
    _scatter_status_xy_endpoints(
        ax, latest, "estimated_training_flops", "env_steps",
        "trainable_params", param_cmap, param_norm,
    )
    _style_axis(
        ax, "Estimated model-training compute (FLOPs)",
        "Environment samples (steps)",
    )
    ax.set_title("Environment samples vs XLA-estimated compute — observed checkpoints")
    fig.colorbar(
        ScalarMappable(norm=param_norm, cmap=param_cmap),
        ax=ax, label="Trainable parameters",
    )
    ax.legend(handles=_run_status_handles(latest), title="Run state", frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir, "samples_vs_compute", formats)


def _plot_budget_minimum_scatter(
    observations: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    color_column: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    title: str,
    stem: str,
    output_dir: Path,
    formats: Sequence[str],
    cmap_name: str,
) -> None:
    if observations.empty:
        return
    norm = _log_norm(observations[color_column])
    cmap = plt.get_cmap(cmap_name)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for status in RUN_STATUS_ORDER:
        rows = observations.loc[observations["run_status"].eq(status)]
        if rows.empty:
            continue
        ax.scatter(
            rows[x_column], rows[y_column], c=rows[color_column],
            cmap=cmap, norm=norm, marker=RUN_STATUS_MARKERS[status],
            s=58 if status == "ongoing" else 44, alpha=0.78,
            edgecolors="black", linewidths=0.35,
        )
    _style_axis(ax, xlabel, ylabel)
    ax.set_title(title)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=colorbar_label)
    ax.legend(
        handles=_run_status_handles(observations), title="Run state", frameon=False
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, stem, formats)


def plot_budget_minimum_points(
    observations: pd.DataFrame,
    metric: str,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """Make isoFLOP candidates and Jens-style empirical-optimum dot plots."""
    optima = select_budget_optimal_observations(observations)
    common = dict(output_dir=output_dir, formats=formats)
    _plot_budget_minimum_scatter(
        observations,
        x_column="trainable_params", y_column="loss",
        color_column="training_budget_flops",
        xlabel="Trainable parameters", ylabel=metric,
        colorbar_label="Compute budget (FLOPs)",
        title="Best loss per run and compute budget vs model size",
        stem="best_loss_vs_parameters_budget_points", cmap_name="plasma",
        **common,
    )
    _plot_budget_minimum_scatter(
        optima,
        x_column="training_budget_flops", y_column="loss",
        color_column="trainable_params",
        xlabel="Compute budget (FLOPs)", ylabel=metric,
        colorbar_label="Trainable parameters",
        title="Empirical optimal loss vs compute budget",
        stem="best_loss_vs_compute_budget_points", cmap_name="viridis",
        **common,
    )
    _plot_budget_minimum_scatter(
        optima,
        x_column="training_budget_flops", y_column="trainable_params",
        color_column="loss",
        xlabel="Compute budget (FLOPs)", ylabel="Trainable parameters",
        colorbar_label=metric,
        title="Empirical loss-optimal model size vs compute budget",
        stem="parameters_vs_compute_budget_points", cmap_name="magma_r",
        **common,
    )
    _plot_budget_minimum_scatter(
        optima,
        x_column="training_budget_flops", y_column="env_steps",
        color_column="trainable_params",
        xlabel="Compute budget (FLOPs)", ylabel="Environment samples (steps)",
        colorbar_label="Trainable parameters",
        title="Empirical loss-optimal samples vs compute budget",
        stem="samples_vs_compute_budget_points", cmap_name="viridis",
        **common,
    )


def plot_parameter_checkpoints(
    checkpoints: pd.DataFrame,
    metric: str,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    if checkpoints.empty:
        return
    values = sorted(checkpoints["checkpoint_env_steps"].unique())
    colors = plt.get_cmap("plasma")(np.linspace(0.10, 0.88, len(values)))
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    checkpoint_handles: list[Line2D] = []
    for checkpoint, color in zip(values, colors):
        rows = checkpoints[checkpoints["checkpoint_env_steps"] == checkpoint]
        for status in RUN_STATUS_ORDER:
            status_rows = rows.loc[rows["run_status"].eq(status)]
            if status_rows.empty:
                continue
            ax.scatter(
                status_rows["trainable_params"], status_rows["loss"],
                color=color, marker=RUN_STATUS_MARKERS[status], alpha=0.76,
                s=54 if status == "ongoing" else 42,
                edgecolors="black", linewidths=0.3,
            )
        checkpoint_handles.append(
            Line2D(
                [], [], linestyle="none", marker="o", color=color, markersize=7,
                label=f"~{_format_count(float(checkpoint))} env steps",
            )
        )
    _style_axis(ax, "Trainable parameters", metric)
    ax.set_title("CRL loss vs parameters at observed checkpoints — no fit")
    checkpoint_legend = ax.legend(
        handles=checkpoint_handles, title="Checkpoint", frameon=False, loc="lower left"
    )
    ax.add_artist(checkpoint_legend)
    ax.legend(
        handles=_run_status_handles(checkpoints), title="Run state",
        frameon=False, loc="upper right",
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "loss_vs_parameters_checkpoints", formats)


def _plot_isoflop_profile(
    observations: pd.DataFrame,
    metric: str,
    output_dir: Path,
    formats: Sequence[str],
    *,
    title: str,
    stem: str,
) -> None:
    if observations.empty:
        return
    budgets = sorted(observations["isoflop_budget"].unique())
    colors = plt.get_cmap("plasma")(np.linspace(0.08, 0.90, len(budgets)))
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    budget_handles: list[Line2D] = []
    for budget, color in zip(budgets, colors):
        rows = observations.loc[observations["isoflop_budget"].eq(budget)]
        profile = (
            rows.groupby("trainable_params", as_index=False)["loss"]
            .median()
            .sort_values("trainable_params")
        )
        ax.plot(
            profile["trainable_params"], profile["loss"], color=color,
            linewidth=1.5, alpha=0.82,
        )
        for status in RUN_STATUS_ORDER:
            status_rows = rows.loc[rows["run_status"].eq(status)]
            if status_rows.empty:
                continue
            ax.scatter(
                status_rows["trainable_params"], status_rows["loss"],
                color=color, marker=RUN_STATUS_MARKERS[status],
                s=58 if status == "ongoing" else 44,
                edgecolors="black", linewidths=0.35, zorder=3,
            )
        budget_handles.append(
            Line2D(
                [], [], color=color, marker="o", markersize=6, linewidth=1.5,
                label=f"{_format_flops(float(budget))} (n={len(rows)})",
            )
        )
    _style_axis(ax, "Trainable parameters", metric)
    ax.set_title(title)
    budget_legend = ax.legend(
        handles=budget_handles, title="Compute budget", frameon=False,
        loc="upper right", ncol=2 if len(budget_handles) > 10 else 1,
        fontsize="small",
    )
    ax.add_artist(budget_legend)
    ax.legend(
        handles=_run_status_handles(observations), title="Run state",
        frameon=False, loc="lower left",
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, stem, formats)


def plot_isoflop_profiles(
    observations: pd.DataFrame,
    metric: str,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """Plot the aggregate isoFLOP profile across all available widths."""
    _plot_isoflop_profile(
        observations,
        metric,
        output_dir,
        formats,
        title="CRL isoFLOP profiles — checkpoint interpolation, no fit",
        stem="loss_vs_parameters_isoflops",
    )


def plot_isoflop_profiles_by_width(
    observations: pd.DataFrame,
    metric: str,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """Plot one isoFLOP figure per width so depth is the only size axis."""
    if observations.empty:
        return
    for width in sorted(observations["width"].dropna().astype(int).unique()):
        width_observations = observations.loc[observations["width"].eq(width)]
        _plot_isoflop_profile(
            width_observations,
            metric,
            output_dir,
            formats,
            title=(
                f"CRL isoFLOP profiles — width {width}, "
                "checkpoint interpolation, no fit"
            ),
            stem=f"loss_vs_parameters_isoflops_w{width}",
        )


def write_data(
    history: pd.DataFrame,
    checkpoints: pd.DataFrame,
    isoflops: pd.DataFrame,
    budget_minima: pd.DataFrame,
    budget_optima: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_parquet(output_dir / "scaling_history.parquet", index=False)
    history.to_csv(output_dir / "scaling_history.csv", index=False)
    checkpoints.to_csv(output_dir / "checkpoint_observations.csv", index=False)
    isoflops.to_csv(output_dir / "isoflop_observations.csv", index=False)
    budget_minima.to_csv(output_dir / "budget_minimum_observations.csv", index=False)
    budget_optima.to_csv(output_dir / "budget_optimal_observations.csv", index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B entity/project")
    parser.add_argument("--group-prefix", default=DEFAULT_GROUP_PREFIX)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--run-variant", choices=("all", "extended", "legacy"), default="all",
        help="Restrict plots to extended `_ext_v2` runs, legacy runs, or both",
    )
    parser.add_argument("--seeds", type=parse_int_list, default=(1, 2, 3))
    parser.add_argument("--checkpoints", type=parse_int_list, default=DEFAULT_CHECKPOINTS)
    parser.add_argument(
        "--isoflop-budgets", type=parse_int_list, default=DEFAULT_ISOFLOP_BUDGETS
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flop-profile", type=Path, default=DEFAULT_FLOP_PROFILE)
    parser.add_argument("--formats", type=parse_str_list, default=("png", "pdf"))
    parser.add_argument("--refresh", action="store_true", help="Refresh W&B metadata/history before plotting")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    unsupported_formats = sorted(set(args.formats) - {"png", "pdf", "svg"})
    if unsupported_formats:
        raise ValueError(f"Unsupported output formats: {', '.join(unsupported_formats)}")

    raw = fetch_history(
        project=args.project,
        group_prefix=args.group_prefix,
        seeds=args.seeds,
        metric=args.metric,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
        samples=args.samples,
        max_workers=args.max_workers,
    )
    if raw.empty:
        raise RuntimeError(f"No W&B history found for {args.project!r} and prefix {args.group_prefix!r}")
    history = normalize_history(raw, args.metric, args.group_prefix)
    history = filter_run_variant(history, args.run_variant)
    flop_profile = load_flop_profile(args.flop_profile)
    history = add_flop_estimates(history, flop_profile)
    checkpoints = select_checkpoint_observations(history, args.checkpoints)
    isoflops = interpolate_isoflop_observations(history, args.isoflop_budgets)
    budget_minima = select_budget_minimum_observations(
        history, args.isoflop_budgets
    )
    budget_optima = select_budget_optimal_observations(budget_minima)

    write_data(
        history, checkpoints, isoflops, budget_minima, budget_optima,
        args.output_dir,
    )
    plot_compute_trajectories(history, args.metric, args.output_dir, args.formats)
    plot_parameters(history, args.metric, args.output_dir, args.formats)
    plot_dataset_trajectories(history, args.metric, args.output_dir, args.formats)
    plot_parameters_vs_compute(history, args.metric, args.output_dir, args.formats)
    plot_samples_vs_compute(history, args.output_dir, args.formats)
    plot_budget_minimum_points(
        budget_minima, args.metric, args.output_dir, args.formats
    )
    plot_parameter_checkpoints(checkpoints, args.metric, args.output_dir, args.formats)
    plot_isoflop_profiles(isoflops, args.metric, args.output_dir, args.formats)
    plot_isoflop_profiles_by_width(
        isoflops, args.metric, args.output_dir, args.formats
    )

    print(
        f"Selected {history['run_id'].nunique()} {args.run_variant} runs and "
        f"{len(history)} valid history rows"
    )
    print(f"Latest environment step: {int(history['env_steps'].max()):,}")
    print(f"Wrote plots and tidy data to {args.output_dir.resolve()}")
    print(
        "Compute is XLA-estimated arithmetic FLOPs for optimizer updates and "
        "rollout-policy inference."
    )
    print(f"Excluded from FLOPs: {flop_profile['scope'].split('; ', 1)[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
