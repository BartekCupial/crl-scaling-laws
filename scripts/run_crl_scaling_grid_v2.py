#!/usr/bin/env python3
"""Plan and execute the expanded Humanoid CRL scaling grid on multiple GPUs.

The scientific queue is seed-first.  Within each seed, the cheapest runs are
launched first.  Smoke timings order seed 1; each completed main seed then
provides the observed timings used to order the next seed.  Two approximately
iso-parameter frontier architectures remain behind a frozen-prediction gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_crl_scaling_grid import (  # noqa: E402
    REPO_ROOT,
    file_sha256,
    flag_name,
    git_metadata,
    preflight_runner,
    runner_environment,
)
from run_crl_scaling_grid_4gpu import ProcessRegistry, default_gpu_ids, parse_gpu_ids  # noqa: E402


DEFAULT_PLAN = REPO_ROOT / "configs" / "crl_scaling_humanoid_v2.json"
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Experiment:
    grid_id: str
    stage: str
    model_family: str
    width: int
    depth: int
    seed: int
    total_env_steps: int
    num_epochs: int
    expected_trainable_params: int
    wandb_group: str
    run_name: str

    @property
    def architecture_key(self) -> tuple[str, int, int]:
        return self.model_family, self.width, self.depth


@dataclass(frozen=True)
class RunResult:
    experiment: Experiment
    gpu: str
    return_code: int
    runtime_seconds: float
    log_path: Path


def load_plan(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        plan = json.load(file)
    if plan.get("schema_version") != 2:
        raise ValueError(f"Unsupported v2 grid schema in {path}: {plan.get('schema_version')!r}")
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    axes = plan["axes"]
    if plan["fixed_args"].get("env_id") != "humanoid":
        raise ValueError("The exact v2 parameter counts are specific to Humanoid")
    if axes["model_families"] != ["simba"]:
        raise ValueError("The v2 scientific queue must contain only the SimBa model family")
    if axes["widths"] != [128, 256, 512]:
        raise ValueError("The v2 width grid must be [128, 256, 512]")
    expected_depths = {
        "128": [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192],
        "256": [4, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        "512": [4, 8, 12, 16, 24, 32],
    }
    if axes["depths_by_width"] != expected_depths:
        raise ValueError("The v2 width/depth grid changed from the locked 27-architecture design")
    if axes["seeds"] != [1, 2, 3]:
        raise ValueError("The v2 seed order must be [1, 2, 3]")
    expected_heldout = [
        {"width": 256, "depth": 128},
        {"width": 512, "depth": 32},
    ]
    if axes["heldout_architectures"] != expected_heldout:
        raise ValueError("The two approximately iso-parameter frontier models must remain held out")


def expected_trainable_params(model_family: str, width: int, depth: int) -> int:
    """Exact Humanoid parameter count, including actor, both critics, and alpha."""
    if model_family == "residual":
        return 730 * width + 163 + 3 * depth * width**2 + 9 * depth * width
    if model_family == "simba":
        return (
            728 * width
            + 163
            + (17 * depth - 16) * width**2
            + (13 * depth - 10) * width
        )
    raise ValueError(f"Unknown model family: {model_family}")


def family_code(model_family: str) -> str:
    return {"residual": "res", "simba": "sim"}[model_family]


def stage_training_budget(
    stage_config: dict[str, Any], trainable_params: int
) -> tuple[int, int]:
    """Resolve fixed or log-parameter-interpolated steps and logging epochs."""
    schedule = stage_config.get("step_schedule")
    if schedule is None:
        return int(stage_config["total_env_steps"]), int(stage_config["num_epochs"])
    if schedule.get("type") != "log_parameter_interpolation":
        raise ValueError(f"Unsupported step schedule: {schedule.get('type')!r}")
    small_params = int(schedule["small_model_params"])
    large_params = int(schedule["large_model_params"])
    small_steps = int(schedule["small_model_steps"])
    large_steps = int(schedule["large_model_steps"])
    if not 0 < small_params < large_params or not 0 < large_steps <= small_steps:
        raise ValueError("Invalid log-parameter step schedule endpoints")
    fraction = (
        (math.log(trainable_params) - math.log(small_params))
        / (math.log(large_params) - math.log(small_params))
    )
    fraction = min(1.0, max(0.0, fraction))
    raw_steps = math.exp(
        math.log(small_steps)
        + fraction * (math.log(large_steps) - math.log(small_steps))
    )
    rounding = int(schedule.get("round_to_steps", 1))
    total_env_steps = int(round(raw_steps / rounding) * rounding)
    max_steps_per_epoch = int(schedule["max_env_steps_per_epoch"])
    minimum_epochs = int(schedule.get("minimum_num_epochs", 1))
    num_epochs = max(minimum_epochs, math.ceil(total_env_steps / max_steps_per_epoch))
    return total_env_steps, num_epochs


def make_experiment(
    plan: dict[str, Any],
    stage: str,
    model_family: str,
    width: int,
    depth: int,
    seed: int,
) -> Experiment:
    stage_config = plan["stages"][stage]
    code = family_code(model_family)
    params = expected_trainable_params(model_family, width, depth)
    total_env_steps, num_epochs = stage_training_budget(stage_config, params)
    budget = "smoke1m" if stage == "smoke" else f"{total_env_steps // 1_000_000}m_ext"
    run_name = f"humanoid_{code}_w{width:03d}_d{depth:03d}_s{seed}_{budget}_v2"
    stage_code = {"smoke": "S", "fit": "M", "heldout": "H"}[stage]
    return Experiment(
        grid_id=f"{stage_code}-S{seed}-{code.upper()}-W{width:03d}-D{depth:03d}",
        stage=stage,
        model_family=model_family,
        width=width,
        depth=depth,
        seed=seed,
        total_env_steps=total_env_steps,
        num_epochs=num_epochs,
        expected_trainable_params=params,
        wandb_group=stage_config["wandb_group_template"].format(seed=seed),
        run_name=run_name,
    )


def build_experiments(plan: dict[str, Any]) -> list[Experiment]:
    axes = plan["axes"]
    families = axes["model_families"]
    widths = axes["widths"]
    seeds = axes["seeds"]
    depths_by_width = {
        int(width): depths for width, depths in axes["depths_by_width"].items()
    }
    architectures = [
        (width, depth)
        for width in widths
        for depth in depths_by_width[width]
    ]
    heldout_architectures = {
        (item["width"], item["depth"]) for item in axes["heldout_architectures"]
    }
    fitting_architectures = [
        architecture for architecture in architectures if architecture not in heldout_architectures
    ]

    experiments: list[Experiment] = []
    smoke_seed = seeds[0]
    for family in families:
        for width, depth in architectures:
            experiments.append(make_experiment(plan, "smoke", family, width, depth, smoke_seed))
    for seed in seeds:
        for family in families:
            for width, depth in fitting_architectures:
                experiments.append(make_experiment(plan, "fit", family, width, depth, seed))
    for seed in seeds:
        for family in families:
            for item in axes["heldout_architectures"]:
                experiments.append(
                    make_experiment(plan, "heldout", family, item["width"], item["depth"], seed)
                )

    run_names = [experiment.run_name for experiment in experiments]
    grid_ids = [experiment.grid_id for experiment in experiments]
    if len(set(run_names)) != len(run_names) or len(set(grid_ids)) != len(grid_ids):
        raise ValueError("Every v2 experiment must have a unique run name and grid ID")
    return experiments


def experiment_batches(
    experiments: Sequence[Experiment], phase: str, seeds: Sequence[int]
) -> list[tuple[str, list[Experiment]]]:
    smoke = [experiment for experiment in experiments if experiment.stage == "smoke"]
    fits = {
        seed: [
            experiment
            for experiment in experiments
            if experiment.stage == "fit" and experiment.seed == seed
        ]
        for seed in seeds
    }
    heldout = {
        seed: [
            experiment
            for experiment in experiments
            if experiment.stage == "heldout" and experiment.seed == seed
        ]
        for seed in seeds
    }
    mapping = {
        "smoke": [("smoke-seed-1", smoke)],
        "seed1": [("fit-seed-1", fits[1])],
        "seed2": [("fit-seed-2", fits[2])],
        "seed3": [("fit-seed-3", fits[3])],
        "fit": [(f"fit-seed-{seed}", fits[seed]) for seed in seeds],
        "pre-heldout": [("smoke-seed-1", smoke)]
        + [(f"fit-seed-{seed}", fits[seed]) for seed in seeds],
        "heldout": [(f"heldout-seed-{seed}", heldout[seed]) for seed in seeds],
    }
    return mapping[phase]


def prerequisite_experiments(
    experiments: Sequence[Experiment], phase: str
) -> list[Experiment]:
    smoke = [experiment for experiment in experiments if experiment.stage == "smoke"]
    fit_seed_1 = [
        experiment for experiment in experiments if experiment.stage == "fit" and experiment.seed == 1
    ]
    fit_seed_2 = [
        experiment for experiment in experiments if experiment.stage == "fit" and experiment.seed == 2
    ]
    all_fitting = [experiment for experiment in experiments if experiment.stage == "fit"]
    if phase in ("smoke", "pre-heldout"):
        return []
    if phase in ("seed1", "fit"):
        return smoke
    if phase == "seed2":
        return smoke + fit_seed_1
    if phase == "seed3":
        return smoke + fit_seed_1 + fit_seed_2
    if phase == "heldout":
        return all_fitting
    raise ValueError(f"Unknown phase: {phase}")


def validate_prerequisites(
    experiments: Sequence[Experiment], phase: str, run_root: Path
) -> None:
    missing = [
        experiment
        for experiment in prerequisite_experiments(experiments, phase)
        if not (run_root / experiment.run_name / "COMPLETE").is_file()
    ]
    if missing:
        preview = "\n".join(f"  - {experiment.grid_id}" for experiment in missing[:10])
        suffix = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise RuntimeError(
            f"Phase {phase} is gated by {len(missing)} incomplete prerequisite run(s):\n"
            f"{preview}{suffix}"
        )


def experiment_lookup(experiments: Sequence[Experiment]) -> dict[tuple[str, int, str, int, int], Experiment]:
    return {
        (experiment.stage, experiment.seed, *experiment.architecture_key): experiment
        for experiment in experiments
    }


def runtime_record_path(run_root: Path, experiment: Experiment) -> Path:
    return run_root / experiment.run_name / "runtime.json"


def read_runtime(run_root: Path, experiment: Experiment) -> float | None:
    path = runtime_record_path(run_root, experiment)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))["runtime_seconds"]
        return float(value)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def reference_experiment(
    experiment: Experiment,
    lookup: dict[tuple[str, int, str, int, int], Experiment],
) -> Experiment | None:
    family, width, depth = experiment.architecture_key
    if experiment.stage == "smoke":
        return None
    if experiment.stage == "fit" and experiment.seed == 1:
        return lookup.get(("smoke", 1, family, width, depth))
    if experiment.stage == "fit":
        return lookup.get(("fit", experiment.seed - 1, family, width, depth))
    if experiment.stage == "heldout" and experiment.seed == 1:
        return lookup.get(("smoke", 1, family, width, depth))
    return lookup.get(("heldout", experiment.seed - 1, family, width, depth))


def sort_experiments(
    experiments: Sequence[Experiment],
    all_experiments: Sequence[Experiment],
    run_root: Path,
) -> list[Experiment]:
    lookup = experiment_lookup(all_experiments)

    def key(experiment: Experiment) -> tuple[int, float, int, str]:
        reference = reference_experiment(experiment, lookup)
        observed_runtime = read_runtime(run_root, reference) if reference else None
        if observed_runtime is not None:
            return (0, observed_runtime, experiment.expected_trainable_params, experiment.run_name)
        return (1, float(experiment.expected_trainable_params), experiment.depth, experiment.run_name)

    return sorted(experiments, key=key)


def ordering_basis(
    experiment: Experiment,
    all_experiments: Sequence[Experiment],
    run_root: Path,
) -> str:
    reference = reference_experiment(experiment, experiment_lookup(all_experiments))
    observed_runtime = read_runtime(run_root, reference) if reference else None
    if observed_runtime is not None:
        return f"observed {observed_runtime / 3600:.2f}h from {reference.grid_id}"
    return f"parameter proxy {experiment.expected_trainable_params:,}"


def build_command(
    runner: Sequence[str],
    fixed_args: dict[str, Any],
    experiment: Experiment,
    run_dir: Path,
) -> list[str]:
    args = dict(fixed_args)
    args.update(
        {
            "seed": experiment.seed,
            "use_simba": 1 if experiment.model_family == "simba" else 0,
            "critic_network_width": experiment.width,
            "actor_network_width": experiment.width,
            "critic_depth": experiment.depth,
            "actor_depth": experiment.depth,
            "total_env_steps": experiment.total_env_steps,
            "num_epochs": experiment.num_epochs,
            "wandb_group": experiment.wandb_group,
            "wandb_dir": str(run_dir),
            "exp_name": experiment.run_name,
        }
    )
    command = list(runner)
    for name, value in args.items():
        if isinstance(value, bool):
            command.append(flag_name(name) if value else f"--no-{name.replace('_', '-')}")
        else:
            command.extend((flag_name(name), str(value)))
    return command


def classify_existing(
    experiments: Sequence[Experiment], run_root: Path, resume_completed: bool,
    skip_existing: bool = False,
) -> tuple[list[Experiment], list[Experiment]]:
    pending: list[Experiment] = []
    completed: list[Experiment] = []
    conflicts: list[Path] = []
    for experiment in experiments:
        run_dir = run_root / experiment.run_name
        if not run_dir.exists():
            pending.append(experiment)
        elif skip_existing or (resume_completed and (run_dir / "COMPLETE").is_file()):
            completed.append(experiment)
        else:
            conflicts.append(run_dir)
    if conflicts:
        paths = "\n".join(f"  - {path}" for path in conflicts)
        raise RuntimeError(f"Refusing to overwrite existing or incomplete run directories:\n{paths}")
    return pending, completed


def safe_gpu_label(gpu: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in gpu)


def run_one(
    experiment: Experiment,
    gpu: str,
    plan_path: Path,
    plan_hash: str,
    runner: Sequence[str],
    fixed_args: dict[str, Any],
    run_root: Path,
    prediction_artifact: Path | None,
    prediction_hash: str | None,
    metadata: dict[str, Any],
    registry: ProcessRegistry,
) -> RunResult:
    run_dir = run_root / experiment.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    command = build_command(runner, fixed_args, experiment, run_dir)
    log_path = run_dir / "launcher.log"
    launch_record = {
        "experiment": asdict(experiment),
        "gpu": gpu,
        "command": command,
        "command_shell": shlex.join(command),
        "grid_plan": str(plan_path),
        "grid_plan_sha256": plan_hash,
        "prediction_artifact": str(prediction_artifact) if prediction_artifact else None,
        "prediction_artifact_sha256": prediction_hash,
        "git": metadata,
    }
    (run_dir / "launch.json").write_text(json.dumps(launch_record, indent=2) + "\n", encoding="utf-8")

    environment = runner_environment()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "CRL_SCALING_RUN_ID": experiment.grid_id,
        }
    )
    with PRINT_LOCK:
        print(
            f"START GPU {gpu}: {experiment.grid_id} {experiment.run_name} "
            f"({experiment.expected_trainable_params:,} params)",
            flush=True,
        )

    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            # Keep tmux/terminal foreground-group signals from reaching a
            # training child directly.  Deliberate launcher shutdowns are
            # forwarded through ProcessRegistry instead.
            start_new_session=True,
        )
        registry.add(process)
        try:
            return_code = process.wait()
        finally:
            registry.remove(process)
    runtime_seconds = time.monotonic() - started
    runtime_record = {
        "runtime_seconds": runtime_seconds,
        "return_code": return_code,
        "gpu": gpu,
    }
    runtime_record_path(run_root, experiment).write_text(
        json.dumps(runtime_record, indent=2) + "\n", encoding="utf-8"
    )
    if return_code == 0:
        (run_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
    else:
        (run_dir / "FAILED").write_text(f"exit_code={return_code}\n", encoding="utf-8")
    with PRINT_LOCK:
        status = "COMPLETE" if return_code == 0 else f"FAILED ({return_code})"
        print(
            f"{status} GPU {gpu}: {experiment.grid_id} in {runtime_seconds / 3600:.2f}h; {log_path}",
            flush=True,
        )
    return RunResult(experiment, gpu, return_code, runtime_seconds, log_path)


def run_batch(
    experiments: Sequence[Experiment],
    gpu_ids: Sequence[str],
    run_one_kwargs: dict[str, Any],
    shutdown_requested: threading.Event | None = None,
) -> bool:
    if not experiments:
        return True
    pending: queue.Queue[Experiment] = queue.Queue()
    initial_count = min(len(experiments), len(gpu_ids))
    for experiment in experiments[initial_count:]:
        pending.put(experiment)
    results: list[RunResult] = []
    results_lock = threading.Lock()

    def worker(gpu: str, first_experiment: Experiment) -> None:
        experiment: Experiment | None = first_experiment
        while experiment is not None and not (
            shutdown_requested is not None and shutdown_requested.is_set()
        ):
            result = run_one(experiment=experiment, gpu=gpu, **run_one_kwargs)
            with results_lock:
                results.append(result)
            if shutdown_requested is not None and shutdown_requested.is_set():
                break
            try:
                next_experiment = pending.get_nowait()
            except queue.Empty:
                experiment = None
            else:
                experiment = next_experiment

    threads = [
        threading.Thread(
            target=worker,
            args=(gpu_ids[index], experiments[index]),
            name=f"gpu-{safe_gpu_label(gpu_ids[index])}",
        )
        for index in range(initial_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failures = [result for result in results if result.return_code != 0]
    if failures:
        for failure in failures:
            print(
                f"FAILED {failure.experiment.grid_id} on GPU {failure.gpu}: {failure.log_path}",
                file=sys.stderr,
            )
        print(f"{len(failures)} experiment(s) failed; other GPU queues continued.", file=sys.stderr)
        return False
    return len(results) == len(experiments)


def display_batches(
    batches: Sequence[tuple[str, Sequence[Experiment]]],
    all_experiments: Sequence[Experiment],
    run_root: Path,
    gpu_ids: Sequence[str],
) -> None:
    total = sum(len(experiments) for _label, experiments in batches)
    print(f"Selected jobs: {total}; GPUs: {', '.join(gpu_ids)}")
    for batch_index, (label, experiments) in enumerate(batches, start=1):
        ordered = sort_experiments(experiments, all_experiments, run_root)
        print(f"\nBatch {batch_index}: {label} ({len(ordered)} jobs)")
        for position, experiment in enumerate(ordered, start=1):
            placement = (
                f"GPU {gpu_ids[position - 1]}"
                if position <= len(gpu_ids)
                else "first free GPU"
            )
            print(
                f"  {position:>2}. {experiment.grid_id:<25} {placement:<16} "
                f"{ordering_basis(experiment, all_experiments, run_root)}"
            )
        if batch_index != len(batches):
            print("  barrier: the batch must finish successfully before the next seed starts")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--phase",
        choices=("smoke", "seed1", "seed2", "seed3", "fit", "pre-heldout", "heldout"),
        default="smoke",
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpu_ids,
        default=default_gpu_ids(),
        help="Comma-separated GPU IDs (default: first four exposed devices)",
    )
    parser.add_argument("--prediction-artifact", type=Path)
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Skip directories containing COMPLETE; incomplete directories still cause an error",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip every existing run directory, including active/failed runs",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Launch at most this many new jobs, for safely chunking the queue across allocations",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError("--max-jobs must be a positive integer")
    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)
    all_experiments = build_experiments(plan)
    batches = experiment_batches(all_experiments, args.phase, plan["axes"]["seeds"])
    run_root = (REPO_ROOT / plan["run_root"]).resolve()
    prediction_artifact = args.prediction_artifact.resolve() if args.prediction_artifact else None
    contains_heldout = any(
        experiment.stage == "heldout" for _label, batch in batches for experiment in batch
    )
    if contains_heldout and (prediction_artifact is None or not prediction_artifact.is_file()):
        raise ValueError("Held-out frontier runs require --prediction-artifact pointing to a frozen file")

    display_batches(batches, all_experiments, run_root, args.gpus)
    if not args.execute:
        print("\nDry run only. Add --execute to launch this queue.")
        return 0

    validate_prerequisites(all_experiments, args.phase, run_root)
    selected = [experiment for _label, batch in batches for experiment in batch]
    pending, completed = classify_existing(
        selected, run_root, args.resume_completed, args.skip_existing
    )
    if completed:
        print(f"Skipping {len(completed)} completed experiment(s).", flush=True)
    pending_names = {experiment.run_name for experiment in pending}
    runner = [str(item) for item in plan["runner"]]
    preflight_runner(runner, REPO_ROOT)
    run_root.mkdir(parents=True, exist_ok=True)
    registry = ProcessRegistry()
    metadata = git_metadata(REPO_ROOT)
    run_one_kwargs = {
        "plan_path": plan_path,
        "plan_hash": file_sha256(plan_path),
        "runner": runner,
        "fixed_args": plan["fixed_args"],
        "run_root": run_root,
        "prediction_artifact": prediction_artifact,
        "prediction_hash": file_sha256(prediction_artifact) if prediction_artifact else None,
        "metadata": metadata,
        "registry": registry,
    }
    interrupted = False
    shutdown_requested = threading.Event()

    def forward_signal(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        shutdown_requested.set()
        registry.signal_all(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    jobs_left = args.max_jobs
    for label, batch in batches:
        if interrupted:
            return 1
        remaining = [experiment for experiment in batch if experiment.run_name in pending_names]
        ordered = sort_experiments(remaining, all_experiments, run_root)
        if jobs_left is not None:
            ordered = ordered[:jobs_left]
        print(f"\nLaunching {label}: {len(ordered)} pending job(s)", flush=True)
        for position, experiment in enumerate(ordered, start=1):
            print(
                f"  {position:>2}. {experiment.grid_id}: "
                f"{ordering_basis(experiment, all_experiments, run_root)}",
                flush=True,
            )
        succeeded = run_batch(
            ordered, args.gpus, run_one_kwargs,
            shutdown_requested=shutdown_requested,
        )
        if interrupted or not succeeded:
            return 1
        if jobs_left is not None:
            jobs_left -= len(ordered)
            if jobs_left == 0:
                print("Reached --max-jobs limit; remaining jobs were left untouched.", flush=True)
                return 0
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
