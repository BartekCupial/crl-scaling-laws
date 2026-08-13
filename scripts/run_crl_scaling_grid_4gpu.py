#!/usr/bin/env python3
"""Run the Humanoid SimBa grid across four GPUs at experiment level.

Each training process sees exactly one GPU.  Smoke and main are separate
batches so that no 100M-step run starts before every smoke run succeeds.
"""

from __future__ import annotations

import argparse
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_crl_scaling_grid import (  # noqa: E402
    REPO_ROOT,
    Experiment,
    build_experiments,
    load_plan,
)


DEFAULT_PLAN = REPO_ROOT / "configs" / "crl_scaling_humanoid_simba_v1.json"
SINGLE_GPU_LAUNCHER = SCRIPT_DIR / "run_crl_scaling_grid.py"
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class RunResult:
    experiment: Experiment
    gpu: str
    return_code: int
    log_path: Path


class ProcessRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[str]] = set()

    def add(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes.add(process)

    def remove(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes.discard(process)

    def signal_all(self, signum: int) -> None:
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                process.send_signal(signum)


def parse_gpu_ids(value: str) -> list[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise argparse.ArgumentTypeError("at least one GPU ID is required")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError(f"GPU IDs must be unique: {gpu_ids}")
    return gpu_ids


def default_gpu_ids() -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices and visible_devices != "-1":
        gpu_ids = parse_gpu_ids(visible_devices)
        if len(gpu_ids) >= 4:
            return gpu_ids[:4]
    return ["0", "1", "2", "3"]


def experiment_batches(experiments: Sequence[Experiment], stage: str) -> list[list[Experiment]]:
    smoke = [experiment for experiment in experiments if experiment.stage == "smoke"]
    main = [experiment for experiment in experiments if experiment.stage == "main"]
    heldout = [experiment for experiment in experiments if experiment.stage == "heldout"]
    if stage == "smoke":
        return [smoke]
    if stage == "main":
        return [main]
    if stage == "pre-heldout":
        return [smoke, main]
    if stage == "heldout":
        return [heldout]
    if stage == "all":
        return [smoke, main, heldout]
    raise ValueError(f"Unknown stage: {stage}")


def single_run_command(
    experiment: Experiment,
    plan_path: Path,
    prediction_artifact: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(SINGLE_GPU_LAUNCHER),
        "--plan",
        str(plan_path),
        "--stage",
        experiment.stage,
        "--start-at",
        experiment.grid_id,
        "--stop-after",
        experiment.grid_id,
        "--execute",
    ]
    if experiment.stage == "heldout":
        if prediction_artifact is None:
            raise ValueError("The held-out M64 run requires --prediction-artifact")
        command.extend(("--prediction-artifact", str(prediction_artifact)))
    return command


def print_schedule(batches: Sequence[Sequence[Experiment]], gpu_ids: Sequence[str]) -> None:
    for batch_index, experiments in enumerate(batches, start=1):
        print(f"Batch {batch_index}: {experiments[0].stage}")
        for index, experiment in enumerate(experiments):
            if index < len(gpu_ids):
                placement = f"starts on GPU {gpu_ids[index]}"
            else:
                placement = "queued for the first free GPU"
            print(f"  {experiment.grid_id}: depth {experiment.depth:<2} {placement}")
        if batch_index != len(batches):
            print("  barrier: this entire batch must succeed before the next batch starts")


def safe_gpu_label(gpu: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in gpu)


def run_one(
    experiment: Experiment,
    gpu: str,
    plan_path: Path,
    prediction_artifact: Path | None,
    log_dir: Path,
    registry: ProcessRegistry,
) -> RunResult:
    command = single_run_command(experiment, plan_path, prediction_artifact)
    log_path = log_dir / f"{experiment.grid_id}_{experiment.run_name}_gpu{safe_gpu_label(gpu)}.log"
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )

    with PRINT_LOCK:
        print(f"START GPU {gpu}: {experiment.grid_id} {experiment.run_name}", flush=True)
        print(f"  console log: {log_path}", flush=True)

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
            start_new_session=True,
        )
        registry.add(process)
        try:
            return_code = process.wait()
        finally:
            registry.remove(process)

    with PRINT_LOCK:
        status = "COMPLETE" if return_code == 0 else f"FAILED ({return_code})"
        print(f"{status} GPU {gpu}: {experiment.grid_id}; log: {log_path}", flush=True)
    return RunResult(experiment, gpu, return_code, log_path)


def run_batch(
    experiments: Sequence[Experiment],
    gpu_ids: Sequence[str],
    plan_path: Path,
    prediction_artifact: Path | None,
    log_dir: Path,
    registry: ProcessRegistry,
) -> bool:
    pending: queue.Queue[Experiment] = queue.Queue()
    initial_count = min(len(experiments), len(gpu_ids))
    for experiment in experiments[initial_count:]:
        pending.put(experiment)

    stop_scheduling = threading.Event()
    results: list[RunResult] = []
    results_lock = threading.Lock()

    def worker(gpu: str, first_experiment: Experiment) -> None:
        experiment: Experiment | None = first_experiment
        while experiment is not None and not stop_scheduling.is_set():
            result = run_one(
                experiment,
                gpu,
                plan_path,
                prediction_artifact,
                log_dir,
                registry,
            )
            with results_lock:
                results.append(result)
            if result.return_code != 0:
                stop_scheduling.set()
                break
            try:
                next_experiment = pending.get_nowait()
            except queue.Empty:
                experiment = None
            else:
                if stop_scheduling.is_set():
                    pending.put(next_experiment)
                    experiment = None
                else:
                    experiment = next_experiment

    threads = [
        threading.Thread(
            target=worker,
            args=(gpu_ids[index], experiments[index]),
            name=f"gpu-{gpu_ids[index]}",
        )
        for index in range(initial_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failures = [result for result in results if result.return_code != 0]
    if failures:
        print("Failed experiments:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure.experiment.grid_id} on GPU {failure.gpu}: {failure.log_path}", file=sys.stderr)
        not_started = len(experiments) - len(results)
        if not_started:
            print(f"{not_started} experiment(s) were not started after the failure.", file=sys.stderr)
        return False
    return len(results) == len(experiments)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--stage",
        choices=("smoke", "main", "pre-heldout", "heldout", "all"),
        default="smoke",
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpu_ids,
        default=default_gpu_ids(),
        help="Comma-separated GPU IDs (default: first four devices exposed by the allocation)",
    )
    parser.add_argument("--prediction-artifact", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)
    experiments = build_experiments(plan)
    batches = experiment_batches(experiments, args.stage)
    prediction_artifact = args.prediction_artifact.resolve() if args.prediction_artifact else None
    if any(experiment.stage == "heldout" for batch in batches for experiment in batch):
        if prediction_artifact is None or not prediction_artifact.is_file():
            raise ValueError("--prediction-artifact must point to the frozen depth-64 prediction artifact")

    print(f"Plan: {plan_path}")
    print(f"GPUs: {', '.join(args.gpus)}")
    print_schedule(batches, args.gpus)
    if not args.execute:
        print("\nDry run only. Add --execute to launch this schedule.")
        return 0

    log_dir = (REPO_ROOT / plan["run_root"] / ".parallel_launcher" / plan_path.stem).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    registry = ProcessRegistry()

    interrupted = False

    def forward_signal(signum: int, _frame) -> None:
        nonlocal interrupted
        interrupted = True
        registry.signal_all(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    for batch in batches:
        if interrupted:
            return 1
        succeeded = run_batch(
            batch,
            args.gpus,
            plan_path,
            prediction_artifact,
            log_dir,
            registry,
        )
        if interrupted or not succeeded:
            return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
