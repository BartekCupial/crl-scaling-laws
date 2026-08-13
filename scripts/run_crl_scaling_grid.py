#!/usr/bin/env python3
"""Generate or sequentially execute a Humanoid CRL scaling grid.

Dry-run is the default. Runs execute sequentially and stop at the first failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = REPO_ROOT / "configs" / "crl_scaling_humanoid_residual_v1.json"


@dataclass(frozen=True)
class Experiment:
    grid_id: str
    stage: str
    depth: int
    total_env_steps: int
    num_epochs: int
    wandb_group: str
    run_name: str


def load_plan(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        plan = json.load(file)
    if plan.get("schema_version") != 1:
        raise ValueError(f"Unsupported grid schema in {path}: {plan.get('schema_version')!r}")
    return plan


def build_experiments(plan: dict[str, Any]) -> list[Experiment]:
    experiments: list[Experiment] = []
    for item in plan["grid"]:
        stage_name = item["stage"]
        # The held-out run has main-run settings but remains a separate launch gate.
        stage_config = plan["stages"]["main" if stage_name == "heldout" else stage_name]
        depth = int(item["depth"])
        experiments.append(
            Experiment(
                grid_id=item["id"],
                stage=stage_name,
                depth=depth,
                total_env_steps=int(stage_config["total_env_steps"]),
                num_epochs=int(stage_config["num_epochs"]),
                wandb_group=stage_config["wandb_group"],
                run_name=stage_config["run_name_template"].format(depth=depth),
            )
        )
    validate_experiments(experiments)
    return experiments


def validate_experiments(experiments: Sequence[Experiment]) -> None:
    expected_ids = [
        "S04",
        "S08",
        "S16",
        "S32",
        "S64",
        "M04",
        "M08",
        "M16",
        "M32",
        "M64",
    ]
    actual_ids = [experiment.grid_id for experiment in experiments]
    if actual_ids != expected_ids:
        raise ValueError(f"Grid order changed: expected {expected_ids}, got {actual_ids}")
    run_names = [experiment.run_name for experiment in experiments]
    if len(set(run_names)) != len(run_names):
        raise ValueError("Run names must be unique")


def select_experiments(
    experiments: Sequence[Experiment], stage: str, start_at: str | None, stop_after: str | None
) -> list[Experiment]:
    if stage == "smoke":
        selected = [experiment for experiment in experiments if experiment.stage == "smoke"]
    elif stage == "main":
        selected = [experiment for experiment in experiments if experiment.stage == "main"]
    elif stage == "heldout":
        selected = [experiment for experiment in experiments if experiment.stage == "heldout"]
    elif stage == "pre-heldout":
        selected = [experiment for experiment in experiments if experiment.stage != "heldout"]
    else:
        selected = list(experiments)

    ids = [experiment.grid_id for experiment in selected]
    if start_at:
        if start_at not in ids:
            raise ValueError(f"--start-at {start_at} is not in selected stage {stage}: {ids}")
        selected = selected[ids.index(start_at) :]
        ids = [experiment.grid_id for experiment in selected]
    if stop_after:
        if stop_after not in ids:
            raise ValueError(f"--stop-after {stop_after} is not in selected range: {ids}")
        selected = selected[: ids.index(stop_after) + 1]
    return selected


def flag_name(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def runner_environment() -> dict[str, str]:
    env = dict(os.environ)
    active_venv = env.get("VIRTUAL_ENV")
    expected_venv = REPO_ROOT / ".venv"
    if active_venv and Path(active_venv).resolve() != expected_venv.resolve():
        env.pop("VIRTUAL_ENV")
    return env


def build_command(
    runner: Sequence[str],
    fixed_args: dict[str, Any],
    experiment: Experiment,
    run_dir: Path,
    slurm_job_id: str | None = None,
) -> list[str]:
    args = dict(fixed_args)
    args.update(
        {
            "total_env_steps": experiment.total_env_steps,
            "num_epochs": experiment.num_epochs,
            "critic_depth": experiment.depth,
            "actor_depth": experiment.depth,
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
    if slurm_job_id:
        command = [
            "srun",
            f"--jobid={slurm_job_id}",
            "--overlap",
            "--nodes=1",
            "--ntasks=1",
            *command,
        ]
    return command


def preflight_runner(runner: Sequence[str], cwd: Path) -> None:
    result = subprocess.run(
        [*runner, "--help"],
        cwd=cwd,
        env=runner_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Trainer preflight failed. The scaling trainer is not installed at the configured runner.\n"
            f"Command: {shlex.join([*runner, '--help'])}\n"
            f"Output:\n{result.stdout.strip()}"
        )
def git_metadata(cwd: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    ).stdout
    return {
        "commit": git("rev-parse", "HEAD"),
        "tracked_patch_sha256": hashlib.sha256(diff).hexdigest(),
        "status": git("status", "--short"),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_grid(experiments: Iterable[Experiment]) -> None:
    print("ID   stage     depth  env steps     epochs  run name")
    print("---  --------  -----  ------------  ------  ------------------------------------")
    for experiment in experiments:
        print(
            f"{experiment.grid_id:<4} {experiment.stage:<9} {experiment.depth:>5}  "
            f"{experiment.total_env_steps:>12,}  {experiment.num_epochs:>6}  {experiment.run_name}"
        )


def run_sequentially(
    plan_path: Path,
    plan: dict[str, Any],
    experiments: Sequence[Experiment],
    runner: Sequence[str],
    run_root: Path,
    slurm_job_id: str | None,
    prediction_artifact: Path | None,
) -> None:
    if any(experiment.stage == "heldout" for experiment in experiments):
        if prediction_artifact is None or not prediction_artifact.is_file():
            raise RuntimeError(
                "M64 is a held-out prediction test. Supply --prediction-artifact pointing to the frozen "
                "depth-64 prediction artifact created after M32."
            )

    existing = [run_root / experiment.run_name for experiment in experiments if (run_root / experiment.run_name).exists()]
    if existing:
        paths = "\n".join(f"  - {path}" for path in existing)
        raise RuntimeError(f"Refusing to overwrite existing run directories:\n{paths}")

    preflight_runner(runner, REPO_ROOT)
    metadata = git_metadata(REPO_ROOT)
    plan_hash = file_sha256(plan_path)
    prediction_hash = file_sha256(prediction_artifact) if prediction_artifact else None
    run_root.mkdir(parents=True, exist_ok=True)

    child: subprocess.Popen[str] | None = None

    def forward_signal(signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            # The trainer owns epoch-boundary checkpointing; forward the signal so it can shut down cleanly.
            child.send_signal(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    for index, experiment in enumerate(experiments, start=1):
        run_dir = run_root / experiment.run_name
        run_dir.mkdir(parents=False, exist_ok=False)
        command = build_command(runner, plan["fixed_args"], experiment, run_dir, slurm_job_id)
        launch_record = {
            "grid_id": experiment.grid_id,
            "queue_position": index,
            "queue_length": len(experiments),
            "experiment": experiment.__dict__,
            "command": command,
            "command_shell": shlex.join(command),
            "grid_plan": str(plan_path),
            "grid_plan_sha256": plan_hash,
            "prediction_artifact": str(prediction_artifact) if prediction_artifact else None,
            "prediction_artifact_sha256": prediction_hash,
            "git": metadata,
        }
        (run_dir / "launch.json").write_text(json.dumps(launch_record, indent=2) + "\n", encoding="utf-8")

        print(f"\n[{index}/{len(experiments)}] {experiment.grid_id}: {experiment.run_name}", flush=True)
        print(shlex.join(command), flush=True)
        log_path = run_dir / "launcher.log"
        with log_path.open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env={**runner_environment(), "CRL_SCALING_RUN_ID": experiment.grid_id},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert child.stdout is not None
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = child.wait()
            child = None
        if return_code != 0:
            raise RuntimeError(
                f"{experiment.grid_id} failed with exit code {return_code}; stopping the sequential queue. "
                f"See {log_path}."
            )
        (run_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="Checked-in grid configuration")
    parser.add_argument(
        "--stage",
        choices=("smoke", "main", "heldout", "pre-heldout", "all"),
        default="smoke",
        help="Grid slice to show or run (default: smoke)",
    )
    parser.add_argument("--start-at", help="Resume selection at this grid ID, e.g. S16")
    parser.add_argument("--stop-after", help="Stop selection after this grid ID")
    parser.add_argument(
        "--runner",
        help="Override the runner as one shell-like string, e.g. 'uv run --no-sync train.py'",
    )
    parser.add_argument("--run-root", type=Path, help="Override the local run root")
    parser.add_argument("--slurm-job-id", help="Wrap each experiment in a sequential srun step for this allocation")
    parser.add_argument(
        "--prediction-artifact",
        type=Path,
        help="Frozen M64 prediction artifact; mandatory when executing the held-out run",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute sequentially. Without this flag the script only prints the grid and commands.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)
    experiments = build_experiments(plan)
    selected = select_experiments(experiments, args.stage, args.start_at, args.stop_after)
    runner = shlex.split(args.runner) if args.runner else [str(item) for item in plan["runner"]]
    run_root = (args.run_root or (REPO_ROOT / plan["run_root"])).resolve()

    display_grid(selected)
    print("\nCommands:")
    for experiment in selected:
        command = build_command(
            runner,
            plan["fixed_args"],
            experiment,
            run_root / experiment.run_name,
            args.slurm_job_id,
        )
        print(f"\n# {experiment.grid_id}")
        print(shlex.join(command))

    if not args.execute:
        family = "SimBa" if plan["fixed_args"].get("use_simba") == 1 else "residual"
        print(f"\nDry run only. Add --execute to launch the selected {family} grid sequentially.")
        return 0

    run_sequentially(
        plan_path=plan_path,
        plan=plan,
        experiments=selected,
        runner=runner,
        run_root=run_root,
        slurm_job_id=args.slurm_job_id,
        prediction_artifact=args.prediction_artifact.resolve() if args.prediction_artifact else None,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
