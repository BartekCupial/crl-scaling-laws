#!/usr/bin/env python3
"""Warm-resume one interrupted extended run from its newest parameter checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def checkpoint_epoch(log_text: str, checkpoint_steps: int) -> int:
    matches = re.findall(
        r"epoch (\d+) out of \d+ complete.*?'training/envsteps': ([0-9.]+)",
        log_text,
    )
    exact = [int(epoch) for epoch, steps in matches if int(float(steps)) == checkpoint_steps]
    if not exact:
        raise ValueError(f"Cannot map checkpoint step {checkpoint_steps} to an epoch")
    return exact[-1]


def wandb_run_id(run_dir: Path) -> str:
    candidates = sorted((run_dir / "wandb").glob("run-*-*"))
    if not candidates:
        raise ValueError(f"No W&B run directory found under {run_dir}")
    return candidates[-1].name.rsplit("-", 1)[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "FAILED").is_file():
        raise ValueError(f"Expected FAILED marker in {run_dir}")
    launch = json.loads((run_dir / "launch.json").read_text(encoding="utf-8"))
    checkpoints = sorted(
        run_dir.glob("checkpoints/step_*.pkl"),
        key=lambda path: int(path.stem.removeprefix("step_")),
    )
    if not checkpoints:
        raise ValueError(f"No parameter checkpoints found in {run_dir}")
    checkpoint = checkpoints[-1]
    checkpoint_steps = int(checkpoint.stem.removeprefix("step_"))
    log_path = run_dir / "launcher.log"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    saved_epoch = checkpoint_epoch(log_text, checkpoint_steps)
    logged_epochs = [int(value) for value in re.findall(r"epoch (\d+) out of", log_text)]
    latest_logged_epoch = max(logged_epochs)
    command = [str(item) for item in launch["command"]]
    command.extend(
        [
            "--resume-checkpoint", str(checkpoint),
            "--resume-env-steps", str(checkpoint_steps),
            "--resume-epoch", str(saved_epoch + 1),
            "--wandb-step-offset", str(latest_logged_epoch - saved_epoch),
            "--wandb-run-id", wandb_run_id(run_dir),
        ]
    )
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\nWARM RESUME checkpoint={checkpoint} saved_epoch={saved_epoch} "
            f"latest_logged_epoch={latest_logged_epoch} gpu={args.gpu}\n"
        )
        log.flush()
        started = time.monotonic()
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=environment, text=True,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )

        def forward(signum: int, _frame: object) -> None:
            if process.poll() is None:
                process.send_signal(signum)

        signal.signal(signal.SIGINT, forward)
        signal.signal(signal.SIGTERM, forward)
        return_code = process.wait()
    continuation_seconds = time.monotonic() - started
    runtime_path = run_dir / "runtime.json"
    previous = json.loads(runtime_path.read_text(encoding="utf-8"))
    previous["runtime_seconds"] = float(previous.get("runtime_seconds", 0)) + continuation_seconds
    previous["continuation_seconds"] = continuation_seconds
    previous["return_code"] = return_code
    previous["gpu"] = args.gpu
    runtime_path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    if return_code == 0:
        (run_dir / "FAILED").unlink()
        (run_dir / "COMPLETE").write_text("ok (parameter-only warm resume)\n", encoding="utf-8")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
