#!/usr/bin/env python3
"""Profile model-training FLOPs for the Humanoid SimBa scaling grid with XLA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


# FLOP counts should not depend on accelerator availability.  Pinning the
# profiler to CPU also keeps this utility from taking a GPU away from a run.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlib  # noqa: E402
import optax  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simba import EncoderSimba  # noqa: E402
from train import Actor, TrainingState, Transition  # noqa: E402


DEFAULT_PLAN = REPO_ROOT / "configs" / "crl_scaling_humanoid_v2.json"
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "crl_scaling_humanoid_v2_flops.json"
OBS_DIM = 268
GOAL_DIM = 3
ACTION_SIZE = 17


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cost_totals(lowered: Any) -> dict[str, float]:
    analysis = lowered.cost_analysis()
    if isinstance(analysis, list):
        analyses = analysis
    else:
        analyses = [analysis]
    return {
        key: float(sum(item.get(key, 0.0) for item in analyses))
        for key in ("flops", "transcendentals", "bytes accessed")
    }


def profile_architecture(
    width: int,
    depth: int,
    *,
    batch_size: int,
    rollout_batch_size: int,
) -> dict[str, int]:
    actor = Actor(
        action_size=ACTION_SIZE,
        network_width=width,
        network_depth=depth,
        skip_connections=4,
        use_relu=0,
    )
    sa_encoder = EncoderSimba(
        network_width=width,
        network_depth=depth,
        skip_connections=4,
        use_relu=False,
    )
    g_encoder = EncoderSimba(
        network_width=width,
        network_depth=depth,
        skip_connections=4,
        use_relu=False,
    )
    actor_key, sa_key, goal_key, step_key = jax.random.split(jax.random.PRNGKey(0), 4)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, jnp.ones((1, OBS_DIM + GOAL_DIM))),
        tx=optax.adam(3e-4),
    )
    critic_state = TrainState.create(
        apply_fn=None,
        params={
            "sa_encoder": sa_encoder.init(
                sa_key, jnp.ones((1, OBS_DIM + ACTION_SIZE))
            ),
            "g_encoder": g_encoder.init(goal_key, jnp.ones((1, GOAL_DIM))),
        },
        tx=optax.adam(3e-4),
    )
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": jnp.asarray(0.0, dtype=jnp.float32)},
        tx=optax.adam(3e-4),
    )
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
    )
    transitions = Transition(
        observation=jnp.zeros((batch_size, OBS_DIM + GOAL_DIM)),
        action=jnp.zeros((batch_size, ACTION_SIZE)),
        reward=jnp.zeros((batch_size,)),
        discount=jnp.zeros((batch_size,)),
        extras={"future_state": jnp.zeros((batch_size, OBS_DIM))},
    )

    def actor_loss(actor_params, critic_params, log_alpha, batch, key):
        state = batch.observation[:, :OBS_DIM]
        goal = batch.extras["future_state"][:, :GOAL_DIM]
        observation = jnp.concatenate([state, goal], axis=1)
        means, log_stds = actor.apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(
            key, shape=means.shape, dtype=means.dtype
        )
        action = jnp.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(
            x_ts, loc=means, scale=stds
        ) - jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)
        sa_repr = sa_encoder.apply(
            critic_params["sa_encoder"], jnp.concatenate([state, action], axis=-1)
        )
        goal_repr = g_encoder.apply(critic_params["g_encoder"], goal)
        q_value = -jnp.sqrt(jnp.sum((sa_repr - goal_repr) ** 2, axis=-1))
        return jnp.mean(jnp.exp(log_alpha) * log_prob - q_value), log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        target_entropy = -0.5 * ACTION_SIZE
        return alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy))

    def critic_loss(critic_params, batch):
        state = batch.observation[:, :OBS_DIM]
        goal = batch.observation[:, OBS_DIM:]
        sa_repr = sa_encoder.apply(
            critic_params["sa_encoder"],
            jnp.concatenate([state, batch.action], axis=-1),
        )
        goal_repr = g_encoder.apply(critic_params["g_encoder"], goal)
        logits = -jnp.sqrt(
            jnp.sum((sa_repr[:, None, :] - goal_repr[None, :, :]) ** 2, axis=-1)
        )
        loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        loss += 0.1 * jnp.mean(logsumexp**2)
        return loss, logsumexp

    def sgd_step(state, batch, key):
        key, critic_key, actor_key = jax.random.split(key, 3)
        (actor_value, log_prob), actor_grad = jax.value_and_grad(
            actor_loss, has_aux=True
        )(
            state.actor_state.params,
            state.critic_state.params,
            state.alpha_state.params["log_alpha"],
            batch,
            actor_key,
        )
        actor_state = state.actor_state.apply_gradients(grads=actor_grad)
        alpha_value, alpha_grad = jax.value_and_grad(alpha_loss)(
            state.alpha_state.params, log_prob
        )
        alpha_state = state.alpha_state.apply_gradients(grads=alpha_grad)
        (critic_value, logsumexp), critic_grad = jax.value_and_grad(
            critic_loss, has_aux=True
        )(state.critic_state.params, batch)
        critic_state = state.critic_state.apply_gradients(grads=critic_grad)
        state = state.replace(
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
            gradient_steps=state.gradient_steps + 1,
        )
        metrics = {
            "actor_loss": actor_value,
            "alpha_loss": alpha_value,
            "critic_loss": critic_value,
            "log_alpha": alpha_state.params["log_alpha"],
            "logsumexp": logsumexp.mean(),
            "sample_entropy": -log_prob,
        }
        del critic_key
        return state, metrics

    def rollout_actor_step(actor_params, observation, key):
        means, log_stds = actor.apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        action = jnp.tanh(
            means
            + stds
            * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        )
        return action

    update_cost = cost_totals(
        jax.jit(sgd_step).lower(training_state, transitions, step_key)
    )
    rollout_observation = jnp.zeros((rollout_batch_size, OBS_DIM + GOAL_DIM))
    rollout_cost = cost_totals(
        jax.jit(rollout_actor_step).lower(
            actor_state.params, rollout_observation, step_key
        )
    )
    return {
        "width": width,
        "depth": depth,
        "sgd_update_flops": round(update_cost["flops"]),
        "sgd_update_transcendentals": round(update_cost["transcendentals"]),
        "rollout_flops_per_env_step": round(
            rollout_cost["flops"] / rollout_batch_size
        ),
        "rollout_transcendentals_per_env_step": round(
            rollout_cost["transcendentals"] / rollout_batch_size
        ),
    }


def architectures_from_plan(path: Path) -> list[tuple[int, int]]:
    with path.open(encoding="utf-8") as file:
        plan = json.load(file)
    return [
        (int(width), int(depth))
        for width in plan["axes"]["widths"]
        for depth in plan["axes"]["depths_by_width"][str(width)]
    ]


def parse_architectures(value: str) -> list[tuple[int, int]]:
    try:
        architectures = [
            tuple(int(item) for item in pair.split(":"))
            for pair in value.split(",")
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected WIDTH:DEPTH pairs") from error
    if not architectures or any(len(pair) != 2 for pair in architectures):
        raise argparse.ArgumentTypeError("expected WIDTH:DEPTH pairs")
    return [(width, depth) for width, depth in architectures]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--architectures", type=parse_architectures)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    architectures = args.architectures or architectures_from_plan(args.plan)
    profiles = []
    for index, (width, depth) in enumerate(architectures, start=1):
        print(
            f"Profiling {index}/{len(architectures)}: width={width}, depth={depth}",
            file=sys.stderr,
            flush=True,
        )
        profiles.append(
            profile_architecture(
                width,
                depth,
                batch_size=args.batch_size,
                rollout_batch_size=args.rollout_batch_size,
            )
        )
    document = {
        "schema_version": 1,
        "method": "JAX lowered HLO cost_analysis on one exact actor+critic SGD update and one stochastic rollout actor step",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": jax.default_backend(),
        "profiler_sha256": file_sha256(Path(__file__).resolve()),
        "train_sha256": file_sha256(REPO_ROOT / "train.py"),
        "simba_sha256": file_sha256(REPO_ROOT / "simba.py"),
        "batch_size": args.batch_size,
        "rollout_batch_size": args.rollout_batch_size,
        "obs_dim": OBS_DIM,
        "goal_dim": GOAL_DIM,
        "action_size": ACTION_SIZE,
        "scope": "Model training and rollout-policy inference; excludes physics simulation, replay-buffer operations, evaluation, checkpoint I/O, and compilation",
        "flop_convention": "XLA arithmetic FLOPs (multiply-add is two FLOPs); transcendental operations are reported separately and excluded from FLOPs",
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(profiles)} profiles to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
