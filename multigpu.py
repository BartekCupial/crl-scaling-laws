"""Small SPMD helpers shared by the CRL multi-GPU training path."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def synchronize_gradients(grads: Any, num_devices: int, axis_name: str) -> Any:
    """Averages a replicated parameter gradient across local devices."""
    if num_devices == 1:
        return grads
    return jax.lax.pmean(grads, axis_name=axis_name)


def contrastive_logits_and_positives(
    sa_repr: jax.Array,
    g_repr: jax.Array,
    num_devices: int,
    axis_name: str,
) -> tuple[jax.Array, jax.Array]:
    """Builds local-anchor/global-goal InfoNCE logits and positive entries.

    For data parallel training, each device owns a contiguous shard of both
    representations.  Gathering goals preserves the original global batch as
    the negative set.  The positive for local row ``i`` is at the corresponding
    global goal offset for this device.
    """
    local_batch_size = sa_repr.shape[0]
    if num_devices == 1:
        all_g_repr = g_repr
        positive_indices = jnp.arange(local_batch_size)
    else:
        all_g_repr = jax.lax.all_gather(
            g_repr, axis_name=axis_name, axis=0, tiled=True
        )
        positive_indices = (
            jax.lax.axis_index(axis_name) * local_batch_size
            + jnp.arange(local_batch_size)
        )

    logits = -jnp.sqrt(
        jnp.sum((sa_repr[:, None, :] - all_g_repr[None, :, :]) ** 2, axis=-1)
    )
    positives = logits[jnp.arange(local_batch_size), positive_indices]
    return logits, positives
