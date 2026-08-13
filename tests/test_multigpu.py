import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import jax.numpy as jnp

from multigpu import contrastive_logits_and_positives, synchronize_gradients


ROOT = Path(__file__).resolve().parents[1]


class MultiGpuHelpersTest(unittest.TestCase):
    def test_single_device_helpers_are_identity_compatible(self):
        sa = jnp.asarray([[0.0, 1.0], [2.0, 3.0]])
        goals = jnp.asarray([[1.0, 0.0], [3.0, 2.0]])
        logits, positives = contrastive_logits_and_positives(
            sa, goals, num_devices=1, axis_name="devices"
        )
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(jnp.array_equal(positives, jnp.diag(logits)))
        grads = {"weight": jnp.asarray([1.0, 2.0])}
        self.assertIs(synchronize_gradients(grads, 1, "devices"), grads)

    def test_four_device_loss_and_gradient_match_global_batch(self):
        program = textwrap.dedent(
            """
            import jax
            import jax.numpy as jnp
            from multigpu import contrastive_logits_and_positives, synchronize_gradients

            devices, local_batch, features = 4, 2, 3
            x = jnp.arange(devices * local_batch * features, dtype=jnp.float32).reshape(
                devices * local_batch, features
            ) / 10
            goals = jnp.flip(x, axis=0) + 0.3
            weight = (
                jnp.arange(features * features, dtype=jnp.float32).reshape(features, features)
                / 20
            )

            def loss_from_logits(logits, positives):
                return -jnp.mean(positives - jax.nn.logsumexp(logits, axis=1))

            def global_loss(weight):
                # The real critic has separate state-action and goal encoders.
                sa_repr = x @ weight
                g_repr = goals @ (weight + 0.07 * jnp.eye(features))
                logits, positives = contrastive_logits_and_positives(
                    sa_repr, g_repr, 1, "devices"
                )
                return loss_from_logits(logits, positives)

            expected_loss, expected_grad = jax.value_and_grad(global_loss)(weight)

            def local_step(weight, local_x, local_goals):
                def local_loss(weight):
                    logits, positives = contrastive_logits_and_positives(
                        local_x @ weight,
                        local_goals @ (weight + 0.07 * jnp.eye(features)),
                        devices,
                        "devices",
                    )
                    return loss_from_logits(logits, positives)

                loss, grad = jax.value_and_grad(local_loss)(weight)
                return (
                    jax.lax.pmean(loss, "devices"),
                    synchronize_gradients(grad, devices, "devices"),
                )

            weights = jnp.broadcast_to(weight, (devices,) + weight.shape)
            losses, grads = jax.pmap(local_step, axis_name="devices")(
                weights,
                x.reshape(devices, local_batch, features),
                goals.reshape(devices, local_batch, features),
            )
            assert jnp.allclose(expected_loss, losses[0], rtol=1e-5, atol=1e-5)
            assert jnp.allclose(expected_grad, grads[0], rtol=1e-5, atol=1e-5)
            """
        )
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
