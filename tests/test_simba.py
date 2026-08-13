import unittest

import jax
import jax.numpy as jnp

from simba import EncoderSimba, l2normalize


class SimbaEncoderTest(unittest.TestCase):
    def test_l2_normalize_handles_zero_vectors(self):
        output = l2normalize(jnp.zeros((2, 5)), axis=-1)
        self.assertTrue(bool(jnp.all(jnp.isfinite(output))))
        self.assertTrue(bool(jnp.all(output == 0)))

    def test_depths_have_finite_outputs_and_gradients(self):
        inputs = jnp.linspace(-1.0, 1.0, 4 * 285).reshape(4, 285)
        previous_params = 0
        for depth in (4, 8, 16, 32, 64):
            with self.subTest(depth=depth):
                encoder = EncoderSimba(network_width=256, network_depth=depth)
                params = encoder.init(jax.random.PRNGKey(depth), inputs)
                output = encoder.apply(params, inputs)
                gradients = jax.grad(lambda p: jnp.square(encoder.apply(p, inputs)).mean())(params)
                param_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))

                self.assertEqual(output.shape, (4, 64))
                self.assertTrue(bool(jnp.all(jnp.isfinite(output))))
                self.assertTrue(
                    all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(gradients))
                )
                self.assertGreater(param_count, previous_params)
                previous_params = param_count

    def test_rejects_zero_depth(self):
        with self.assertRaises(ValueError):
            EncoderSimba(network_depth=0).init(jax.random.PRNGKey(0), jnp.ones((1, 3)))


if __name__ == "__main__":
    unittest.main()
