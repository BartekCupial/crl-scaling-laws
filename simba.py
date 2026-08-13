"""SimBa critic encoder used by the original JaxGCRL experiments.

This is a direct port of ``EncoderSimba`` and its HyperLERP building blocks
from the W&B-saved training source used for the completed Humanoid SimBa runs.
"""

import math

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


EPS = 1e-8


def l2normalize(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    l2norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / jnp.maximum(l2norm, EPS)


class Scaler(nn.Module):
    dim: int
    init: float = 1.0
    scale: float = 1.0

    def setup(self) -> None:
        self.scaler = self.param(
            "scaler",
            nn.initializers.constant(self.scale),
            self.dim,
        )
        self.forward_scaler = self.init / self.scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.scaler * self.forward_scaler * x


class HyperDense(nn.Module):
    hidden_dim: int

    def setup(self) -> None:
        self.w = nn.Dense(
            name="hyper_dense",
            features=self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=1.0, column_axis=0),
            use_bias=False,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.w(x)


class HyperMLP(nn.Module):
    hidden_dim: int
    out_dim: int
    scaler_init: float
    scaler_scale: float
    eps: float = EPS

    def setup(self) -> None:
        self.w1 = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)
        self.w2 = HyperDense(self.out_dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.w1(x)
        x = self.scaler(x)
        x = nn.relu(x) + self.eps
        x = self.w2(x)
        return l2normalize(x, axis=-1)


class HyperEmbedder(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    c_shift: float

    def setup(self) -> None:
        self.w = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        new_axis = jnp.ones(x.shape[:-1] + (1,)) * self.c_shift
        x = jnp.concatenate([x, new_axis], axis=-1)
        x = l2normalize(x, axis=-1)
        x = self.w(x)
        x = self.scaler(x)
        return l2normalize(x, axis=-1)


class HyperLERPBlock(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    expansion: int = 4

    def setup(self) -> None:
        self.mlp = HyperMLP(
            hidden_dim=self.hidden_dim * self.expansion,
            out_dim=self.hidden_dim,
            scaler_init=self.scaler_init / math.sqrt(self.expansion),
            scaler_scale=self.scaler_scale / math.sqrt(self.expansion),
        )
        self.alpha_scaler = Scaler(
            self.hidden_dim,
            init=self.alpha_init,
            scale=self.alpha_scale,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.mlp(x)
        x = residual + self.alpha_scaler(x - residual)
        return l2normalize(x, axis=-1)


class EncoderSimba(nn.Module):
    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 4
    # Retained for command/config compatibility with the residual encoders.
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False

    def setup(self) -> None:
        if self.network_depth < 1:
            raise ValueError("SimBa network_depth must be at least 1")

        num_blocks = self.network_depth - 1
        scaler_init = math.sqrt(2 / self.network_width)
        scaler_scale = math.sqrt(2 / self.network_width)
        alpha_init = 1 / (num_blocks + 1)
        alpha_scale = 1 / math.sqrt(self.network_width)

        self.embedder = HyperEmbedder(
            hidden_dim=self.network_width,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=3,
        )
        self.encoder = nn.Sequential(
            [
                HyperLERPBlock(
                    hidden_dim=self.network_width,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
                for _ in range(num_blocks)
            ]
        )

    @nn.compact
    def __call__(self, data: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        x = self.embedder(data)
        x = self.encoder(x)
        return nn.Dense(
            self.repr_dim,
            kernel_init=lecun_uniform,
            bias_init=nn.initializers.zeros,
        )(x)
