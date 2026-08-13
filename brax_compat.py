"""Runtime compatibility fixes for Brax 0.10.1 with MuJoCo newer than 3.1.5."""

from typing import Optional

import jax
from brax import contact
from brax import math
from brax.base import Contact
from brax.base import System
from brax.base import Transform
from jax import numpy as jp
from mujoco import mjx


def _get_contact(sys: System, x: Transform) -> Optional[Contact]:
    """Calculates contacts without the removed ``mjx.ncon`` helper."""
    data = mjx.make_data(sys)
    if data.ncon == 0:
        return None

    @jax.vmap
    def local_to_global(pos1, quat1, pos2, quat2):
        pos = pos1 + math.rotate(pos2, quat1)
        mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
        return pos, mat

    x = x.concatenate(Transform.zero((1,)))
    xpos = x.pos[sys.geom_bodyid - 1]
    xquat = x.rot[sys.geom_bodyid - 1]
    geom_xpos, geom_xmat = local_to_global(xpos, xquat, sys.geom_pos, sys.geom_quat)

    data = data.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
    data = mjx.collision(sys, data)
    collision = data.contact
    elasticity = (sys.elasticity[collision.geom1] + sys.elasticity[collision.geom2]) * 0.5
    body1 = jp.array(sys.geom_bodyid)[collision.geom1] - 1
    body2 = jp.array(sys.geom_bodyid)[collision.geom2] - 1
    return Contact(elasticity=elasticity, link_idx=(body1, body2), **collision.__dict__)


def apply_brax_compatibility_patch() -> None:
    """Patch Brax in memory; idempotent and persistent across ``uv sync``."""
    if not hasattr(mjx, "ncon"):
        contact.get = _get_contact
