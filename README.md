# 1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities

<p align="center">
    <a href= "https://arxiv.org/abs/2503.14858">
        <img src="https://img.shields.io/badge/arXiv-2311.10090-b31b1b.svg" /></a>
    <a href= "https://github.com/wang-kevin3290/scaling-crl/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
    <a href= "https://wang-kevin3290.github.io/scaling-crl/">
        <img src="https://img.shields.io/badge/website-purple" /></a>
</p>

> [!IMPORTANT]  
> [Our work was selected for the Best Paper Award at NeurIPS 2025!](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/#:~:text=1000%20Layer%20Networks%20for%20Self%2DSupervised%20RL%3A%20Scaling%20Depth%20Can%20Enable%20New%20Goal%2DReaching%20Capabilities) 🥳

Email kw6487@princeton.edu with questions/comments/suggestions.

![Environments](assets/envs.gif)
Our work builds on top of [JAXGCRL](https://github.com/MichalBortkiewicz/JaxGCRL), feel free to check it out!!

# Installation

```sh
uv sync
```
Then just fix the two Brax issues described below, and you'll be all set.


## Fixing two bugs in brax 0.10.1
1. There is a minor bug in brax's contact.py file. To fix it, first locate the brax contact.py file in your virtual environment: 
```
find .venv -name contact.py
```
Then open the file and replace it with the following code:
```python
from typing import Optional
from brax import math
from brax.base import Contact
from brax.base import System
from brax.base import Transform
import jax
from jax import numpy as jp
from mujoco import mjx

def get(sys: System, x: Transform) -> Optional[Contact]:
    """Calculates contacts.
    Args:
        sys: system defining the kinematic tree and other properties
        x: link transforms in world frame
    Returns:
        Contact pytree
    """
    #NOTE: THIS WAS MODIFIED SINCE AFTER MUJOCO 3.1.5, mjx.ncon IS NOT AVAILABLE
    # ncon = mjx.ncon(sys)
    # if not ncon:
    #   return None
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
    geom_xpos, geom_xmat = local_to_global(
        xpos, xquat, sys.geom_pos, sys.geom_quat
    )
    # pytype: disable=wrong-arg-types
    d = data.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
    d = mjx.collision(sys, d)
    # pytype: enable=wrong-arg-types
    c = d.contact
    elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5
    body1 = jp.array(sys.geom_bodyid)[c.geom1] - 1
    body2 = jp.array(sys.geom_bodyid)[c.geom2] - 1
    link_idx = (body1, body2)
    return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
```
2. There is also a minor bug in brax's json.py file. To fix it, first locate the brax json.py file in your virtual environment:
```
find .venv -name json.py | grep "/brax/io/json.py"
```
Then open the file and change the if statement in line 159 to:  
```python
if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all():
```


# Running experiments
Now, we are ready to run the train script. To run the code, you'll need a GPU. For Humanoid-based environments, it may require up to 80GB of GPU memory (for deep networks). Below is an example command to run the training script (an additional example can be found in the provided slurm script `job.slurm`): 

```sh
uv run train.py --env_id "humanoid" --eval_env_id "humanoid" --num_epochs 100 --total_env_steps 100000000 --critic_depth 16 --actor_depth 16 --actor_skip_connections 4 --critic_skip_connections 4 --batch_size 512 --vis_length 1000 --save_buffer 0 
```


>[!NOTE]
>If you would like the experiments to be synced to wandb, you should go to `train.py` and replace the default values of `wandb_entity` and `wandb_project_name` (line 34-35 of the `train.py` file) with your particular wandb entity and wandb project name. Alternatively, these two can also be set as hyperparameter flags when running the train script.

## Exploratory scaling plots

The plotting script uses [`wandb-cache`](https://github.com/BartekCupial/wandb-cache) to cache run metadata and complete training histories as Parquet. It plots every available intermediate observation from fitting and held-out runs, excludes smoke runs, and does not fit or extrapolate a scaling law.

The compute axis reports cumulative XLA-estimated arithmetic FLOPs rather than `parameters × environment steps`. The estimate profiles the exact actor update, contrastive critic update, Adam updates, and stochastic rollout-policy inference for every width/depth pair. A multiply-add counts as two FLOPs. Physics simulation, replay-buffer processing, evaluation, checkpoint I/O, and JIT compilation are excluded. Transcendental operations are exported separately because XLA does not count them as arithmetic FLOPs.

The checked-in profile is generated with:

```bash
uv run --no-sync python scripts/profile_crl_flops.py
```

The plotting script rejects the profile if `train.py` or `simba.py` has changed, forcing the FLOP counts to be regenerated rather than silently using stale estimates.

```sh
uv run --no-sync python scripts/plot_crl_scaling.py --refresh
```

This writes seven PNG/PDF figures plus tidy CSV/Parquet data under `figures/scaling_laws/`: loss versus estimated FLOPs, parameters, dataset size, parameters at fixed environment-step checkpoints, Chinchilla-style isoFLOP profiles, parameters versus compute, and environment samples versus compute. The default loss is `training/critic_loss`; all axes are logarithmic. The tidy data include cumulative `estimated_training_flops`, `gradient_updates`, per-update FLOPs, separately counted transcendental operations, and `estimated_flops_per_second`.

`training/critic_loss` is a training loss, not a held-out dev loss. To create genuine dev-loss plots, first log a positive dev metric during training and pass its W&B key with `--metric`, for example `--metric eval/dev_critic_loss`. The same loss-versus-parameters and loss-versus-compute figures will then use that metric.

The isoFLOP plot uses log-linear interpolation between observed checkpoints from the same run. It never extrapolates beyond a run's observed compute range and does not fit parabolas or scaling-law curves. Interpolated points are exported to `isoflop_observations.csv`; compute budgets can be changed with `--isoflop-budgets`.

The budget-point analysis first treats every run as a truncated run at each
compute budget it reaches. Each candidate uses the lowest observed loss up to
that budget; its loss, environment-sample count, and actual observation compute
all come from the same checkpoint. The loss-versus-parameters isoFLOP plot shows
all of these candidates. The loss, parameters, and samples versus compute plots
then show only the single lowest-loss candidate at each budget, matching the
empirical analogue of the optimal-point plots in Tuyls et al. without fitting a
parabola. Candidates are exported to `budget_minimum_observations.csv` and the
one-per-budget frontier to `budget_optimal_observations.csv`.

The extended v2 plan removes the 1, 2, 3, and 5 PF regimes from the default
frontier and replaces the fixed 100M-step main horizon with a parameter-aware
schedule. The 0.95M-parameter model receives 2B environment steps, the
142M-parameter model receives 400M, and intermediate architectures are
log-interpolated between those endpoints. Evaluations remain at most about 5M
environment steps apart. Extended run names contain `_ext`, so completed 100M
runs are preserved and continue to provide early observations while the longer
runs are active.

Preview or launch the extended seed-1 queue on four GPUs with:

```sh
uv run --no-sync python scripts/run_crl_scaling_grid_v2.py --phase seed1 --gpus 0,1,2,3
uv run --no-sync python scripts/run_crl_scaling_grid_v2.py --phase seed1 --gpus 0,1,2,3 --execute
```

The latest point from a W&B run whose state is `running` is drawn as an upward triangle on every plot. Finished runs retain their normal circle or stage marker.

While runs are active, refresh the figures every five minutes from a second terminal or tmux pane:

```sh
watch -n 300 'uv run --no-sync python scripts/plot_crl_scaling.py --refresh'
```

To inspect only the extended 400M--2B runs and keep their outputs separate from
the combined figures:

```sh
uv run --no-sync python scripts/plot_crl_scaling.py --refresh \
  --run-variant extended --output-dir figures/scaling_laws_ext
```

Use `--metric <wandb-key>` to inspect another positive-valued metric.

# Citing Scaling CRL 📜
```bibtex
@inproceedings{wang2025,
  title     = {1000 Layer Networks for Self-Supervised {RL}: Scaling Depth Can Enable New Goal-Reaching Capabilities},
  author    = {Kevin Wang and Ishaan Javali and Micha{\l} Bortkiewicz and Tomasz Trzcinski and Benjamin Eysenbach},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2025},
  url       = {https://openreview.net/forum?id=s0JVsx3bx1}
}
```



<!-- 
## Troubleshooting Potential Errors

**If you encounter the following error:**
```AttributeError: module 'mujoco.mjx' has no attribute 'ncon'```  

**Fix:**
1. Locate the brax contact.py file in your conda environment: 
   ```
   find ~/.conda/envs/scaling-crl -name contact.py
   ```
2. Open the file and replace it with the following code:

    ```python
    from typing import Optional
    from brax import math
    from brax.base import Contact
    from brax.base import System
    from brax.base import Transform
    import jax
    from jax import numpy as jp
    from mujoco import mjx

    def get(sys: System, x: Transform) -> Optional[Contact]:
        """Calculates contacts.
        Args:
            sys: system defining the kinematic tree and other properties
            x: link transforms in world frame
        Returns:
            Contact pytree
        """
        #NOTE: THIS WAS MODIFIED SINCE AFTER MUJOCO 3.1.5, mjx.ncon IS NOT AVAILABLE
        # ncon = mjx.ncon(sys)
        # if not ncon:
        #   return None
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
        geom_xpos, geom_xmat = local_to_global(
            xpos, xquat, sys.geom_pos, sys.geom_quat
        )
        # pytype: disable=wrong-arg-types
        d = data.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
        d = mjx.collision(sys, d)
        # pytype: enable=wrong-arg-types
        c = d.contact
        elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5
        body1 = jp.array(sys.geom_bodyid)[c.geom1] - 1
        body2 = jp.array(sys.geom_bodyid)[c.geom2] - 1
        link_idx = (body1, body2)
        return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
    ```
3. Save the file and rerun the training script.


**If you encounter the following error:** ```Error rendering final policy: unsupported operand type(s) for ==: 'ArrayImpl' and 'list'```  

**Fix:**
1. Locate the brax json.py file in your conda environment:
   ```
   find ~/.conda/envs/scaling-crl -name json.py | grep "/brax/io/json.py"
   ```
2. Open the file and change the if statement in line 159 to:
    ```python
    if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all():
    ```
3. Save the file and rerun the training script. -->
