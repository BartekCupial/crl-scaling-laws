#!/bin/bash -l
#SBATCH --job-name=crl-2g-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=130
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:2
#SBATCH --account=plgcontrastivesft-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm_logs/helios-multigpu-smoke-%j.out
#SBATCH --error=slurm_logs/helios-multigpu-smoke-%j.err

set -euo pipefail

REPO_ROOT=/net/scratch/hscra/plgrid/plgbartekcupial/crl-scaling-laws
NUM_DEVICES="${CRL_NUM_DEVICES:-2}"
ENV_ID="${CRL_ENV_ID:-humanoid}"
EVAL_ENV_ID="${CRL_EVAL_ENV_ID:-${ENV_ID}}"
ENV_CODE="${CRL_ENV_CODE:-${ENV_ID}}"
DEPTH="${CRL_DEPTH:-32}"
JOB_LABEL="${SLURM_JOB_ID:-interactive}"
ATTEMPT_LABEL="${CRL_SMOKE_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${ENV_CODE}_${NUM_DEVICES}gpu_simba_w512_d${DEPTH}_smoke_${JOB_LABEL}_${ATTEMPT_LABEL}"
RUN_DIR="${REPO_ROOT}/runs/helios_multigpu_smoke/${RUN_NAME}"

module purge
module load GCCcore/13.2.0
module load CUDA/12.8.0
module load cuDNN/8.9.7.29-CUDA-12.8.0
module load NCCL/2.26.2-CUDA-12.8.0
module load Python/3.11.5

cd "${REPO_ROOT}"
source .venv/bin/activate
mkdir -p "${RUN_DIR}"

export JAX_PLATFORMS=cuda
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

python - "${NUM_DEVICES}" <<'PY'
import ctypes
import sys

import jax

expected = int(sys.argv[1])
ctypes.CDLL("libnccl.so.2")
devices = jax.devices()
print("JAX devices:", devices, flush=True)
assert len(devices) == expected, f"Expected {expected} devices, got {devices}"
PY

echo "Run directory: ${RUN_DIR}"
/usr/bin/time -f $'END-TO-END WALL TIME: %e seconds' \
  uv run --no-sync train.py \
    --env-id "${ENV_ID}" \
    --eval-env-id "${EVAL_ENV_ID}" \
    --seed 1 \
    --num-devices "${NUM_DEVICES}" \
    --episode-length 1000 \
    --num-envs 512 \
    --num-eval-envs 128 \
    --critic-network-width 512 \
    --actor-network-width 512 \
    --critic-depth "${DEPTH}" \
    --actor-depth "${DEPTH}" \
    --critic-skip-connections 4 \
    --actor-skip-connections 4 \
    --use-simba 1 \
    --batch-size 512 \
    --num-sgd-batches-per-training-step 800 \
    --training-steps-multiplier 1 \
    --num-episodes-per-env 1 \
    --unroll-length 62 \
    --min-replay-size 1000 \
    --max-replay-size 10000 \
    --actor-lr 0.0003 \
    --critic-lr 0.0003 \
    --alpha-lr 0.0003 \
    --gamma 0.99 \
    --logsumexp-penalty-coeff 0.1 \
    --entropy-param 0.5 \
    --eval-actor 0 \
    --expl-actor 1 \
    --use-relu 0 \
    --total-env-steps 1000000 \
    --num-epochs 5 \
    --no-track \
    --no-capture-vis \
    --checkpoint \
    --wandb-dir "${RUN_DIR}" \
    --exp-name "${RUN_NAME}" \
  2>&1 | tee "${RUN_DIR}/launcher.log"

printf 'ok\n' > "${RUN_DIR}/COMPLETE"
echo "Smoke test complete: ${RUN_DIR}"
