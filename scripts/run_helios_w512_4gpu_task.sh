#!/bin/bash
set -euo pipefail

: "${CRL_ENV_ID:?missing CRL_ENV_ID}"
: "${CRL_EVAL_ENV_ID:?missing CRL_EVAL_ENV_ID}"
: "${CRL_ENV_CODE:?missing CRL_ENV_CODE}"
: "${CRL_STEPS_MODE:?missing CRL_STEPS_MODE}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array}"

DEPTHS=(4 8 12 16 24 32 40 48 64)
HUMANOID_STEPS=(842000000 647000000 561000000 508000000 443000000 403000000 400000000 400000000 400000000)
HUMANOID_EPOCHS=(169 130 113 102 100 100 100 100 100)
INDEX="${SLURM_ARRAY_TASK_ID}"
if (( INDEX < 0 || INDEX >= ${#DEPTHS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${INDEX}" >&2
  exit 2
fi

DEPTH="${DEPTHS[$INDEX]}"
SEED="${CRL_SEED:-1}"
if [[ "${CRL_SMOKE:-0}" == 1 ]]; then
  STEPS=1000000
  EPOCHS=5
  STAGE=smoke
  TRACK_FLAG=--no-track
elif [[ "${CRL_STEPS_MODE}" == humanoid_extended ]]; then
  STEPS="${HUMANOID_STEPS[$INDEX]}"
  EPOCHS="${HUMANOID_EPOCHS[$INDEX]}"
  STAGE=main
  TRACK_FLAG=--track
else
  STEPS="${CRL_FIXED_STEPS:?missing CRL_FIXED_STEPS}"
  EPOCHS="${CRL_FIXED_EPOCHS:?missing CRL_FIXED_EPOCHS}"
  STAGE=main
  TRACK_FLAG=--track
fi

REPO_ROOT=/net/scratch/hscra/plgrid/plgbartekcupial/crl-scaling-laws
STEPS_MILLIONS=$((STEPS / 1000000))
RUN_ROOT="${REPO_ROOT}/runs/scaling_laws_helios_3env_w512_4gpu_v1"
RUN_NAME=$(printf '%s_sim_w512_d%03d_s%d_%dm_%s_helios4g_v1' \
  "${CRL_ENV_CODE}" "${DEPTH}" "${SEED}" "${STEPS_MILLIONS}" "${STAGE}")
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"

if [[ -f "${RUN_DIR}/COMPLETE" ]]; then
  echo "Already complete: ${RUN_DIR}"
  exit 0
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite incomplete run directory: ${RUN_DIR}" >&2
  exit 3
fi
mkdir -p "${RUN_DIR}"

module purge
module load GCCcore/13.2.0
module load CUDA/12.8.0
module load cuDNN/8.9.7.29-CUDA-12.8.0
module load NCCL/2.26.2-CUDA-12.8.0
module load Python/3.11.5

cd "${REPO_ROOT}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda NCCL_DEBUG=WARN PYTHONUNBUFFERED=1

python - <<'PY'
import ctypes
import jax
ctypes.CDLL("libnccl.so.2")
assert len(jax.devices()) == 4, jax.devices()
print("JAX devices:", jax.devices(), flush=True)
PY

set +e
/usr/bin/time -f $'END-TO-END WALL TIME: %e seconds' \
  uv run --no-sync train.py \
    --env-id "${CRL_ENV_ID}" --eval-env-id "${CRL_EVAL_ENV_ID}" \
    --seed "${SEED}" --num-devices 4 \
    --episode-length 1000 --num-envs 512 --num-eval-envs 128 \
    --critic-network-width 512 --actor-network-width 512 \
    --critic-depth "${DEPTH}" --actor-depth "${DEPTH}" \
    --critic-skip-connections 4 --actor-skip-connections 4 --use-simba 1 \
    --batch-size 512 --num-sgd-batches-per-training-step 800 \
    --training-steps-multiplier 1 --num-episodes-per-env 1 --unroll-length 62 \
    --min-replay-size 1000 --max-replay-size 10000 \
    --actor-lr 0.0003 --critic-lr 0.0003 --alpha-lr 0.0003 \
    --gamma 0.99 --logsumexp-penalty-coeff 0.1 --entropy-param 0.5 \
    --eval-actor 0 --expl-actor 1 --use-relu 0 \
    --total-env-steps "${STEPS}" --num-epochs "${EPOCHS}" \
    "${TRACK_FLAG}" \
    --wandb-entity ideas-ncbr --wandb-project-name crl_scaling_laws \
    --wandb-mode online --wandb-group "${CRL_ENV_CODE}_crl_helios_w512_4gpu_v1_s${SEED}" \
    --no-capture-vis --checkpoint --wandb-dir "${RUN_DIR}" --exp-name "${RUN_NAME}" \
  2>&1 | tee "${RUN_DIR}/launcher.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

if (( TRAIN_STATUS == 0 )); then
  printf 'ok\n' > "${RUN_DIR}/COMPLETE"
else
  printf 'exit_code=%d\n' "${TRAIN_STATUS}" > "${RUN_DIR}/FAILED"
fi
exit "${TRAIN_STATUS}"
