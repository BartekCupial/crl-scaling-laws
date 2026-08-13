#!/bin/bash -l
#SBATCH --job-name=crl-w512-4g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=260
#SBATCH --mem=400G
#SBATCH --time=47:59:59
#SBATCH --gres=gpu:4
#SBATCH --account=plgcontrastivesft-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm_logs/helios-w512-s%a-%A_%a.out
#SBATCH --error=slurm_logs/helios-w512-s%a-%A_%a.err

set -euo pipefail

: "${CRL_SEED:?CRL_SEED must be exported by submit_helios_w512_4gpu.sh}"
: "${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array}"

DEPTHS=(4 8 12 16 24 32 40 48 64)
TOTAL_ENV_STEPS=(842000000 647000000 561000000 508000000 443000000 403000000 400000000 400000000 400000000)
NUM_EPOCHS=(169 130 113 102 100 100 100 100 100)

TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#DEPTHS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_INDEX}" >&2
  exit 2
fi

DEPTH="${DEPTHS[$TASK_INDEX]}"
STEPS="${TOTAL_ENV_STEPS[$TASK_INDEX]}"
EPOCHS="${NUM_EPOCHS[$TASK_INDEX]}"
STEPS_MILLIONS=$((STEPS / 1000000))

REPO_ROOT=/net/scratch/hscra/plgrid/plgbartekcupial/crl-scaling-laws
RUN_ROOT="${REPO_ROOT}/runs/scaling_laws_helios_w512_4gpu_v1"
RUN_NAME=$(printf 'humanoid_sim_w512_d%03d_s%d_%dm_ext_helios4g_v1' \
  "${DEPTH}" "${CRL_SEED}" "${STEPS_MILLIONS}")
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

export JAX_PLATFORMS=cuda
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

python - <<'PY'
import ctypes
import jax

ctypes.CDLL("libnccl.so.2")
devices = jax.devices()
print("JAX devices:", devices, flush=True)
assert len(devices) == 4, devices
PY

{
  echo "seed=${CRL_SEED} depth=${DEPTH} steps=${STEPS} epochs=${EPOCHS}"
  echo "slurm_job_id=${SLURM_JOB_ID} array_task_id=${SLURM_ARRAY_TASK_ID}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  module list
} > "${RUN_DIR}/helios_job.txt" 2>&1

set +e
/usr/bin/time -f $'END-TO-END WALL TIME: %e seconds' \
  uv run --no-sync train.py \
    --env-id humanoid \
    --eval-env-id humanoid \
    --seed "${CRL_SEED}" \
    --num-devices 4 \
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
    --batch-size 256 \
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
    --total-env-steps "${STEPS}" \
    --num-epochs "${EPOCHS}" \
    --track \
    --wandb-entity ideas-ncbr \
    --wandb-project-name crl_scaling_laws \
    --wandb-mode online \
    --wandb-group "humanoid_crl_helios_w512_4gpu_v1_s${CRL_SEED}" \
    --no-capture-vis \
    --checkpoint \
    --wandb-dir "${RUN_DIR}" \
    --exp-name "${RUN_NAME}" \
  2>&1 | tee "${RUN_DIR}/launcher.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

if (( TRAIN_STATUS == 0 )); then
  printf 'ok\n' > "${RUN_DIR}/COMPLETE"
else
  printf 'exit_code=%d\n' "${TRAIN_STATUS}" > "${RUN_DIR}/FAILED"
fi
exit "${TRAIN_STATUS}"
