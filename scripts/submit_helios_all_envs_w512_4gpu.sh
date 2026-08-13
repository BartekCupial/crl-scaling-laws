#!/bin/bash
set -euo pipefail

REPO_ROOT=/net/scratch/hscra/plgrid/plgbartekcupial/crl-scaling-laws
HUMANOID="${REPO_ROOT}/scripts/helios_w512_4gpu.sbatch"
ANT_U4="${REPO_ROOT}/scripts/helios_ant_u4_w512_4gpu.sbatch"
ARM_PUSH="${REPO_ROOT}/scripts/helios_arm_push_hard_w512_4gpu.sbatch"

cd "${REPO_ROOT}"
mkdir -p slurm_logs runs/scaling_laws_helios_3env_w512_4gpu_v1

submit() {
  local result
  result=$(sbatch --parsable "$@")
  printf '%s' "${result%%;*}"
}

# Depth index 5 is width 512 / depth 32.  Production arrays start only when
# the matching environment's real four-GPU smoke test succeeds.
H_SMOKE=$(submit --array=5-5 --time=00:45:00 --export=ALL,CRL_SMOKE=1 "${HUMANOID}")
A_SMOKE=$(submit --array=5-5 --time=00:45:00 --export=ALL,CRL_SMOKE=1 "${ANT_U4}")
P_SMOKE=$(submit --array=5-5 --time=00:45:00 --export=ALL,CRL_SMOKE=1 "${ARM_PUSH}")

H1=$(submit --array=0-8 --dependency="afterok:${H_SMOKE}" --export=ALL,CRL_SEED=1 "${HUMANOID}")
A1=$(submit --array=0-8 --dependency="afterok:${A_SMOKE}" --export=ALL,CRL_SEED=1 "${ANT_U4}")
P1=$(submit --array=0-8 --dependency="afterok:${P_SMOKE}" --export=ALL,CRL_SEED=1 "${ARM_PUSH}")
SEED1_DEP="afterok:${H1}:${A1}:${P1}"

H2=$(submit --array=0-8 --dependency="${SEED1_DEP}" --export=ALL,CRL_SEED=2 "${HUMANOID}")
A2=$(submit --array=0-8 --dependency="${SEED1_DEP}" --export=ALL,CRL_SEED=2 "${ANT_U4}")
P2=$(submit --array=0-8 --dependency="${SEED1_DEP}" --export=ALL,CRL_SEED=2 "${ARM_PUSH}")
SEED2_DEP="afterok:${H2}:${A2}:${P2}"

H3=$(submit --array=0-8 --dependency="${SEED2_DEP}" --export=ALL,CRL_SEED=3 "${HUMANOID}")
A3=$(submit --array=0-8 --dependency="${SEED2_DEP}" --export=ALL,CRL_SEED=3 "${ANT_U4}")
P3=$(submit --array=0-8 --dependency="${SEED2_DEP}" --export=ALL,CRL_SEED=3 "${ARM_PUSH}")

echo "Submitted 3 smoke jobs and 81 scientific runs."
echo "Smoke: humanoid=${H_SMOKE}, ant_u4=${A_SMOKE}, arm_push_hard=${P_SMOKE}"
echo "Seed 1: ${H1},${A1},${P1}"
echo "Seed 2: ${H2},${A2},${P2}"
echo "Seed 3: ${H3},${A3},${P3}"
echo "Monitor: squeue -u ${USER}"
