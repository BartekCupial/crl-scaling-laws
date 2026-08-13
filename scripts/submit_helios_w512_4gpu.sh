#!/bin/bash
set -euo pipefail

REPO_ROOT=/net/scratch/hscra/plgrid/plgbartekcupial/crl-scaling-laws
SBATCH_SCRIPT="${REPO_ROOT}/scripts/helios_w512_4gpu.sbatch"

cd "${REPO_ROOT}"
mkdir -p slurm_logs runs/scaling_laws_helios_w512_4gpu_v1

# Submit all nine architectures for a seed as an array.  Later seeds begin
# only after every task in the preceding seed has completed successfully.
SEED1_JOB=$(sbatch --parsable --array=0-8 --export=ALL,CRL_SEED=1 "${SBATCH_SCRIPT}")
SEED1_JOB=${SEED1_JOB%%;*}
SEED2_JOB=$(sbatch --parsable --array=0-8 --dependency="afterok:${SEED1_JOB}" \
  --export=ALL,CRL_SEED=2 "${SBATCH_SCRIPT}")
SEED2_JOB=${SEED2_JOB%%;*}
SEED3_JOB=$(sbatch --parsable --array=0-8 --dependency="afterok:${SEED2_JOB}" \
  --export=ALL,CRL_SEED=3 "${SBATCH_SCRIPT}")
SEED3_JOB=${SEED3_JOB%%;*}

echo "Submitted width-512 four-GPU grid (27 runs):"
echo "  seed 1: ${SEED1_JOB}_[0-8]"
echo "  seed 2: ${SEED2_JOB}_[0-8], afterok:${SEED1_JOB}"
echo "  seed 3: ${SEED3_JOB}_[0-8], afterok:${SEED2_JOB}"
echo "Monitor with: squeue -j ${SEED1_JOB},${SEED2_JOB},${SEED3_JOB}"
