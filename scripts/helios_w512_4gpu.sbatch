#!/bin/bash -l
#SBATCH --job-name=crl-hum-w512
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=260
#SBATCH --mem=400G
#SBATCH --time=47:59:59
#SBATCH --gres=gpu:4
#SBATCH --account=plgllmreasoning-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm_logs/helios-humanoid-s%a-%A_%a.out
#SBATCH --error=slurm_logs/helios-humanoid-s%a-%A_%a.err

export CRL_ENV_ID=humanoid
export CRL_EVAL_ENV_ID=humanoid
export CRL_ENV_CODE=humanoid
export CRL_STEPS_MODE=humanoid_extended
exec bash /net/scratch/hscra/plgrid/plgbartekcupial/crl-scaling-laws/scripts/run_helios_w512_4gpu_task.sh
