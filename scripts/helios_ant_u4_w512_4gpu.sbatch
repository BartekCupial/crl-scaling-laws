#!/bin/bash -l
#SBATCH --job-name=crl-ant-u4-w512
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=260
#SBATCH --mem=400G
#SBATCH --time=47:59:59
#SBATCH --gres=gpu:4
#SBATCH --account=plgcontrastivesft-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm_logs/helios-ant-u4-s%a-%A_%a.out
#SBATCH --error=slurm_logs/helios-ant-u4-s%a-%A_%a.err

export CRL_ENV_ID=ant_u4_maze
export CRL_EVAL_ENV_ID=ant_u4_maze_eval
export CRL_ENV_CODE=ant_u4
export CRL_STEPS_MODE=fixed
export CRL_FIXED_STEPS=400000000
export CRL_FIXED_EPOCHS=100
exec bash /net/scratch/hscra/plgrid/plgbartekcupial/crl-scaling-laws/scripts/run_helios_w512_4gpu_task.sh
