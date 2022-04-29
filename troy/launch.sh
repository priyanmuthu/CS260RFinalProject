#!/bin/bash
#SBATCH --job-name=nccl-perf     # create a short name for your job
#SBATCH --nodes 2                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH -p seas_dgx1
#SBATCH --open-mode=append
#SBATCH --output=train.out

module purge
module load cuda/10.2.89-fasrc01
module load nccl/2.7.3-fasrc01
module load Anaconda/5.0.1-fasrc02

conda activate torchdist || source activate torchdist

export WORLD_SIZE=8

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export PYTHONUNBUFFERED=TRUE

python3 --version

srun -u python3 -u my_train_dist.py /n/home02/tappel/dodrio/dist_training/imagenette2 imagenette2 10