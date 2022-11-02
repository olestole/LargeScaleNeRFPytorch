#!/usr/bin/bash

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:V10032
#SBATCH --account=ie-idi
#SBATCH --time=03:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16000
#SBATCH --job-name="training_mnist_test"
#SBATCH --output=test-srun.out
#SBATCH --mail-user=olesto@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

# Purge modules and load tensorflow
module purge
module load Anaconda3/2020.07

# List loaded modules for reproducibility
module list

# Create venv and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run block_nerf_eval
bash $SLURM_SUBMIT_DIR/scripts/block_nerf_eval.sh

uname -a