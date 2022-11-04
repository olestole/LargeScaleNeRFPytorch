#!/usr/bin/bash

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=ie-idi
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16000
#SBATCH --job-name="LargeScaleNeRFPytorch Run"
#SBATCH --output=run-srun.out
#SBATCH --mail-user=olesto@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
DATA_DIR=${WORKDIR}/data/hovedbygget_dataset

cd ${WORKDIR}
echo "we are running from this directory: $WORKDIR"
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

# Installation
conda create -n nerf-block python=3.9
pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow 
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Other libs for reconstructing custom scenes
conda install -c conda-forge colmap
conda install -c conda-forge imagemagick
conda install pytorch-scatter -c pyg

# Run training
python run.py --config configs/default.py

uname -a