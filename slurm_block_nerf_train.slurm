#!/usr/bin/bash

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=ie-idi
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16000
#SBATCH --job-name="LargeScaleNeRFPytorch BlockNerfTrain"
#SBATCH --output=block_nerf_gpu-srun.out
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

# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Train 10 sub-models
ROOT_DIR=$WORKDIR/data/pytorch_waymo_dataset
BATCH_SIZE=1024
NUM_EPOCHS=1
NUM_GPUS=1
EXP_NAME=exp

# Run the training script
for BLOCK_INDEX in {0..10}
do
    python train_block_nerf.py --root_dir $ROOT_DIR --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --num_gpus $NUM_GPUS --exp_name $EXP_NAME --block_index $BLOCK_INDEX
done

uname -a