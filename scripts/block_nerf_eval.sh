SAVE_PATH=$SLURM_SUBMIT_DIR/data/result_pytorch_waymo
ROOT_DIR=$SLURM_SUBMIT_DIR/data/pytorch_waymo_dataset
CKPT_DIR=$SLURM_SUBMIT_DIR/data/ckpts
CHUNK=16384

python eval_block_nerf.py --chunk $CHUNK --save_path $SAVE_PATH --root_dir $ROOT_DIR --ckpt_dir $CKPT_DIR # 3090ti