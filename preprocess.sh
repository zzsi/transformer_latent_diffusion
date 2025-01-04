#!/usr/bin/env bash
#
# Train model

# This is the username in Dockerfile.
USER=cvl
GPU_DEVICES=${GPU_DEVICES:-0}  # default GPU id

# Create a data/container_cache directory if it doesn't exist
mkdir -p data/container_cache
# You may need to run this command to fix the permissions:
# sudo chmod a+rw -R data/container_cache

docker run --runtime nvidia -it --rm \
	--shm-size 16G \
	--gpus "device=${GPU_DEVICES}" \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/home/$USER/.cache \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	tld \
	python -m tld.preprocess_data $@

# For diagnostic run, pass in the following args:
# --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61

