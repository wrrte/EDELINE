#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

python src/main.py env.train.id=FrostbiteNoFrameskip-v4 common.seed=42 training.num_workers_data_loaders=4 retrieval.enable=False