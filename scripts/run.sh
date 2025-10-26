#!/bin/bash


SEED=42

# 运行训练脚本
echo "Starting training with seed $SEED..."
python ../src/project.py --seed $SEED
echo "Training finished."