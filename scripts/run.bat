@echo off

set SEED=42

rem 运行训练脚本
echo "Starting training with seed %SEED%..."
python ..\src\project.py --seed %SEED%
echo "Training finished."