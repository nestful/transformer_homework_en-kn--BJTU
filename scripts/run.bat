@echo off

echo ======================================================
echo   Starting All Transformer Ablation Studies...
echo ======================================================

rem 运行 Python 版本的“总指挥”脚本
rem 它会自动完成所有消融实验
python src\run_experiments.py

echo ======================================================
echo   All experiments finished. Check results/ folder.
echo ======================================================


