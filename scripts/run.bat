@echo off

rem 切换到项目的根目录 (scripts 文件夹的上一级)
rem 这能保证无论你在哪里执行这个脚本，路径都是正确的
cd /d "%~dp0"
cd ..

echo ======================================================
echo   Starting All Transformer Ablation Studies...
echo ======================================================

rem 运行 Python 版本的“总指挥”脚本
rem 它会自动完成所有消融实验
python src\run_experiments.py

echo ======================================================
echo   All experiments finished. Check results/ folder.
echo ======================================================

rem 暂停一下，方便查看运行结果
pause
