# run_experiments.py
import subprocess
import sys

# 定义所有实验的配置
# 每个元组的第一个元素是实验名称，第二个是参数列表
SEED = 42

experiments = [
    (
        "baseline",
        ["--N", "6", "--h", "8", "--d_model", "256", "--d_ff", "512"]
    ),
    (
        "fewer_layers_N3",
        ["--N", "3", "--h", "8", "--d_model", "256", "--d_ff", "512"]
    ),
    (
        "fewer_heads_h2",
        ["--N", "6", "--h", "2", "--d_model", "256", "--d_ff", "512"]
    ),
    (
        "no_pe",
        ["--N", "6", "--h", "8", "--d_model", "256", "--d_ff", "512", "--no_pe"]
    ),
    (
        "smaller_dim",
        ["--N", "6", "--h", "8", "--d_model", "128", "--d_ff", "256"]
    ),
]

# 获取 python 解释器路径
python_executable = sys.executable

# 循环运行所有实验
for name, params in experiments:
    print("=" * 50)
    print(f"Starting Experiment: {name}")
    print("=" * 50)

    # 构建命令
    command = [
                  python_executable,
                  "src/main_ablation.py",
                  "--exp_name", name,
                  "--seed", str(SEED),
              ] + params

    # 执行命令
    # 使用 Popen 可以在 PyCharm 控制台中实时看到输出
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()  # 等待当前实验完成

    if process.returncode != 0:
        print(f"Experiment {name} failed with return code {process.returncode}")
        break  # 如果一个实验失败，可以选择停止后续实验

print("\nAll experiments completed!")