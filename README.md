# 从零实现Transformer: 英-卡纳达语机器翻译

本项目是 Vaswani et al. (2017) 论文《Attention Is All You Need》中 Transformer 模型的完整 PyTorch 实现。该模型在一个公开的英语-卡纳达语平行语料库上进行了训练，用于完成一个简单的机器翻译任务。

项目的核心不仅在于实现了一个标准的 Encoder-Decoder Transformer，还包含了一系列详细的**消融实验**，以系统性地分析模型中各个关键组件（如层数、注意力头数、位置编码等）对最终性能的影响。

## 项目结构

```text
.
├── .gitignore          # Git忽略文件配置
├── data.zip            # 压缩的数据集 (需要解压)
├── data/               # 存放解压后的数据集 (train/, validation/, test/)
├── results/
│   ├── loss_curve_baseline.png
│   ├── loss_curve_fewer_heads_h2.png
│   ├── loss_curve_fewer_layers_N3.png
│   ├── loss_curve_no_pe.png
│   └── loss_curve_smaller_dim.png
├── scripts/
│   ├── run.sh          # [推荐] Linux/macOS 一键运行所有实验的脚本
│   └── run.bat         # [推荐] Windows 一键运行所有实验的脚本
├── src/
│   ├── main_ablation.py  # 核心模型定义与参数化训练代码
│   ├── run_experiments.py# 驱动所有消融实验的Python启动器
│   └── predict.py        # 用于加载模型进行翻译预测的脚本
├── README.md             # 本说明文件
└── requirements.txt      # Python环境依赖
```
> **注意**: `models/` 目录将在第一次运行训练时自动创建，用于存放训练好的 `.pth` 模型文件。

## 环境配置与安装

### 1. 克隆仓库```bash
git clone https://github.com/nestful/transformer_homework_en-kn--BJTU.git
cd transformer_homework_en-kn--BJTU
```

### 2. 解压数据集
`data` 文件夹由 `data.zip` 提供。请先解压它。
```bash
unzip data.zip
# 如果没有unzip, 可以手动解压
```

### 3. 创建Conda环境 (推荐)
```bash
# 创建并激活 Conda 环境
conda create -n transformer_env python=3.10 -y
conda activate transformer_env
```

### 4. 安装依赖
所有需要的库都已在 `requirements.txt` 中列出。
```bash
pip install -r requirements.txt
```

## 如何运行实验

### 方式一：一键运行所有消融实验 (推荐)

为了方便地复现报告中的所有实验结果，请使用 `scripts` 文件夹下的启动脚本。该脚本会自动依次执行基线模型和所有消融实验的训练。

*   **在 Linux / macOS 系统下:**
    ```bash
    bash scripts/run.sh
    ```

*   **在 Windows 系统下:**
    ```bash
    .\scripts\run.bat
    ```
    (或者直接双击 `run.bat` 文件)

训练过程中的所有输出（模型文件和损失曲线图）将分别保存在自动创建的 `models/` 和已有的 `results/` 目录下。

### 方式二：手动运行单个实验

如果你只想运行某一个特定的实验（例如，只复现基线模型），可以直接调用 `main_ablation.py` 脚本并传入相应参数。

**复现基线模型的精确命令 (含随机种子):**
```bash
python src/main_ablation.py --exp_name "baseline" --N 6 --h 8 --d_model 256 --d_ff 512 --epochs 50 --seed 42
```

你可以通过修改参数来进行其他实验，例如运行一个没有位置编码的实验：```bash
python src/main_ablation.py --exp_name "no_pe_manual" --no_pe --seed 42
```

## 使用已训练模型进行预测

训练完成后，你可以使用 `predict.py` 脚本来加载任意一个已保存的模型，并对新的英文句子进行翻译。

**使用示例:**
```bash
# 确保 models/transformer_baseline.pth 文件已存在
python src/predict.py --model_path "models/transformer_baseline.pth" --text "hello world"
```

## 实验结果

所有实验的训练和验证损失曲线图都保存在 `results/` 文件夹中。关于这些结果的详细定量和定性分析，请参阅本项目的 PDF 实验报告。
