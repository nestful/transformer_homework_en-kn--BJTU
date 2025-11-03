# 基于Transformer架构的英-卡纳达语翻译

本项目从零开始使用 PyTorch 实现了一个完整的 Encoder-Decoder Transformer 模型，用于英语到卡纳达语的机器翻译任务。

## 项目结构
```text
├── .gitignore # Git忽略文件配置
├── data/ # 存放数据集的目录
├── results/
│ └── loss_curve.png # 训练损失曲线图
├── scripts/
│ ├── run.sh # Linux/macOS 运行脚本
│ └── run.bat # Windows 运行脚本
├── src/
│ ├── project.py # 核心模型定义与训练代码
│ └── predict.py # 预测/翻译脚本
├── README.md # 本说明文件
└── requirements.txt # Python环境依赖
```
## 环境设置

建议使用 Conda 创建虚拟环境，并确保你的机器拥有支持 CUDA 的 NVIDIA GPU。

1.  **创建并激活 Conda 环境:**
    ```bash
    conda create -n transformer_gpu python=3.10 -y
    conda activate transformer_gpu
    ```

2.  **安装依赖:**
    本项目依赖 PyTorch 的特定 GPU 版本。请使用 `pip` 安装 `requirements.txt` 中的所有依赖。
    ```bash
    pip install -r requirements.txt
    ```
    *注意：`requirements.txt` 中包含了与 CUDA 11.8 兼容的 PyTorch 版本。如果你的 CUDA 版本不同，请访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 获取对应的安装命令。*

## 使用说明

### 数据集准备

请将英语-卡纳达语平行语料库解压后，确保 `train`, `validation`, `test` 三个子文件夹位于项目根目录下的 `data/` 文件夹中。

数据集下载链接：https://huggingface.co/datasets/Helsinki-NLP/opus-100

### 模型训练

要从头开始训练模型，请运行脚本。脚本会自动开始训练，并将最终的模型 (`transformer_en_kn.pth`) 和损失曲线图 (`results/loss_curve.png`)保存在相应的位置。

- **在 Windows 上:**
  ```bash
  cd scripts
  run.bat

- **在 Linux 或 macOS 上:**

  codeBash

  ```
  chmod +x scripts/run.sh
  ./scripts/run.sh
  ```

### 进行翻译

使用 predict.py 脚本来加载已训练好的模型并翻译新的句子。

codeBash

```
python src/predict.py
```

## 复现实验

为了精确复现我的实验结果，请使用以下确切的命令行。这里我们设定了随机种子为 42。

- **Windows:** python ..\src\project.py --seed 42
- **Linux/macOS:** python ../src/project.py --seed 42

**硬件要求:**

- 训练过程在 NVIDIA GeForce RTX 3060 Laptop GPU 上完成。
- BATCH_SIZE 设置为 16。
- 训练 50 个 epoch 大约耗时 5小时。

## 实验结果

模型的训练和验证损失曲线如下所示。可以看出，模型在训练集上有效学习，但在验证集上出现了过拟合现象。

![image-20251026182323036](README.assets/image-20251026182323036.png)
