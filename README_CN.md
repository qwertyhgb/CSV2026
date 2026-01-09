# 🩺 CSV 2026 挑战赛 Baseline

[![CSV 2026 Challenge Site](https://img.shields.io/badge/Official-CSV%202026%20Challenge-red?style=for-the-badge)](http://www.csv-isbi.net/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](./LICENSE)

**🏆 CSV 2026 挑战赛官方 Baseline** — 本仓库提供 **CSV 挑战赛** 的官方基线实现，采用半监督 UniMatch 框架进行 **超声图像中颈动脉斑块分割与易损性评估**，支持两种骨干网络：轻量级 UNet 和高性能 Echocare 编码器（基于 SwinUNETR）。官方规则、数据集下载及评测服务器请访问 [CSV 2026 挑战赛官网](http://www.csv-isbi.net/)

---

## 📁 1. 数据准备

将下载的 🖼️ 训练数据压缩包（由组委会提供）解压至 `data/` 目录。

**🤖 预训练权重**：如需使用 Echocare 模型训练，请从 [此链接](https://cashkisi-my.sharepoint.com/:u:/g/personal/cares-copilot_cair-cas_org_hk/IQBgK6rK8TAtQq8IjADsgp52AbmyC03ubimwqr3qh8ZH6DI?e=ABYQzg) 下载预训练的 Echocare 编码器权重，并将 `echocare_encoder.pth` 文件放置于 `pretrain/` 目录。最终目录结构如下：

```text
CSV2026_Baseline/
├─ data/
|  └─ train/
|     ├─ images/        # .h5 图像文件 (long_img & trans_img)
|     └─ labels/        # _label.h5 标签文件 (long_mask, trans_mask, cls)
└─ pretrain/ 
   └─ echocare_encoder.pth    # Echocare 预训练编码器权重
```

## 🧰 2. 快速开始

推荐使用 **Python 3.10** 和 **CUDA 12.1 (cu121)**。
最低 PyTorch 版本要求：**>= 2.4.1**。

```bash
# 📥 克隆仓库
git clone https://github.com/dndins/CSV-2026-Baseline.git
cd CSV-2026-Baseline

# 🎯 创建 Python 环境
conda create -n csv-baseline python=3.10 -y
conda activate csv-baseline

pip install --index-url https://download.pytorch.org/whl/cu121   torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

pip install -r requirements.txt
```

> ⚠️ **注意：**  
> 如果您的 CUDA 版本不同，请先按照官方说明安装 PyTorch，然后运行：
> ```bash
> pip install -r requirements.txt --no-deps
> ```


## 🧩 3. 创建本地训练/验证集划分

生成类别平衡的验证集和 JSON 划分文件：

```bash
python split_train_valid_fold.py --root ./data --seed 2026 --val_size 50
```

此命令将在 `data/` 目录下创建：
- `train_labeled.json` - 有标签训练集
- `train_unlabeled.json` - 无标签训练集
- `valid.json` - 验证集


## 🚀 4. 训练

### 推荐方案（高性能 — Echocare）

```bash
python train.py \
  --train-labeled-json ./data/train_labeled.json \
  --train-unlabeled-json ./data/train_unlabeled.json \
  --valid-labeled-json ./data/valid.json \
  --model Echocare \
  --echo_care_ckpt ./pretrain/echocare_encoder.pth \
  --save_path ./checkpoints \
  --gpu 0 \
  --train_epochs 100 \
  --batch_size 4
```

### 轻量方案（低显存 — UNet）

```bash
python train.py --model UNet --gpu 0
```

训练检查点保存位置：
```text
./checkpoints/best.pth
./checkpoints/latest.pth
```


## 🔍 5. 推理

对发布的验证集图像进行推理：

```bash
python inference.py \
  --val-dir ./data/val \
  --checkpoint ./checkpoints/best.pth \
  --encoder-pth ./pretrain/echocare_encoder.pth \
  --resize-target 256 \
  --gpu 0
```

预测结果保存至：

```text
./data/val/preds/{case_name}_pred.h5
```

每个文件包含：
- `long_mask` - 纵切面分割掩码
- `trans_mask` - 横切面分割掩码
- `cls` - 分类结果


## 📦 6. 打包预测结果提交

**⚠️ 注意！**

此提交方式适用于验证阶段。在此阶段，组委会将提供验证集图像。
参赛者需将验证结果保存为 H5 格式（与训练集标签格式相同），
并打包为 "preds.tar.gz" 格式通过注册平台提交。

```bash
cd data/val
tar -czvf preds.tar.gz preds/
```

在测试阶段，参赛者需以 Docker 格式提交方法。


## 🧠 7. 注意事项与技巧

- 🤖 使用 `--model Echocare` 获得最佳性能
- 🤖 显存有限时使用 `--model UNet`
- 🎯 根据 GPU 显存调整 `--batch-size`（Echocare 建议 8，UNet 建议 32）
- 🔧 使用 `--amp` 启用自动混合精度加速训练
- 📊 使用 TensorBoard 监控训练进度


## 📚 Baseline 方法致谢

本 Baseline 基于以下工作开发：

> **[1]** Yang L, Qi L, Feng L, et al.  
> *Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation.*  
> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

> **[2]** Zhang H, Wu Y, Zhao M, et al.  
> *A Fully Open and Generalizable Foundation Model for Ultrasound Clinical Applications.*  
> arXiv preprint arXiv:2509.11752, 2025.

> **[3]** Hatamizadeh A, Nath V, Tang Y, et al.  
> *Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images.*  
> International MICCAI brainlesion workshop. Cham: Springer International Publishing, 2021: 272-284.

> **[4]** Hu E J, Shen Y, Wallis P, et al.  
> *Lora: Low-rank adaptation of large language models.*  
> ICLR, 2022, 1(2): 3.

本 Baseline 的训练方法基于 [1] 中的方法构建。
Echocare [2] 是基于 Swin UNETR [3] 架构训练的自监督超声基础模型。
我们使用 LoRA [4] 对 Echocare 编码器进行微调以适应本任务。

---

## 🏁 祝您好运，科研顺利！

期待您参与 CSV 2026 挑战赛 🚀
