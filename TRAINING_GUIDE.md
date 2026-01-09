# CSV 2026 Baseline 训练指南

> 本指南基于 RTX 3090 (24GB) + CUDA 12.2 环境编写

---

## 第一步：创建 Python 环境

```bash
# 创建并激活 conda 环境
conda create -n csv-baseline python=3.10 -y
conda activate csv-baseline
```

---

## 第二步：安装 PyTorch

由于你的 CUDA 版本是 12.2，使用 cu121 版本的 PyTorch：

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

验证安装：
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

预期输出：
```
PyTorch: 2.5.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

---

## 第三步：安装依赖

```bash
pip install -r requirements.txt
```

---

## 第四步：准备数据

### 4.1 下载训练数据
1. 访问 [CSV 2026 官网](http://www.csv-isbi.net/) 注册账号
2. 下载训练数据压缩包
3. 解压到 `data/train/` 目录

### 4.2 下载预训练权重
1. 下载 [Echocare 预训练权重](https://cashkisi-my.sharepoint.com/:u:/g/personal/cares-copilot_cair-cas_org_hk/IQBgK6rK8TAtQq8IjADsgp52AbmyC03ubimwqr3qh8ZH6DI?e=ABYQzg)
2. 将 `echocare_encoder.pth` 放到 `pretrain/` 目录

### 4.3 验证目录结构
```
CSV-2026-Baseline/
├── data/
│   └── train/
│       ├── images/          # 包含 xxxx.h5 文件
│       └── labels/          # 包含 xxxx_label.h5 文件
├── pretrain/
│   └── echocare_encoder.pth # 预训练权重
├── train.py
├── inference.py
└── ...
```

---

## 第五步：划分数据集

```bash
python split_train_valid_fold.py --root ./data --seed 2026 --val_size 50
```

成功后会在 `data/` 目录生成：
- `train_labeled.json` - 有标签训练集
- `train_unlabeled.json` - 无标签训练集  
- `valid.json` - 验证集

---

## 第六步：开始训练

### 推荐配置（RTX 3090 24GB）

```bash
python train.py ^
  --train-labeled-json ./data/train_labeled.json ^
  --train-unlabeled-json ./data/train_unlabeled.json ^
  --valid-labeled-json ./data/valid.json ^
  --model Echocare ^
  --echo_care_ckpt ./pretrain/echocare_encoder.pth ^
  --save_path ./checkpoints ^
  --gpu 0 ^
  --train_epochs 100 ^
  --batch_size 8 ^
  --amp True
```

> 💡 **参数说明：**
> - `--batch_size 8`：RTX 3090 可以稳定运行，如显存不足可降到 4
> - `--amp True`：启用混合精度训练，加速并节省显存
> - `--train_epochs 100`：训练 100 轮

### 备选：轻量级 UNet（如需快速测试）

```bash
python train.py --model UNet --gpu 0 --batch_size 16
```

---

## 第七步：监控训练

### 使用 TensorBoard 查看训练曲线

```bash
tensorboard --logdir ./checkpoints/tensorboard
```

浏览器访问 `http://localhost:6006` 查看：
- 训练损失曲线
- 验证 Dice/NSD 指标
- 分类 F1 分数

---

## 第八步：推理与提交

### 8.1 对验证集推理

```bash
python inference.py ^
  --val-dir ./data/val ^
  --checkpoint ./checkpoints/best.pth ^
  --encoder-pth ./pretrain/echocare_encoder.pth ^
  --resize-target 256 ^
  --gpu 0
```

### 8.2 打包提交文件

```bash
cd data/val
tar -czvf preds.tar.gz preds/
```

将 `preds.tar.gz` 上传至比赛平台。

---

## 常见问题

### Q1: CUDA out of memory
```bash
# 减小 batch_size
--batch_size 4
```

### Q2: 训练中断如何恢复？
训练会自动保存 `latest.pth`，重新运行相同命令即可自动恢复。

### Q3: 如何查看 GPU 使用情况？
```bash
nvidia-smi -l 1  # 每秒刷新
```

---

## 预期训练时间

| 模型 | Batch Size | 每 Epoch 时间 | 100 Epochs 总时间 |
|------|------------|---------------|-------------------|
| Echocare | 8 | ~3-5 分钟 | ~5-8 小时 |
| UNet | 16 | ~1-2 分钟 | ~2-3 小时 |

---

## 检查点说明

训练完成后，`checkpoints/` 目录包含：
- `best.pth` - 综合得分最高的模型
- `best_seg.pth` - 分割得分最高的模型
- `best_cls.pth` - 分类得分最高的模型
- `latest.pth` - 最新的检查点（用于断点续训）

推荐使用 `best.pth` 进行最终推理提交。
