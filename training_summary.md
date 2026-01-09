# CSV-2026 Baseline 训练总结报告

## 1. 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | Echocare (175.1M 参数) |
| Batch Size | 4 |
| 学习率 | 0.0001 (cosine decay) |
| 训练轮数 | 100 epochs |
| 混合精度 | FP16 |
| 预训练权重 | echocare_encoder.pth |
| 验证集 | 50 samples (25 class0 + 25 class1) |

## 2. 最佳性能

### 综合最佳 (Epoch 13, total_score=0.49)
- Mean Foreground Dice: 0.692
- Mean Foreground NSD: 0.423
- F1 Score: 0.6909

### 分割最佳 (Epoch 29, seg_score=0.5484)
| 指标 | Long | Trans |
|------|------|-------|
| Plaque Dice | 0.59 | 0.52 |
| Plaque NSD | 0.41 | 0.34 |
| Vessel Dice | 0.90 | 0.82 |
| Vessel NSD | 0.53 | 0.44 |
| **Mean Dice** | **0.709** | |
| **Mean NSD** | **0.429** | |

### 分类最佳 (Epoch 3, cls_score=0.6957)
- F1 Score: 0.6957
- Class 0 准确率: 20% (5/25)
- Class 1 召回率: 96% (24/25)

## 3. 训练曲线分析

### Loss 收敛情况
- 初始 Total Loss: ~1.99 → 最终: ~0.007
- Loss_x (主损失): 1.60 → 0.006
- Loss_s (半监督): 0.40 → 0.002
- Loss_fp: 0.38 → 0.000
- **收敛良好，无明显过拟合**

### MaskRatio 变化
- Epoch 1: 0.79 → Epoch 100: 0.999
- 伪标签置信度逐步提升，半监督学习生效

## 4. 关键观察

### 分割任务
- **Vessel 分割效果好**: Long Dice 稳定在 0.90，Trans Dice ~0.80
- **Plaque 分割较难**: Long Dice ~0.58，Trans Dice ~0.52
- **Long 视图优于 Trans 视图**: 约 10% 的 Dice 差距

### 分类任务
- F1 Score 波动较大 (0.47~0.69)
- 模型倾向于预测 class 1 (高风险)
- 最终稳定在 F1 ~0.57

### 训练稳定性
- Epoch 13 后 best score 未更新 (0.49)
- 分割持续改进至 Epoch 29
- 后期 (Epoch 30+) 进入平台期

## 5. 改进建议

1. **分类任务**
   - 增加分类损失权重
   - 尝试 Focal Loss 处理类别不平衡
   - 考虑分类头单独微调

2. **分割任务**
   - Plaque 分割可尝试更强的数据增强
   - Trans 视图可增加专门的注意力机制

3. **训练策略**
   - 学习率可尝试 warmup
   - 早停可设在 Epoch 50 左右
   - 增加验证集样本量

## 6. 输出文件

| 文件 | 说明 |
|------|------|
| best.pth | 综合最佳模型 (Epoch 13) |
| best_seg.pth | 分割最佳模型 (Epoch 29) |
| best_cls.pth | 分类最佳模型 (Epoch 3) |

## 7. 训练时长

- 总时长: ~13.5 小时 (13:38 → 02:52)
- 单 Epoch: ~8 分钟
