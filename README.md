# CAFormer-S24模型的复现

本仓库提供论文《A novel CNN-ViT-based deep learning model for early skin cancer diagnosis》中提出的CAFormer-S24架构的开源实现，便于在皮肤癌图像分类任务中复现其训练流程。

## 模型简介

- **混合特征抽取**：通过卷积干层捕获局部纹理细节，并在每个阶段使用带通道注意力的局部感知单元。
- **Transformer 表征**：将卷积特征分块为序列，经多层自注意力网络建模全局依赖，并通过可学习的类别标记聚合。
- **多模态融合头**：同时利用卷积特征的全局平均表示与 Transformer 的全局上下文特征，提高类别判别力。

## 代码结构

```
caformer/
  ├── __init__.py            # 暴露模型工厂函数
  ├── modules.py             # 基础模块（卷积、注意力、融合头等）
  └── model.py               # CAFormer 主体结构
configs/
  └── caformer_base.yaml     # 默认训练配置
train.py                     # 训练脚本
```

## 运行环境

- Python 3.10+
- PyTorch >= 2.1
- Torchvision >= 0.16

可以使用如下命令安装依赖：

```bash
pip install torch torchvision
```

## 数据准备

将皮肤癌图像数据集按照 ImageNet 风格组织：

```
dataset/
  ├── train/
  │   ├── class_0/
  │   ├── class_1/
  │   └── ...
  └── val/
      ├── class_0/
      ├── class_1/
      └── ...
```

## 训练

```bash
python train.py /path/to/dataset \
    --epochs 100 \
    --batch-size 32 \
    --lr 3e-4 \
    --num-classes 7
```

训练过程会在 `outputs/` 目录下保存最新与最佳模型权重，并以 JSON 行格式输出每个 epoch 的指标。

## 自定义配置

如需调整网络结构或训练超参，可参考 `configs/caformer_base.yaml` 修改参数，再通过 `build_caformer` 传入：

```python
from caformer import build_caformer
model = build_caformer(num_classes=3, stem_channels=(48, 96, 192))
```

## 许可

该实现仅用于学术研究，请遵循原论文及所用数据集的版权协议。
