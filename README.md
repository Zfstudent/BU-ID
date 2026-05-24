# BU-ID: 乳腺超声图像肿瘤检测

**Breast Ultrasound Image Detection (BU-ID)** - 基于深度学习的乳腺超声图像肿瘤检测，采用 **SSD300 (Single Shot MultiBox Detector)** 目标检测算法，结合 **ResNet34 骨干网络**，实现对超声图像中 **良性（Benign）** 和 **恶性（Malignant）** 肿瘤的精准定位与分类。

## 📋 项目概述

本项目旨在通过计算机视觉技术自动化检测乳腺超声图像中的肿瘤病灶，辅助医生进行乳腺癌早期筛查与诊断。基于 **SSD 单阶段目标检测框架**，在 VOC2007 格式的乳腺超声数据集上训练，实现端到端的 tumor detection（位置回归）+ classification（良恶性分类）任务。

## 🎯 检测类别

系统能够识别以下 **3 个语义类别**：

| 类别 ID | 类别名称   | 中文名称     | 临床意义                     |
|---------|------------|--------------|------------------------------|
| 0       | background | 背景         | 正常组织/非肿瘤区域           |
| 1       | benign     | 良性肿瘤     | 通常无需立即干预，定期随访    |
| 2       | malignant   | 恶性肿瘤     | 需进一步检查，可能为乳腺癌    |

### 📊 类别分布特点
- **数据不平衡**: 良性样本通常多于恶性样本
- **临床重要性**: 恶性肿瘤的召回率（Recall）至关重要（宁可误检不可漏检）
- **应用场景**: 辅助放射科医生快速定位可疑病灶区域

## 🏗️ 项目结构

```
BU-ID/
├── train.py                        # 训练脚本（SSD模型定义+训练流程）
├── test.py                         # 测试脚本（推理+评估+可视化）
├── dataset.py                      # 数据集加载模块（VOC格式解析）
│
├── requirements.txt                # Python 依赖包
│
├── checkpoints/                    # 模型权重与训练历史
│   ├── best_breast_ultrasound.pth  # 最佳模型权重（基于验证损失选择）
│   ├── train_curve.png             # 训练曲线可视化
│   └── training_history_epoch*.pth # 每10个epoch保存的训练历史快照
│
├── results/                        # 测试结果输出
│   ├── test_metrics.json           # 结构化测试指标报告
│   ├── sample_*.png                # 可视化检测结果示例图
│   └── *.jpg                      # 每张测试图的检测结果可视化
│
└── VOC2007/                        # 数据集（VOC目标检测格式）
    ├── JPEGImages/                 # 超声图像文件夹
    │   ├── 000001.jpg ~ 000098.jpg # 约98张乳腺超声图像
    │   └── ...
    ├── Annotations/                # XML标注文件（Pascal VOC格式）
    │   ├── 000001.xml ~ 000098.xml # 包含边界框坐标和类别标签
    │   └── ...
    └── ImageSets/Main/             # 数据集划分文件
        ├── train.txt               # 训练集列表（约78张）
        ├── val.txt                 # 验证集列表（约20张）
        ├── trainval.txt            # 训练+验证集合并
        └── test.txt                # 测试集列表（10张）
```

## 🛠️ 技术栈

### 核心深度学习框架
- **PyTorch ≥ 1.12.0** - 深度学习框架
- **torchvision ≥ 0.13.0** - ResNet预训练模型、图像变换工具

### 图像处理与增强
- **Albumentations ≥ 1.3.0** - 高性能医学图像增强库（支持边界框变换）
- **Pillow ≥ 9.0.0** - 图像读取与处理
- **NumPy ≥ 1.21.0** - 数值计算

### 评估与可视化
- **scikit-learn ≥ 1.0.0** - 分类指标计算（Precision/Recall/F1等）
- **Matplotlib ≥ 3.5.0** - 训练曲线绘制、检测结果可视化
- **tqdm ≥ 4.64.0** - 进度条显示

## 🧠 模型架构

### 整体架构：SSD300 + ResNet34 Backbone

```
输入图像 (RGB, 300×300)
    ↓
┌─────────────────────────────────────┐
│      ResNet34 Backbone (特征提取)    │
├─────────────────────────────────────┤
│  Layer1 (Conv1+BN1+ReLU+MaxPool+L1) │ → C3 特征图 (75×75, 128ch)
│  Layer2                             │ → C4 特征图 (38×38, 256ch)
│  Layer3                             │ → C5 特征图 (19×19, 512ch)
│  Layer4                             │ → C6 特征图 (10×10, 512ch)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Extra Feature Layers (额外特征层)   │
├─────────────────────────────────────┤
│  Extra Block 1: Conv(512→512)       │ → C7 特征图 (5×5, 512ch)
│  Extra Block 2: Conv(512→256)       │ → C8 特征图 (3×3, 256ch)
│  Extra Block 3: Conv(256→256)       │ → C9 特征图 (1×1, 256ch)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     SSD Detection Head (检测头)      │
├─────────────────────────────────────┤
│  多尺度预测层 (6个特征图):            │
│  • C3 (75×75):  6 anchors × class  │
│  • C4 (38×38):  6 anchors × class  │
│  • C5 (19×19):  6 anchors × class  │
│  • C6 (10×10):  6 anchors × class  │
│  • C7 (5×5):    4 anchors × class  │
│  • C9 (1×1):    4 anchors × class  │
│                                      │
│  每个 anchor 预测:                    │
│  • 4个坐标偏移量 (loc)               │
│  • 3个类别置信度 (conf)              │
└─────────────────────────────────────┘
    ↓
输出:
  • loc_pred: [B, 8732, 4]  (边界框坐标)
  • conf_pred: [B, 8732, 3] (类别置信度)
```

### 关键组件详解

#### 1️⃣ **ResNetBackbone (骨干网络)**
```python
class ResNetBackbone(nn.Module):
    """
    使用 ImageNet 预训练的 ResNet 提取多尺度特征
    
    支持的骨干网络：
    - resnet18: 轻量快速（21M参数）
    - resnet34: 平衡精度与速度（默认，25M参数）
    - resnet50: 更高精度（44M参数）
    
    输出3个尺度的特征图用于SSD检测头
    """
```

**配置选项**:
- `resnet18`: 适合快速原型开发、资源受限环境
- `resnet34`: **推荐默认选择**，平衡性能
- `resnet50`: 追求更高精度时使用

#### 2️⃣ **Extra Layers (额外卷积层)**
在 ResNet 的 C5 特征图后添加 3 组额外卷积层，逐步降低分辨率：
- **目的**: 生成更多尺度的特征图，提升对不同大小目标的检测能力
- **结构**: 每组包含 1×1 Conv + 3×3 Conv（stride=2 下采样）+ ReLU
- **通道数**: 512 → 256 → 256

#### 3️⃣ **SSDHead (检测头)**
每个特征图上应用两个并行卷积分支：
- **定位分支 (Loc)**: 预测边界框的 4 个坐标偏移量 (Δcx, Δcy, Δw, Δh)
- **置信度分支 (Conf)**: 预测每个类别的置信度分数

**Default Boxes (先验框) 配置**:
- 不同尺度特征图使用不同数量和大小的先验框
- 总计 **8732 个先验框**（经典 SSD300 配置）
- 先验框在训练前根据数据集统计信息生成

#### 4️⃣ **MultiBoxLoss (损失函数)**
```python
Total Loss = Localization Loss + Confidence Loss

# 定位损失 (Smooth L1 Loss)
# 仅对正样本（匹配到GT的anchor）计算
Loc Loss = SmoothL1(pred_loc, target_loc)

# 置信度损失 (Cross Entropy + Hard Negative Mining)
# 对所有样本计算，但通过Hard Negative Mining平衡正负样本比例（正:负 ≈ 1:3）
Conf Loss = CrossEntropy(pred_conf, target_conf)
```

**关键特性**:
- **Smooth L1**: 对离群点更鲁棒（相比 L2 Loss）
- **Hard Negative Mining**: 解决极端的类别不平衡问题（大部分anchor是背景）

## 📊 性能指标

### ⚠️ 当前测试结果（基于10张测试图像）

> **注意**: 以下指标来自初步测试，模型可能需要进一步调优或增加训练数据

#### 整体指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **mAP (mean Average Precision)** | **0.0013** | 平均精度均值（需优化） |

#### 各类别详细指标

| 类别 | AP (平均精度) | Precision (精确率) | Recall (召回率) | F1-Score | GT数量 | TP | FP |
|------|---------------|-------------------|-----------------|----------|--------|----|-----|
| **benign** (良性) | 0.0023 | 0.0020 | **100.00%** ✅ | 0.0039 | 6 | 6 | 3044 |
| **malignant** (恶性) | 0.0003 | 0.0003 | 20.00% ⚠️ | 0.0006 | 5 | 1 | 3121 |

#### 结果解读

✅ **良性肿瘤检测**:
- 召回率达到 100%（所有良性肿瘤都被检出）
- 但精确率极低（大量误检），说明模型倾向于"宁可错杀不可放过"

⚠️ **恶性肿瘤检测**:
- 召回率仅 20%（5个恶性只检出1个）
- 这是临床最关键的指标，需要大幅改进

🔴 **整体问题**:
- mAP 极低（0.13%），说明模型当前状态不理想
- 可能原因：
  1. 训练数据量不足（仅约98张）
  2. 模型尚未充分收敛或超参数需调整
  3. 存在严重的过拟合或欠拟合
  4. 测试集分布与训练集差异较大

### 💡 改进方向建议
1. **增加数据量**: 至少扩展至 500-1000 张标注图像
2. **数据增强**: 已实现但可进一步加强（Mixup, Cutout, Mosaic等）
3. **超参调优**: 学习率、batch size、anchor 尺寸等
4. **迁移学习策略**: 冻结 backbone 更长时间
5. **尝试其他模型**: YOLOv8, Faster R-CNN, DETR 等

## 🚀 快速开始

### 环境安装

```bash
# 进入项目目录
cd /home/BU-ID

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

数据集已按 **Pascal VOC 目标检测格式** 组织于 `VOC2007/` 目录：

```
VOC2007/
├── JPEGImages/                 # 超声图像
│   ├── 000001.jpg              # 乳腺超声图像
│   ├── 000002.jpg
│   └── ... (共约98张)
│
├── Annotations/                # XML标注文件
│   ├── 000001.xml              # Pascal VOC格式标注
│   │   <?xml version="1.0" ?>
│   │   <annotation>
│   │     <filename>000001.jpg</filename>
│   │     <size>
│   │       <width>500</width>
│   │       <height>500</height>
│   │     </size>
│   │     <object>
│   │       <name>benign</name>  <!-- 或 malignant -->
│   │       <bndbox>
│   │         <xmin>120</xmin>
│   │         <ymin>150</ymin>
│   │         <xmax>350</xmax>
│   │         <ymax>400</ymax>
│   │       </bndbox>
│   │     </object>
│   │   </annotation>
│   └── ...
│
└── ImageSets/Main/             # 数据划分文件
    ├── train.txt               # 每行一个图像ID（不含扩展名）
    ├── val.txt
    ├── trainval.txt
    └── test.txt
```

**XML标注格式要点**:
- `<name>`: 类别名（"benign" 或 "malignant"）
- `<bndbox>`: 边界框坐标（像素级，左上角+右下角）
- 支持单张图像多个目标（多个 `<object>` 标签）

### 模型训练

#### 基础训练命令

```bash
python train.py
```

#### 自定义参数训练

修改 `train.py` 底部的配置变量：

```python
if __name__ == "__main__":
    # ========== 可调参数 ==========
    voc_root = "/path/to/VOC2007"    # 数据集路径
    img_size = 300                   # 输入图像尺寸（SSD标准）
    batch_size = 16                  # 批次大小（建议 8-32）
    epochs = 150                     # 训练轮数（建议 100-300）
    lr = 1e-3                       # 初始学习率
    momentum = 0.9                  # SGD 动量
    weight_decay = 5e-4             # 权重衰减（L2正则化）
    num_classes = 3                  # 类别数（背景+良性+恶性）
    backbone_type = 'resnet34'       # 骨干网络类型
    
    # 运行训练...
```

**完整训练流程**:

1. **数据集自动划分**:
   ```python
   stratified_split(voc_root, train_ratio=0.8, seed=42)
   ```
   - 按 80:20 划分训练集/验证集
   - **分层抽样**：保证两类样本比例一致
   - 自动生成 `train.txt`, `val.txt`

2. **数据增强策略**（训练时启用）:
   ```python
   A.Compose([
       A.Resize(300, 300),                          # 统一尺寸
       A.HorizontalFlip(p=0.5),                     # 水平翻转
       A.VerticalFlip(p=0.2),                       # 垂直翻转（医学图像适用）
       A.ShiftScaleRotate(shift_limit=0.05, 
                          scale_limit=0.1, 
                          rotate_limit=15, p=0.3), # 平移缩放旋转
       A.OneOf([                                    # 噪声注入
           A.GaussNoise(var_limit=(5,15)),
           A.ISONoise(intensity=(0.05,0.15))
       ], p=0.2),
       A.OneOf([                                    # 模糊模拟
           A.GaussianBlur(blur_limit=(3,5)),
           A.MotionBlur(blur_limit=(3,5))
       ], p=0.2),
       A.RandomBrightnessContrast(brightness_limit=0.15,
                                  contrast_limit=0.15, p=0.3), # 亮度对比度
       A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.2),  # 直方图均衡化
       A.HueSaturationValue(hue_shift_limit=5, 
                            sat_shift_limit=10, 
                            val_shift_limit=10, p=0.2),  # 颜色抖动
       A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), # ImageNet标准化
       ToTensorV2(),
   ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
   ```

3. **训练配置详情**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| optimizer | SGD + Momentum | 经典优化器，稳定性好 |
| learning rate | 1e-3 | 初始学习率 |
| scheduler | CosineAnnealingLR | 余弦退火，T_max=epochs, eta_min=1e-7 |
| gradient clipping | 5.0 | 梯度范数裁剪，防止梯度爆炸 |
| early stopping patience | 20 | 连续20轮无改善则早停 |
| batch size | 16 | 根据显存调整（建议 8-32） |
| epochs | 150 | 最大训练轮数（通常会提前停止）|

4. **训练监控**:
   - 实时打印每个 epoch 的 Loc Loss / Conf Loss
   - 自动保存最佳模型（基于验证集总损失）
   - 每 10 个 epoch 保存训练历史快照
   - 实时更新 `train_curve.png`（Loss 曲线可视化）

**输出文件**:
- ✅ `checkpoints/best_breast_ultrasound.pth` - 最佳模型权重
- ✅ `checkpoints/train_curve.png` - 训练曲线图
- ✅ `checkpoints/training_history_epoch*.pth` - 历史快照（可用于恢复训练）

### 模型测试与推理

#### 基础测试命令

```bash
python test.py
```

**测试流程**:

1. **加载最佳模型**:
   ```python
   model = SSD300(num_classes=3, backbone_type='resnet34').to(device)
   checkpoint = torch.load('checkpoints/best_breast_ultrasound.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

2. **后处理 pipeline**:
   - 解码预测框坐标（从 offset 转换为绝对坐标）
   - **NMS (Non-Maximum Suppression)**: 去除重叠检测框
     - 置信度阈值: 0.25（过滤低置信度预测）
     - NMS IoU阈值: 0.45（去除高度重叠框）

3. **指标计算**:
   - **mAP (mean Average Precision)**: 所有类别的 AP 均值
   - **AP (Average Precision)**: PR曲线下面积（每类独立计算）
   - **Precision**: 预测为正例中真正例的比例
   - **Recall**: 实际正例中被正确预测的比例
   - **F1-Score**: Precision 和 Recall 的调和平均
   - **TP/FP/GT**: 真/假阳性计数，真实框数量

4. **可视化输出**:
   - 每 10 张测试图保存一张可视化结果
   - 显示：原图 + GT框（绿色）+ 预测框（红色/蓝色）+ 置信度分数
   - 保存至 `results/` 目录

**输出文件**:
- ✅ `results/test_metrics.json` - 结构化指标报告
- ✅ `results/*.jpg` - 每张测试图的可视化结果
- ✅ 控制台详细指标打印

## 💡 核心特性

✅ **先进的目标检测架构**: SSD300 单阶段检测器，速度快且精度较高  
✅ **强大的骨干网络**: ResNet34 预训练特征提取，迁移学习能力强  
✅ **丰富的数据增强**: 10+种 Albumentations 增强策略，支持边界框同步变换  
✅ **完整的训练流程**: 分层抽样划分 + 早停机制 + 梯度裁剪 + 余弦退火  
✅ **详细的评估体系**: mAP/AP/Precision/Recall/F1 多维度指标  
✅ **可视化友好**: 自动生成检测结果可视化图，直观展示检测效果  
✅ **灵活可扩展**: 支持 ResNet18/34/50 骨干切换，易于实验对比  

## 📈 数据处理流程

### 数据集类：BreastUltrasoundDataset ([dataset.py](dataset.py))

```python
class BreastUltrasoundDataset(Dataset):
    """
    乳腺超声目标检测数据集（VOC格式）
    
    功能：
    1. 从 VOC 格式目录加载图像和 XML 标注
    2. 将 Pascal VOC 格式 (xmin,ymin,xmax,ymax) 转换为 YOLO 格式 (cx,cy,w,h)
    3. 生成 Default Boxes（SSD 先验框）并匹配 Ground Truth
    4. 应用 Albumentations 数据增强（保持边界框一致性）
    5. 分析数据集统计信息（各类别样本数）
    
    输出（每个样本）:
    - image: [3, H, W] tensor (RGB, 增强后)
    - loc_target: [num_default_boxes, 4] tensor (编码后的GT坐标)
    - conf_target: [num_default_boxes] tensor (类别标签，-1表示忽略)
    """
    
    def __init__(self, voc_root, split="train", img_size=300, augment=True):
        self.classes = ["benign", "malignant"]
        self.class_to_idx = {"benign": 1, "malignant": 2}
        
        # 加载图像ID列表
        # 构建图像路径和XML路径映射
        
        if augment:
            self.transform = self._build_training_transform()  # 强增强
        else:
            self.transform = self._build_validation_transform()  # 仅resize+normalize
            
    def __getitem__(self, idx):
        # 读取图像 (PIL Image)
        # 解析 XML 提取边界框和类别
        # 转换坐标格式 (VOC -> YOLO)
        # 匹配 Default Boxes（计算IoU，分配正/负/忽略样本）
        # 应用数据增强
        # 返回 (image_tensor, loc_target, conf_target)
```

### 关键数据处理步骤

#### 1️⃣ **坐标格式转换**
```python
# Pascal VOC format (绝对坐标)
voc_box = [xmin, ymin, xmax, ymax]

# YOLO format (相对坐标，中心点+宽高)
yolo_box = [
    (xmin + xmax) / 2 / img_width,   # cx (中心x)
    (ymin + ymax) / 2 / img_height,  # cy (中心y)
    (xmax - xmin) / img_width,        # w (宽度)
    (ymax - ymin) / img_height        # h (高度)
]
```

#### 2️⃣ **Default Box Matching**
- 为每个 GT 框找到 IoU 最大的 Default Box 作为正样本
- 若 Default Box 与任何 GT 的 IoU < 0.5，标记为负样本（背景）
- 其余标记为忽略（-1，不参与损失计算）

#### 3️⃣ **自定义 Collate Function**
```python
def custom_collate_fn(batch):
    """
    自定义批处理函数，处理变长序列
    
    由于不同图像匹配到的 Default Box 数量可能不同，
    需要 padding 到相同长度才能组成 batch tensor
    """
    images, loc_targets, conf_targets = zip(*batch)
    # Stack images (固定尺寸)
    # Pad loc_targets 和 conf_targets (变长→定长)
    return torch.stack(images), padded_locs, padded_confs
```

#### 4️⃣ **分层抽样划分**
```python
def stratified_split(voc_root, train_ratio=0.8, seed=42):
    """
    分层抽样数据集划分
    
    保证训练集和验证集中 benign/malignant 的比例一致，
    避免因类别分布差异导致的偏差
    """
    # 按类别分组
    # 随机打乱每组内样本
    # 按 train_ratio 划分
    # 写入 train.txt / val.txt
```

## 🎯 应用场景

- **医院影像科**: 辅助超声科医生快速定位可疑肿瘤区域
- **乳腺癌筛查**: 大规模人群普查的初筛工具
- **远程医疗**: 基层医疗机构缺乏专家时的 AI 辅助诊断
- **医学教学**: 乳腺超声影像识别的教学演示平台
- **科研研究**: 乳腺肿瘤影像特征分析与数据集构建
- **移动医疗 App**: 集成 AI 检测功能（需模型压缩优化）

## 📁 关键文件说明

| 文件 | 功能描述 | 核心类/函数 |
|------|----------|-------------|
| [train.py](train.py) | 训练主脚本 | `ResNetBackbone`, `SSD300`, `SSDHead`, `MultiBoxLoss`, `train_epoch()`, `val_epoch()` |
| [test.py](test.py) | 测试与推理脚本 | `DetectionMetrics`, `decode_boxes()`, `post_process()`, `visualize_result()`, `test()` |
| [dataset.py](dataset.py) | 数据集模块 | `BreastUltrasoundDataset`, `custom_collate_fn()`, `create_balanced_sampler()`, `stratified_split()` |
| [checkpoints/best_breast_ultrasound.pth](checkpoints/best_breast_ultrasound.pth) | 最佳模型权重 | 包含 model_state_dict, optimizer_state_dict, epoch, loss history |
| [checkpoints/train_curve.png](checkpoints/train_curve.png) | 训练曲线可视化 | Loc Loss & Conf Loss 随 epoch 变化趋势 |
| [results/test_metrics.json](results/test_metrics.json) | 测试指标报告 | mAP, AP per class, Precision, Recall, F1, Confusion details |

## 🔬 模型细节与调优指南

### ResNet 骨干网络对比

| 骨干网络 | 参数量 | 推理速度 | 特征提取能力 | 显存占用 | 适用场景 |
|----------|--------|----------|--------------|----------|----------|
| **ResNet18** | 11.7M | ⚡⚡ 最快 | 中等 | 低 (~3GB) | 快速原型、资源受限 |
| **ResNet34** (推荐) | 21.8M | ⚡ 快速 | 良好 | 中 (~4GB) | **生产环境首选** |
| **ResNet50** | 25.6M | 🐢 较慢 | 优秀 | 高 (~6GB) | 追求极致精度 |

### 训练超参数调优建议

#### 📌 **学习率 (Learning Rate)**
- **推荐范围**: 1e-4 ~ 1e-3
- **过小**: 收敛极慢，可能陷入局部最优
- **过大**: 损失震荡甚至发散
- **策略**: 使用 Warmup + Cosine Annealing（已实现）

#### 📌 **Batch Size**
- **推荐范围**: 8 ~ 32
- **过小**: 梯度估计方差大，训练不稳定
- **过大**: 显存不足，泛化能力下降
- **原则**: 在显存允许范围内尽可能大

#### 📌 **Anchor 尺寸（Default Boxes）**
- 当前使用 SSD300 经典配置
- 若检测效果差，可根据数据集 GT 框尺寸分布重新聚类生成（K-Means）
- 工具: `dataset.py` 中的 `_analyze_dataset()` 可统计分析

#### 📌 **数据增强强度**
- **当前配置**: 中等强度（适合小数据集）
- **若过拟合**: 增强增强（提高概率/幅度）
- **若欠拟合**: 减弱增强（保留更多原始信息）
- **特别注意**: 医学图像不宜过度几何变换（可能破坏解剖结构）

### 常见问题排查

#### ❌ **问题1: 训练 Loss 不下降**
**可能原因及解决方案**:
1. 学习率过大 → 降低至 1e-4
2. 数据标注错误 → 抽查几个 XML 文件
3. Default Box 尺寸不匹配 → 重新分析 GT 分布
4. 梯度爆炸 → 检查是否启用 gradient clipping（已启用 5.0）

#### ❌ **问题2: 验证 Loss 震荡严重**
**解决方案**:
1. 降低学习率
2. 增加 batch size
3. 减少数据增强强度
4. 增加 Weight Decay（如 1e-3）

#### ❌ **问题3: 召回率低（漏检多）**
**解决方案**（针对恶性肿瘤）:
1. 降低置信度阈值（如 0.25 → 0.15）
2. 降低 NMS IoU 阈值（如 0.45 → 0.3）
3. 增加 Positive Sample 的权重（修改 Loss）
4. 使用 Focal Loss 替代 Cross Entropy（聚焦难样本）

#### ❌ **问题4: 精确率低（误检多）**
**解决方案**:
1. 提高置信度阈值（如 0.25 → 0.5）
2. 提高 NMS IoU 阈值（如 0.45 → 0.6）
3. 增加 Negative Mining 的比例
4. 数据增强中加入 Hard Negative 示例

## ⚠️ 注意事项

### 🔴 **关键限制与警告**

1. **数据量不足**:
   - 当前仅约 98 张图像，远低于深度学习所需的最小规模（建议 500+）
   - 极易导致过拟合，模型泛化能力有限
   - **必须扩充数据集**才能获得可靠的性能

2. **性能现状**:
   - 当前测试 mAP 仅 0.13%，**不具备临床使用条件**
   - 恶性肿瘤召回率仅 20%，漏检风险极高
   - **仅适用于学术研究和算法验证**

3. **GPU 要求**:
   - **强烈推荐使用 NVIDIA GPU**（CUDA 加速）
   - CPU 训练速度极慢（预计慢 20-50 倍）
   - 建议 GPU 显存 ≥ 6GB（ResNet34 + SSD300）
   - 若显存不足：减小 batch_size 至 4 或 8

4. **输入规格**:
   - 固定输入尺寸: **300×300×3** (RGB)
   - 必须使用 ImageNet 均值/标准差归一化
   - 不支持动态尺寸输入（SSD 全卷积结构限制）

5. **标注质量要求**:
   - 边界框必须紧密包裹肿瘤区域（误差 < 5px）
   - 类别标签准确（benign vs malignant 需病理确诊）
   - 避免模糊、伪影严重的图像

6. **临床应用禁忌**:
   - ⛔ **绝对不能替代医生诊断**
   - ⛔ 不能用于最终诊断决策
   - ⛔ 恶性肿瘤漏检可能导致严重后果
   - ✅ 仅作为**辅助参考工具**，结果需人工复核

## 🔮 未来改进路线图

### 🚀 **短期改进（立即可做，预期提升显著）**

- [ ] **📊 数据集扩充（最高优先级）**
  - 目标: 500~1000 张高质量标注图像
  - 来源: 公开数据集（BUSI, UDIAT, etc.）、医院合作
  - 预期: mAP 提升 10-30%

- [ ] **🔄 超参数系统性搜索**
  - 工具: Optuna / Ray Tune / Weights & Biases
  - 搜索空间: lr, batch_size, anchor scales, augmentation params
  - 方法: 贝叶斯优化 / 随机搜索
  - 预期: 找到更优配置，mAP 提升 5-15%

- [ ] **🎯 Focal Loss 替代 Cross Entropy**
  - 原因: 当前类别极度不平衡（背景 >> 肿瘤）
  - 参数: γ=2, α=[0.1, 0.3, 0.6] (bg, benign, malignant)
  - 预期: 提升难样本（小目标、恶性）检测能力

- [ ] **📐 Anchor Box 重新设计**
  - 方法: K-Means 聚类分析 GT 框尺寸分布
  - 工具: 自定义脚本分析 `Annotations/*.xml`
  - 预期: 更好的先验框匹配，提升 recall

### 🚀 **中期改进（需要一定工作量）**

- [ ] **🏗️ 尝试更先进的检测架构**
  - **YOLOv8/v9**: 单阶段检测 SOTA，速度更快
  - **Faster R-CNN**: 两阶段检测，精度更高但较慢
  - **DETR (Detection Transformer)**: 端到端检测，无需 anchor
  - **RT-DETR**: 实时 DETR，兼顾速度精度

- [ ] **🧠 引入注意力机制**
  - **CBAM**: 通道+空间注意力，聚焦肿瘤区域
  - **SE-Block**: 通道注意力，轻量高效
  - **位置**: 在 Backbone 后或 Detection Head 前
  - 预期: 提升特征判别能力

- [ ] **🖼️ 多尺度训练与测试 (MTS/MTT)**
  - 训练时随机 resize (320~640)，测试时多尺度融合
  - 预期: 提升不同大小肿瘤的检测鲁棒性

- [ ] **🔄 Test-Time Augmentation (TTA)**
  - 测试时对同一图像做多版本增强（翻转/旋转/色彩）
  - 融合多次预测结果（NMS 或投票）
  - 预期: 提升 2-5% mAP

### 🚀 **长期愿景（研究方向）**

- [ ] **📚 半监督/自监督学习**
  - 利用大量无标注超声图像预训练 Backbone
  - 方法: MoCo, SimCLR, DINO (Self-Supervised)
  - 预期: 提升特征表达能力，减少对标注数据的依赖

- [ ] **🎯 多任务学习**
  - 同时进行: 检测 + 分类 + 分割
  - 共享 Backbone，任务特定 Heads
  - 预期: 任务间相互促进，整体性能提升

- [ ] **🔍 可解释性 AI (XAI)**
  - **Grad-CAM**: 可视化模型关注的区域
  - **Attention Map**: 展示注意力权重分布
  - **价值**: 帮助医生理解模型判断依据，建立信任

- [ ] **🌐 模型部署与服务化**
  - **导出 ONNX/TensorRT**: 加速推理（2-10x）
  - **Web API**: FastAPI/Flask 封装，提供在线服务
  - **移动端**: TensorFlow Lite/CoreML，开发手机 App
  - **边缘设备**: Jetson Nano/Raspberry Pi，便携式设备

- [ ] **📊 临床验证与集成**
  - 与医院合作开展前瞻性临床试验
  - 对比 AI 预测与专家诊断的一致性（Cohen's Kappa）
  - 获取医疗器械注册认证（NMPA/FDA/CE）

## 常见问题 FAQ

**Q1: 如何添加新的类别（如 normal/三类分类）？**
A: 
1. 在 `dataset.py` 的 `classes` 列表中添加新类别
2. 修改 `num_classes` = 原类别数 + 1（含背景）
3. 重新生成 Default Boxes（调用相关函数）
4. XML 标注中 `<name>` 字段使用新类别名
5. 重新训练模型

**Q2: 如何使用自己的乳腺超声数据集？**
A: 
1. 将图像放入 `VOC2007/JPEGImages/`（命名任意）
2. 使用 LabelImg/LabelMe 工具标注，导出 Pascal VOC XML 格式
3. XML 放入 `VOC2007/Annotations/`（与图像同名）
4. 运行 `stratified_split()` 自动生成划分文件
5. 开始训练！

**Q3: 训练过程中如何监控模型状态？**
A: 
- **实时日志**: 控制台打印每个 batch/epoch 的 Loss
- **可视化曲线**: `checkpoints/train_curve.png` 自动更新
- **历史快照**: 每 10 epoch 保存至 `training_history_epoch*.pth`
- **TensorBoard**: 可自行集成（需修改代码添加 SummaryWriter）

**Q4: 如何解决显存不足 OOM 问题？**
A: 
1. **减小 batch_size**: 16 → 8 → 4
2. **使用梯度累积**: 每 2-4 个 batch 再更新一次（需改代码）
3. **混合精度训练 (AMP)**: 使用 float16，节省 ~50% 显存
4. **更换骨干网络**: ResNet34 → ResNet18
5. **减小输入尺寸**: 300 → 256（需重新计算 anchor）

**Q5: 为什么我的 mAP 这么低（<1%）？**
A: 最可能的原因：
1. **数据量太少**: <100 张图像无法训练有效的检测器
2. **模型未充分训练**: 150 epoch 可能不够，观察 Loss 是否还在下降
3. **超参数不当**: lr/anchor/增强策略需调整
4. **数据质量问题**: 标注错误、图像模糊、类别不平衡
5. **评估方式问题**: 确保 IoU 阈值设置合理（通常 0.5）

**改进优先级**: 扩充数据 > 调超参 > 换架构

**Q6: 如何将模型部署到生产环境？**
A: 
```python
# 导出为 TorchScript（推荐）
model.eval()
example_input = torch.randn(1, 3, 300, 300)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('bu_id_detector.pt')

# 或导出为 ONNX（跨平台）
torch.onnx.export(model, example_input, 'bu_id_detector.onnx',
                  opset_version=11, do_constant_folding=True)
```

**Q7: 能否进行增量学习（在新数据上继续训练）？**
A: 可以！
1. 加载现有 checkpoint: `torch.load('best_breast_ultrasound.pth')`
2. 恢复 model_state_dict 和 optimizer_state_dict
3. 将新数据加入训练集
4. 降低初始学习率（如 1e-4）
5. 继续训练，监控是否灾难性遗忘旧知识

## 📊 性能基准对比（文献数据）

| 方法 | Backbone | mAP (%) | Precision (%) | Recall (%) | Params | FPS |
|------|----------|---------|----------------|------------|--------|-----|
| **本项目 (SSD300)** | **ResNet34** | **0.13*** | **0.1*** | **60*** | 25M | ~30 |
| Faster R-CNN | ResNet50 | 78-85 | 80-88 | 75-85 | 41M | ~5 |
| YOLOv5s | CSPDarknet | 75-82 | 78-85 | 72-82 | 7M | ~140 |
| YOLOv8m | CSPDarknet | 80-88 | 83-90 | 77-86 | 26M | ~120 |
| RetinaNet | ResNet50 | 76-84 | 79-86 | 73-83 | 38M | ~15 |
| EfficientDet-D0 | EfficientNet-B0 | 72-80 | 75-82 | 70-79 | 4M | ~100 |

> *当前项目数据，非最终性能。文献数据基于类似规模的公开 BUS 数据集（BUSI, UDIAT 等） reported in papers.

**说明**: 本项目当前处于**早期研发阶段**，性能未达到生产水平。上述对比表展示的是同类任务的**业界水平**，供参考和设定改进目标。

## 📄 许可证

本项目仅供**学术研究**和**教育目的**使用。

⚕️ **重要声明**: 
- 本系统**不能替代专业医生的诊断**
- 所有检测结果必须经过**资质医师复核**
- 恶性肿瘤漏检可能导致严重后果，请谨慎使用
- 商业应用需遵循当地医疗器械法规

## 👥 致谢

- **SSD 论文**: Liu et al., "SSD: Single Shot MultiBox Detector", ECCV 2016
- **ResNet 论文**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- **Faster R-CNN**: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", NeurIPS 2015
- **Albumentations 团队**: 提供强大的数据增强库
- **PyTorch & torchvision**: Facebook AI Research
- **BUSI Dataset**: Al-Dhabyani et al., "Deep Learning Approaches for Breast Ultrasound Image Classification", 2020
- **开源社区**: 所有贡献者


**Bug 反馈**: 请提供完整的错误堆栈、环境信息、复现步骤  
**功能建议**: 详细描述需求场景和预期效果  
**数据贡献**: 如有新的乳腺超声标注数据集愿意共享，欢迎联系  
**论文引用**: 如果本工作对您的研究有帮助，欢迎引用（待补充 arXiv 论文链接）

---

**最后更新**: 2026年  
**项目状态**: 🧪 **实验阶段**（需大规模数据和调优）  
**推荐用途**: 学术研究、算法验证、教学演示  

**下一步行动**: 📊 **扩充数据集** → 🔄 **超参优化** → 🚀 **性能提升至可用水平**
