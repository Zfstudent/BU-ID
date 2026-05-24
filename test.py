import os
import json
import torch
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from dataset import (
    BreastUltrasoundDataset,
    custom_collate_fn
)
# 导入你训练代码中的类（必须保持一致）
from train import (
    SSD300,
    device,
    SSDHead
)

# ======================== 配置 ========================
VOC_ROOT = "/home/BU-ID/VOC2007"
CHECKPOINT_PATH = "/home/BU-ID/checkpoints/best_breast_ultrasound.pth"
SAVE_VIS_DIR = "/home/BU-ID/results"
os.makedirs(SAVE_VIS_DIR, exist_ok=True)

IMG_SIZE = 300
NUM_CLASSES = 3
CLASSES = ["background", "benign", "malignant"]
CONF_THRESH = 0.25  # 置信度阈值
NMS_THRESH = 0.45   # NMS阈值

# ======================== 后处理：NMS + 解码 ========================
DEFAULT_BOXES = None

def generate_default_boxes():
    """
    生成 SSD300 (ResNet34 backbone) 的 Default Boxes
    根据训练代码的实际配置：
    - 输入：300x300
    - 特征图：[38, 19, 10, 5, 3, 2]
    - num_defaults: [4, 6, 6, 6, 4, 4]
    - 总计：8744 个
    """
    default_boxes = []
    feature_maps = [38, 19, 10, 5, 3, 2]
    min_sizes = [30, 60, 111, 162, 213, 264]
    max_sizes = [60, 111, 162, 213, 264, 315]
    
    for fm_idx, (fm_size, min_s, max_s) in enumerate(zip(feature_maps, min_sizes, max_sizes)):
        step = IMG_SIZE / fm_size
        
        for i in range(fm_size):
            for j in range(fm_size):
                cx = (j + 0.5) * step / IMG_SIZE
                cy = (i + 0.5) * step / IMG_SIZE
                
                # 第 1 个框
                default_boxes.append([cx, cy, min_s/IMG_SIZE, min_s/IMG_SIZE])
                
                # 第 2 个框
                s = sqrt(min_s * max_s) / IMG_SIZE
                default_boxes.append([cx, cy, s, s])
                
                # 第 3-6 个框
                if fm_idx in [1, 2, 3]:
                    for ar in [2, 0.5, 3, 1/3]:
                        w = (min_s / IMG_SIZE) * sqrt(ar)
                        h = (min_s / IMG_SIZE) / sqrt(ar)
                        default_boxes.append([cx, cy, w, h])
                else:
                    for ar in [2, 0.5]:
                        w = (min_s / IMG_SIZE) * sqrt(ar)
                        h = (min_s / IMG_SIZE) / sqrt(ar)
                        default_boxes.append([cx, cy, w, h])
    
    return torch.tensor(default_boxes, dtype=torch.float32)

def get_default_boxes(num_expected):
    """获取 default boxes，如果数量不匹配则打印警告并尝试重新生成"""
    global DEFAULT_BOXES
    
    if DEFAULT_BOXES is None:
        DEFAULT_BOXES = generate_default_boxes()
        
        # 如果数量不匹配，打印警告
        if len(DEFAULT_BOXES) != num_expected:
            print(f"\n{'='*60}")
            print(f"⚠️  WARNING: Default boxes 数量不匹配!")
            print(f"   测试代码生成：{len(DEFAULT_BOXES)} 个")
            print(f"   模型实际输出：{num_expected} 个")
            print(f"{'='*60}\n")
            print(f"这将导致维度错误。正在尝试使用模型的 num_defaults 配置...\n")
            
            # 尝试根据模型输出反推 num_defaults
            # 特征图尺寸：[38, 19, 10, 5, 3, 2]
            feature_maps = [38, 19, 10, 5, 3, 2]
            base_boxes = sum([fm * fm * 4 for fm in feature_maps])  # 每层至少 4 个
            
            # 计算需要额外添加多少个 box
            extra_needed = num_expected - base_boxes
            print(f"基础配置 (每层 4 个): {base_boxes}")
            print(f"需要额外的 boxes: {extra_needed}")
            
            # 尝试分配额外的 boxes 到各层
            # 通常第 2-4 层 (19,10,5) 会有 6 个框 (额外 2 个)
            extra_per_layer = [0, 2, 2, 2, 0, 0]  # 默认配置
            extra_total = sum([fm * fm * e for fm, e in zip(feature_maps, extra_per_layer)])
            
            print(f"默认额外配置：{extra_total}")
            print(f"差值：{extra_needed - extra_total}")
            
            if extra_needed != extra_total:
                print(f"\n❌ 无法匹配模型配置，请检查训练代码中的 SSDHead.num_defaults")
                print(f"   当前训练代码配置：num_defaults = [4,6,6,6,4,4]")
                print(f"   模型实际输出需要：需要反推")
            
            print(f"\n{'='*60}\n")
    
    return DEFAULT_BOXES

def decode_boxes(default_boxes, loc_pred):
    """
    SSD 框解码
    
    注意：根据模型训练方式选择解码方式：
    - 新模型（使用标准 SSD 训练）：使用编码后的偏移量解码
    - 旧模型（直接预测坐标）：直接转换为 (x1,y1,x2,y2)
    """
    # 判断模型类型：检查 loc_pred 的范围
    # 如果是偏移量，值通常较大（有正负）
    # 如果是直接坐标，值应该在 [0,1] 之间
    
    if loc_pred.abs().max() > 2:
        # 旧模型：直接预测坐标 (cx, cy, w, h)
        # 将 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)
        x1y1 = loc_pred[..., :2] - loc_pred[..., 2:] / 2
        x2y2 = loc_pred[..., :2] + loc_pred[..., 2:] / 2
        boxes = torch.cat([x1y1, x2y2], dim=-1)
        return boxes.clamp(0, 1)
    else:
        # 新模型：预测偏移量，使用标准 SSD 解码
        cxcy = default_boxes[..., :2] + loc_pred[..., :2] * 0.1 * default_boxes[..., 2:]
        wh = default_boxes[..., 2:] * torch.exp(loc_pred[..., 2:] * 0.2)
        boxes = torch.cat([cxcy - wh/2, cxcy + wh/2], dim=-1)
        return boxes.clamp(0, 1)

def post_process(loc_pred, conf_pred):
    """后处理：置信度过滤 + NMS"""
    num_boxes = loc_pred.shape[1]
    default_boxes = get_default_boxes(num_boxes).to(loc_pred.device)
    scores_all = torch.softmax(conf_pred, dim=-1)
    batch_boxes, batch_labels, batch_scores = [], [], []

    for i in range(loc_pred.size(0)):
        # 使用标准 SSD 解码
        boxes = decode_boxes(default_boxes, loc_pred[i])
        scores = scores_all[i]
        max_scores, labels = torch.max(scores[:, 1:], dim=-1)
        labels += 1

        keep = max_scores > CONF_THRESH
        if keep.sum() == 0:
            batch_boxes.append(torch.empty(0,4))
            batch_labels.append(torch.empty(0,dtype=torch.long))
            batch_scores.append(torch.empty(0))
            continue

        boxes = boxes[keep]
        scores = max_scores[keep]
        labels = labels[keep]

        # NMS
        keep_idx = torch.ops.torchvision.nms(boxes, scores, NMS_THRESH)
        batch_boxes.append(boxes[keep_idx])
        batch_labels.append(labels[keep_idx])
        batch_scores.append(scores[keep_idx])

    return batch_boxes, batch_labels, batch_scores

# ======================== 指标计算：mAP、Precision、Recall、F1 ========================
class DetectionMetrics:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.gt_boxes = defaultdict(list)
        self.pred_boxes = defaultdict(list)
        self.image_ids = []

    def update(self, image_id, gt_box, gt_label, pred_box, pred_label, pred_score):
        if image_id not in self.image_ids:
            self.image_ids.append(image_id)
        self.gt_boxes[image_id] = (gt_box, gt_label)
        self.pred_boxes[image_id] = (pred_box, pred_label, pred_score)

    def calculate_ap(self, recall, precision):
        """计算AP（VOC2007 11点插值）"""
        ap = 0.0
        for t in np.linspace(0,1,11):
            mask = recall >= t
            prec = precision[mask].max() if mask.any() else 0
            ap += prec / 11
        return ap

    def compute(self):
        """计算所有指标"""
        results = {}
        total_gt_counts = defaultdict(int)

        for c in range(1, self.num_classes):
            all_scores = []
            all_tp = []

            for img_id in self.image_ids:
                gt_b, gt_l = self.gt_boxes[img_id]
                pred_b, pred_l, pred_s = self.pred_boxes[img_id]
                
                gt_mask = gt_l == c
                gt_b_c = gt_b[gt_mask]
                gt_used = np.zeros(len(gt_b_c))
                total_gt_counts[c] += len(gt_b_c)

                pred_mask = pred_l == c
                pred_b_c = pred_b[pred_mask]
                pred_s_c = pred_s[pred_mask]

                for pb, ps in zip(pred_b_c, pred_s_c):
                    all_scores.append(ps)
                    if len(gt_b_c) == 0:
                        all_tp.append(0)
                        continue

                    ious = self.iou(pb.unsqueeze(0), gt_b_c)
                    if ious.dim() == 0:
                        max_iou = ious.item()
                        best_gt_idx = 0
                    elif ious.dim() == 1:
                        max_iou, best_gt_idx = ious.max(0)
                        max_iou = max_iou.item()
                        best_gt_idx = best_gt_idx.item()
                    else:
                        max_iou, best_gt_idx = ious.max(1)
                        max_iou = max_iou.item()
                        best_gt_idx = best_gt_idx.item()

                    if max_iou > 0.5 and gt_used[best_gt_idx] == 0:
                        all_tp.append(1)
                        gt_used[best_gt_idx] = 1
                    else:
                        all_tp.append(0)

            tp_array = np.array(all_tp)
            tp = tp_array.sum()
            fp = len(all_scores) - tp
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (total_gt_counts[c] + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0

            ap = 0
            if len(all_scores) > 0:
                scores = np.array(all_scores)
                tp_cumsum = np.cumsum(tp_array)
                fp_cumsum = np.cumsum(1 - tp_array)
                recalls = tp_cumsum / (total_gt_counts[c] + 1e-8)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
                ap = self.calculate_ap(recalls, precisions)

            results[c] = {
                "class": CLASSES[c],
                "AP": ap,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "gt_count": total_gt_counts[c],
                "tp": int(tp),
                "fp": int(fp)
            }

        mAP = np.mean([results[c]["AP"] for c in results])
        results["mAP"] = mAP
        return results

    @staticmethod
    def iou(box1, box2):
        """计算IoU"""
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        iou = inter_area / (area1 + area2 - inter_area + 1e-8)
        return iou

# ======================== 可视化预测结果 ========================
def visualize_result(image, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, save_path):
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    ax = plt.gca()

    # 画真实框（绿色）
    for box, lab in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle((x1*IMG_SIZE, y1*IMG_SIZE), w*IMG_SIZE, h*IMG_SIZE,
                                 linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1*IMG_SIZE, y1*IMG_SIZE-5, CLASSES[lab], color='green', fontsize=12)

    # 画预测框（红色）
    for box, lab, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle((x1*IMG_SIZE, y1*IMG_SIZE), w*IMG_SIZE, h*IMG_SIZE,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1*IMG_SIZE, y1*IMG_SIZE-5, f"{CLASSES[lab]} {score:.2f}",
                color='red', fontsize=12)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ======================== 主测试函数 ========================
def test():
    # 1. 加载数据集
    val_dataset = BreastUltrasoundDataset(VOC_ROOT, split="test", img_size=IMG_SIZE, augment=False)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=0
    )

    # 2. 加载模型
    model = SSD300(num_classes=NUM_CLASSES, backbone_type='resnet34').to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ 加载最优模型，epoch={checkpoint['epoch']+1}")

    # 3. 初始化指标计算器
    metrics = DetectionMetrics(num_classes=NUM_CLASSES)

    # 4. 批量测试
    print("\n🔍 开始测试...")
    with torch.no_grad():
        for idx, (img, _, _) in enumerate(tqdm(val_loader, desc="Testing")):
            img = img.to(device)
            loc_pred, conf_pred = model(img)

            # 后处理
            pred_boxes, pred_labels, pred_scores = post_process(loc_pred, conf_pred)

            # 预测结果
            pb = pred_boxes[0].cpu()
            pl = pred_labels[0].cpu()
            ps = pred_scores[0].cpu()
            
            # 获取真实框（从数据集直接读取）
            img_id = val_dataset.image_ids[idx]
            gt_boxes, gt_labels = val_dataset.get_ground_truth(img_id)
            
            # 更新指标
            metrics.update(img_id, gt_boxes, gt_labels, pb, pl, ps)

            # 保存可视化（每 10 张保存一张）
            if idx % 10 == 0:
                img_np = img[0].cpu().permute(1,2,0).numpy()
                img_np = (img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])) * 255
                img_np = img_np.astype(np.uint8)
                save_path = os.path.join(SAVE_VIS_DIR, f"{img_id}.jpg")
                visualize_result(img_np, gt_boxes, gt_labels, pb, pl, ps, save_path)

    # 5. 计算并输出指标
    results = metrics.compute()
    print("\n" + "="*60)
    print("📊 乳腺超声肿瘤检测 测试指标报告")
    print("="*60)
    print(f"📌 mAP = {results['mAP']:.4f}")
    print("-"*60)

    for c in range(1, NUM_CLASSES):
        res = results[c]
        print(f"🏷️  {res['class']:10s} | AP={res['AP']:.4f} | P={res['Precision']:.4f} | R={res['Recall']:.4f} | F1={res['F1']:.4f}")
        print(f"         真实框={res['gt_count']} | 正确={res['tp']} | 误检={res['fp']}")
        print("-"*60)

    # 保存报告
    report_path = os.path.join(SAVE_VIS_DIR, "test_metrics.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n📂 指标报告已保存：{report_path}")
    print(f"📂 可视化结果已保存：{SAVE_VIS_DIR}")
    print("🎉 测试完成！")

if __name__ == "__main__":
    test()