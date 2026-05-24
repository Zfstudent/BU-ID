import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision
from math import sqrt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from collections import Counter
import random

# ======================== Default Boxes 生成（SSD 核心）========================
DEFAULT_BOXES = None

def generate_default_boxes():
    """
    生成 SSD300 的 Default Boxes（先验框）
    输入：300x300
    特征图：[38, 19, 10, 5, 3, 2]
    num_defaults: [4, 6, 6, 6, 4, 4]
    总计：8744 个
    """
    default_boxes = []
    img_size = 300
    feature_maps = [38, 19, 10, 5, 3, 2]
    min_sizes = [30, 60, 111, 162, 213, 264]
    max_sizes = [60, 111, 162, 213, 264, 315]
    
    for fm_idx, (fm_size, min_s, max_s) in enumerate(zip(feature_maps, min_sizes, max_sizes)):
        step = img_size / fm_size
        
        for i in range(fm_size):
            for j in range(fm_size):
                cx = (j + 0.5) * step / img_size
                cy = (i + 0.5) * step / img_size
                
                default_boxes.append([cx, cy, min_s/img_size, min_s/img_size])
                
                s = sqrt(min_s * max_s) / img_size
                default_boxes.append([cx, cy, s, s])
                
                if fm_idx in [1, 2, 3]:
                    for ar in [2, 0.5, 3, 1/3]:
                        w = (min_s / img_size) * sqrt(ar)
                        h = (min_s / img_size) / sqrt(ar)
                        default_boxes.append([cx, cy, w, h])
                else:
                    for ar in [2, 0.5]:
                        w = (min_s / img_size) * sqrt(ar)
                        h = (min_s / img_size) / sqrt(ar)
                        default_boxes.append([cx, cy, w, h])
    
    return torch.tensor(default_boxes, dtype=torch.float32)

def get_default_boxes():
    """获取 default boxes（单例模式）"""
    global DEFAULT_BOXES
    if DEFAULT_BOXES is None:
        DEFAULT_BOXES = generate_default_boxes()
    return DEFAULT_BOXES

def encode_default_boxes(default_boxes, gt_boxes, gt_labels, threshold=0.5):
    """
    将真实框编码到 default boxes（SSD 匹配策略）
    
    Args:
        default_boxes: [N, 4], 格式 (cx, cy, w, h)
        gt_boxes: [M, 4], 格式 (cx, cy, w, h)
        gt_labels: [M]
        threshold: IoU 阈值
    
    Returns:
        loc_target: [N, 4], 编码后的偏移量
        conf_target: [N], 类别标签（0 为背景）
    """
    N = default_boxes.shape[0]
    M = gt_boxes.shape[0]
    
    if M == 0:
        loc_target = torch.zeros(N, 4)
        conf_target = torch.zeros(N, dtype=torch.long)
        return loc_target, conf_target
    
    def compute_iou(boxes1, boxes2):
        boxes1 = boxes1.unsqueeze(1)
        boxes2 = boxes2.unsqueeze(0)
        
        b1_x1y1 = boxes1[..., :2] - boxes1[..., 2:] / 2
        b1_x2y2 = boxes1[..., :2] + boxes1[..., 2:] / 2
        b2_x1y1 = boxes2[..., :2] - boxes2[..., 2:] / 2
        b2_x2y2 = boxes2[..., :2] + boxes2[..., 2:] / 2
        
        inter_x1 = torch.max(b1_x1y1[..., 0], b2_x1y1[..., 0])
        inter_y1 = torch.max(b1_x1y1[..., 1], b2_x1y1[..., 1])
        inter_x2 = torch.min(b1_x2y2[..., 0], b2_x2y2[..., 0])
        inter_y2 = torch.min(b1_x2y2[..., 1], b2_x2y2[..., 1])
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        area1 = (b1_x2y2[..., 0] - b1_x1y1[..., 0]) * (b1_x2y2[..., 1] - b1_x1y1[..., 1])
        area2 = (b2_x2y2[..., 0] - b2_x1y1[..., 0]) * (b2_x2y2[..., 1] - b2_x1y1[..., 1])
        
        union = area1 + area2 - inter_area + 1e-8
        iou = inter_area / union
        
        return iou
    
    ious = compute_iou(default_boxes, gt_boxes)
    
    best_gt_for_default, _ = ious.max(dim=1)
    _, best_default_for_gt = ious.max(dim=0)
    
    loc_target = torch.zeros(N, 4)
    conf_target = torch.zeros(N, dtype=torch.long)
    
    for i in range(N):
        if best_gt_for_default[i] < threshold:
            loc_target[i] = torch.zeros(4)
            conf_target[i] = 0
        else:
            matched_gt_idx = ious[i].argmax()
            matched_gt = gt_boxes[matched_gt_idx]
            
            d_cx, d_cy, d_w, d_h = default_boxes[i]
            g_cx, g_cy, g_w, g_h = matched_gt
            
            loc_target[i, 0] = (g_cx - d_cx) / d_w
            loc_target[i, 1] = (g_cy - d_cy) / d_h
            loc_target[i, 2] = torch.log(g_w / d_w)
            loc_target[i, 3] = torch.log(g_h / d_h)
            
            conf_target[i] = gt_labels[matched_gt_idx]
    
    for j in range(M):
        best_default_idx = ious[:, j].argmax()
        matched_gt = gt_boxes[j]
        d_cx, d_cy, d_w, d_h = default_boxes[best_default_idx]
        g_cx, g_cy, g_w, g_h = matched_gt
        
        loc_target[best_default_idx, 0] = (g_cx - d_cx) / d_w
        loc_target[best_default_idx, 1] = (g_cy - d_cy) / d_h
        loc_target[best_default_idx, 2] = torch.log(g_w / d_w)
        loc_target[best_default_idx, 3] = torch.log(g_h / d_h)
        conf_target[best_default_idx] = gt_labels[j]
    
    return loc_target, conf_target


class BreastUltrasoundDataset(Dataset):
    def __init__(self, voc_root: str, split: str = "train", img_size: int = 300,
                 augment: bool = True):
        self.voc_root = voc_root
        self.img_size = img_size
        self.augment = augment

        self.img_dir = os.path.join(voc_root, "JPEGImages")
        self.xml_dir = os.path.join(voc_root, "Annotations")
        self.split_file = os.path.join(voc_root, "ImageSets", "Main", f"{split}.txt")

        with open(self.split_file, "r", encoding="utf-8") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        self.classes = ["benign", "malignant"]
        self.class_to_idx = {cls: idx+1 for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx+1: cls for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes) + 1

        self.dataset_stats = self._analyze_dataset()

        if augment:
            self.transform = self._build_training_transform()
        else:
            self.transform = self._build_validation_transform()

    def _analyze_dataset(self):
        stats = {"total_images": len(self.image_ids), "class_counts": Counter()}
        for img_id in self.image_ids:
            xml_path = os.path.join(self.xml_dir, f"{img_id}.xml")
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    cls_name = obj.find("name").text.strip().lower()
                    if cls_name in self.class_to_idx:
                        stats["class_counts"][cls_name] += 1
            except:
                continue
        return stats

    def _build_training_transform(self):
        return A.Compose([
            A.Resize(height=self.img_size, width=self.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.3),
            A.OneOf([A.GaussNoise(var_limit=(5,15)), A.ISONoise(intensity=(0.05,0.15))], p=0.2),
            A.OneOf([A.GaussianBlur(blur_limit=(3,5)), A.MotionBlur(blur_limit=(3,5))], p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift=10, p=0.2),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

    def _build_validation_transform(self):
        return A.Compose([
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        try:
            img = Image.open(os.path.join(self.img_dir, f"{img_id}.jpg")).convert("RGB")
        except:
            img = Image.open(os.path.join(self.img_dir, f"{img_id}.png")).convert("RGB")
        img = np.array(img)
        h,w = img.shape[:2]
        
        tree = ET.parse(os.path.join(self.xml_dir, f"{img_id}.xml"))
        objects = tree.getroot().findall("object")
        
        bboxes, labels = [], []
        for obj in objects:
            cls = obj.find("name").text.strip().lower()
            if cls not in self.class_to_idx: continue
            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text)/w
            ymin = float(bb.find("ymin").text)/h
            xmax = float(bb.find("xmax").text)/w
            ymax = float(bb.find("ymax").text)/h
            if xmax<=xmin or ymax<=ymin: continue
            cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
            bw, bh = xmax-xmin, ymax-ymin
            bboxes.append([cx,cy,bw,bh])
            labels.append(self.class_to_idx[cls])
        
        if not bboxes:
            bboxes.append([0.5,0.5,0.05,0.05])
            labels.append(1)
        
        aug = self.transform(image=img, bboxes=bboxes, class_labels=labels)
        img = aug["image"]
        
        gt_boxes = torch.tensor(aug["bboxes"], dtype=torch.float32)
        gt_labels = torch.tensor(aug["class_labels"], dtype=torch.long)
        
        return img, gt_boxes, gt_labels
    
    def get_ground_truth(self, img_id):
        xml_path = os.path.join(self.xml_dir, f"{img_id}.xml")
        tree = ET.parse(xml_path)
        objects = tree.getroot().findall("object")
        
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img_path = os.path.join(self.img_dir, f"{img_id}.png")
            img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        gt_boxes = []
        gt_labels = []
        
        for obj in objects:
            cls = obj.find("name").text.strip().lower()
            if cls not in self.class_to_idx: continue
            
            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text)
            ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text)
            ymax = float(bb.find("ymax").text)
            
            if xmax <= xmin or ymax <= ymin:
                continue
            
            x1, y1 = xmin / w, ymin / h
            x2, y2 = xmax / w, ymax / h
            
            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append(self.class_to_idx[cls])
        
        if len(gt_boxes) == 0:
            gt_boxes.append([0.5, 0.5, 0.5, 0.5])
            gt_labels.append(1)
        
        return torch.tensor(gt_boxes, dtype=torch.float32), torch.tensor(gt_labels, dtype=torch.long)

    def __len__(self): 
        return len(self.image_ids)


def custom_collate_fn(batch):
    """
    自定义 collate 函数：处理 batch 并进行 SSD 匹配编码
    """
    imgs = [item[0] for item in batch]
    gt_boxes_list = [item[1] for item in batch]
    gt_labels_list = [item[2] for item in batch]
    
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    
    padded_imgs = []
    for img in imgs:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h > 0 or pad_w > 0:
            padded_img = torch.zeros(c, max_h, max_w, dtype=img.dtype)
            padded_img[:, :h, :w] = img
            padded_imgs.append(padded_img)
        else:
            padded_imgs.append(img)
    
    default_boxes = get_default_boxes()
    loc_targets = []
    conf_targets = []
    
    for gt_boxes, gt_labels in zip(gt_boxes_list, gt_labels_list):
        loc_target, conf_target = encode_default_boxes(default_boxes, gt_boxes, gt_labels)
        loc_targets.append(loc_target)
        conf_targets.append(conf_target)
    
    imgs_tensor = torch.stack(padded_imgs, dim=0)
    loc_targets_tensor = torch.stack(loc_targets, dim=0)
    conf_targets_tensor = torch.stack(conf_targets, dim=0)
    
    return imgs_tensor, loc_targets_tensor, conf_targets_tensor


def create_balanced_sampler(dataset):
    cls_list = []
    for idx in range(len(dataset)):
        _, gt_boxes, gt_labels = dataset[idx]
        if len(gt_labels) > 0:
            cls_list.append(int(gt_labels[0]))
        else:
            cls_list.append(1)
    count = Counter(cls_list)
    weights = [len(cls_list)/(len(count)*count[c]) for c in cls_list]
    return WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)


def stratified_split(voc_root, train_ratio=0.8, seed=42):
    xml_dir = os.path.join(voc_root, "Annotations")
    out_dir = os.path.join(voc_root, "ImageSets/Main")
    os.makedirs(out_dir, exist_ok=True)
    
    data = []
    for f in os.listdir(xml_dir):
        if not f.endswith(".xml"): continue
        tid = f[:-4]
        try:
            root = ET.parse(os.path.join(xml_dir,f)).getroot()
            clss = set(o.find("name").text.lower() for o in root.findall("object"))
            data.append( (tid, "malignant" if "malignant" in clss else "benign") )
        except: continue
    
    random.seed(seed)
    train, val = [], []
    for k in ["benign","malignant"]:
        group = [t for t,c in data if c==k]
        random.shuffle(group)
        s = int(len(group)*train_ratio)
        train += group[:s]
        val += group[s:]
    
    with open(os.path.join(out_dir,"train.txt"),"w") as f: f.write("\n".join(train))
    with open(os.path.join(out_dir,"val.txt"),"w") as f: f.write("\n".join(val))
    print(f"✅ 划分完成：训练={len(train)}, 验证={len(val)}")
    return train, val
