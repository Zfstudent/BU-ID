import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import (
    BreastUltrasoundDataset, 
    custom_collate_fn, 
    create_balanced_sampler,
    stratified_split
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  使用设备：{device}")


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//16, 1),
            nn.ReLU(),
            nn.Conv2d(channels//16, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)


class ResNetBackbone(nn.Module):
    def __init__(self, backbone_type='resnet34'):
        super().__init__()
        import torchvision
        if backbone_type == 'resnet18':
            self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone_type == 'resnet34':
            self.resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

        self.layer1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool, self.resnet.layer1)
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

    def forward(self,x):
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class SSDHead(nn.Module):
    def __init__(self, num_classes=3, in_channels=[128, 256, 512, 256, 256, 256]):
        super().__init__()
        self.num_classes = num_classes
        self.num_defaults = [4,6,6,6,4,4]

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(ch) for ch in in_channels
        ])

        self.loc = nn.ModuleList([
            nn.Conv2d(in_channels[0], self.num_defaults[0]*4, 3,padding=1),
            nn.Conv2d(in_channels[1],self.num_defaults[1]*4,3,padding=1),
            nn.Conv2d(in_channels[2],self.num_defaults[2]*4,3,padding=1),
            nn.Conv2d(in_channels[3], self.num_defaults[3]*4,3,padding=1),
            nn.Conv2d(in_channels[4], self.num_defaults[4]*4,3,padding=1),
            nn.Conv2d(in_channels[5], self.num_defaults[5]*4,3,padding=1),
        ])
        self.conf = nn.ModuleList([
            nn.Conv2d(in_channels[0], self.num_defaults[0]*num_classes,3,padding=1),
            nn.Conv2d(in_channels[1],self.num_defaults[1]*num_classes,3,padding=1),
            nn.Conv2d(in_channels[2],self.num_defaults[2]*num_classes,3,padding=1),
            nn.Conv2d(in_channels[3], self.num_defaults[3]*num_classes,3,padding=1),
            nn.Conv2d(in_channels[4], self.num_defaults[4]*num_classes,3,padding=1),
            nn.Conv2d(in_channels[5], self.num_defaults[5]*num_classes,3,padding=1),
        ])

    def forward(self, features):
        loc, conf = [], []
        for i,f in enumerate(features):
            f_attended = self.attention_blocks[i](f)
            loc.append(self.loc[i](f_attended).permute(0,2,3,1).contiguous())
            conf.append(self.conf[i](f_attended).permute(0,2,3,1).contiguous())
        loc = torch.cat([o.view(o.size(0),-1) for o in loc], dim=1)
        conf = torch.cat([o.view(o.size(0),-1) for o in conf], dim=1)
        return loc.view(loc.size(0),-1,4), conf.view(conf.size(0),-1,self.num_classes)


class SSD300(nn.Module):
    def __init__(self, num_classes=3, backbone_type='resnet34'):
        super().__init__()
        self.backbone = ResNetBackbone(backbone_type)

        if backbone_type in ['resnet18', 'resnet34']:
            last_channel = 512
            extra_channels = [512, 256, 256]
            head_in_channels = [128, 256, 512, 512, 256, 256]
        else:
            last_channel = 2048
            extra_channels = [512, 256, 256]
            head_in_channels = [512, 1024, 2048, 512, 256, 256]

        self.extras = nn.ModuleList([
            nn.Conv2d(last_channel,extra_channels[0],1), nn.ReLU(), nn.Conv2d(extra_channels[0],extra_channels[0],3,2,1), nn.ReLU(),
            nn.Conv2d(extra_channels[0],extra_channels[1],1), nn.ReLU(), nn.Conv2d(extra_channels[1],extra_channels[1],3,2,1), nn.ReLU(),
            nn.Conv2d(extra_channels[1],extra_channels[2],1), nn.ReLU(), nn.Conv2d(extra_channels[2],extra_channels[2],3,2,1), nn.ReLU(),
        ])
        self.head = SSDHead(num_classes, head_in_channels)

    def forward(self,x):
        c3,c4,c5 = self.backbone(x)
        f = c5
        extras = []
        for i,l in enumerate(self.extras):
            f = l(f)
            if i%4==3: extras.append(f)
        feats = [c3,c4,c5] + extras
        return self.head(feats)


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes=3, neg_pos=3):
        super().__init__()
        self.num_classes = num_classes
        self.neg_pos = neg_pos
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    def forward(self, loc_pred, conf_pred, loc_target, conf_target):
        B, N, _ = loc_pred.shape
        
        pos = conf_target > 0
        pos_3d = pos.unsqueeze(-1).expand_as(loc_pred)
        
        if pos.sum() == 0:
            loc_loss = torch.tensor(0., device=loc_pred.device)
        else:
            num_pos = pos.sum().clamp(min=1)
            loc_loss = F.smooth_l1_loss(
                loc_pred[pos_3d],
                loc_target[pos_3d],
                reduction="sum"
            ) / num_pos
        
        conf_loss_per_box = self.ce_loss(
            conf_pred.view(-1, self.num_classes),
            conf_target.view(-1)
        ).view(B, N)
        
        neg_loss = conf_loss_per_box.clone()
        neg_loss[pos] = -float('inf')
        
        _, idx = neg_loss.sort(dim=1, descending=True)
        neg = torch.zeros_like(pos, dtype=torch.bool)
        
        for i in range(B):
            n_pos = pos[i].sum().item()
            n_neg = min(self.neg_pos * n_pos, N - n_pos)
            neg[i, idx[i, :n_neg]] = True
        
        mask = pos | neg
        
        if mask.sum() == 0:
            conf_loss = torch.tensor(0., device=loc_pred.device)
        else:
            conf_loss = conf_loss_per_box[mask].mean()

        return loc_loss, conf_loss


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    loc_loss_sum, conf_loss_sum, count = 0,0,0
    pbar = tqdm(loader, total=len(loader))
    for img, loc_t, conf_t in pbar:
        img = img.to(device)
        loc_t = loc_t.to(device)
        conf_t = conf_t.to(device)
        
        optimizer.zero_grad()
        loc_p, conf_p = model(img)
        ll, cl = criterion(loc_p, conf_p, loc_t, conf_t)
        loss = ll + cl
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        loc_loss_sum += ll.item()
        conf_loss_sum += cl.item()
        count +=1
        pbar.set_description(f"loss={ll+cl:.4f} loc={ll:.4f} conf={cl:.4f}")
    return loc_loss_sum/count, conf_loss_sum/count


def val_epoch(model, loader, criterion, device):
    model.eval()
    ll, cl, cnt = 0,0,0
    with torch.no_grad():
        for img, loc_t, conf_t in loader:
            img = img.to(device)
            loc_t = loc_t.to(device)
            conf_t = conf_t.to(device)
            
            lp, cp = model(img)
            l, c = criterion(lp, cp, loc_t, conf_t)
            ll += l.item()
            cl += c.item()
            cnt +=1
    return ll/cnt, cl/cnt


if __name__ == "__main__":
    voc_root = "/home/BU-ID/VOC2007"
    img_size = 300
    batch_size = 16
    epochs = 150
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    num_classes = 3
    backbone_type = 'resnet34'
    
    checkpoint_dir = "/home/BU-ID/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    stratified_split(voc_root, 0.8)

    train_ds = BreastUltrasoundDataset(voc_root, "train", img_size, augment=True)
    val_ds = BreastUltrasoundDataset(voc_root, "val", img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             collate_fn=custom_collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           collate_fn=custom_collate_fn, num_workers=0, pin_memory=False)

    model = SSD300(num_classes, backbone_type).to(device)
    criterion = MultiBoxLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-7
    )

    best_val = 9999
    patience = 20
    trigger = 0
    train_ll, train_cl = [], []
    val_ll, val_cl = [], []
    history = {'train_loc': [], 'train_conf': [], 'val_loc': [], 'val_conf': [], 'lr': []}

    print("\n🚀 开始训练...")
    print(f"📊 数据集大小：训练集={len(train_ds)}, 验证集={len(val_ds)}")
    print(f"⚙️  配置：batch_size={batch_size}, lr={lr}, momentum={momentum}, weight_decay={weight_decay}")
    print(f"️  正则化：梯度裁剪=5.0, 增强数据=True ")
    print(f"📈 调度器：CosineAnnealingLR (T_max={epochs}, eta_min=1e-7)")
    print(f"🔄 优化器：SGD + Momentum (移除 WeightedSampler)")
    
    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch+1}/{epochs} ========")
        tr_ll, tr_cl = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=5.0)
        v_ll, v_cl = val_epoch(model, val_loader, criterion, device)

        train_ll.append(tr_ll)
        train_cl.append(tr_cl)
        val_ll.append(v_ll)
        val_cl.append(v_cl)
        
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loc'].append(tr_ll)
        history['train_conf'].append(tr_cl)
        history['val_loc'].append(v_ll)
        history['val_conf'].append(v_cl)
        history['lr'].append(current_lr)

        total_val = v_ll + v_cl
        print(f"[VAL] loc={v_ll:.4f} conf={v_cl:.4f} total={total_val:.4f} (lr={current_lr:.2e})")

        if total_val < best_val:
            best_val = total_val
            best_epoch = epoch
            best_model_path = os.path.join(checkpoint_dir, "best_breast_ultrasound.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': total_val,
                'loc_loss': v_ll,
                'conf_loss': v_cl,
                'history': history,
            }, best_model_path)
            trigger = 0
            print(f" 保存最优模型 (Epoch {epoch+1}, val_loss={total_val:.4f})")
        else:
            trigger +=1
            print(f"⚠️  无提升 (trigger={trigger}/{patience})")
            if trigger >= patience:
                print(f"\n早停！连续{patience}轮无提升")
                print(f" 最佳模型来自 Epoch {best_epoch+1}, val_loss={best_val:.4f}")
                break

        scheduler.step(total_val)
        
        curve_path = os.path.join(checkpoint_dir, "train_curve.png")
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(train_ll, label="train loc")
        plt.plot(val_ll, label="val loc")
        plt.title("Loc Loss")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(train_cl, label="train conf")
        plt.plot(val_cl, label="val conf")
        plt.title("Conf Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(curve_path, dpi=150)
        plt.close()
        
        if (epoch + 1) % 10 == 0:
            history_path = os.path.join(checkpoint_dir, f"training_history_epoch{epoch+1}.pth")
            torch.save(history, history_path)

    print("\n" + "="*50)
    print("🎉 训练完成！")
    print(f"📌 最佳模型：Epoch {best_epoch+1}, val_loss={best_val:.4f}")
    print(f"📂 模型已保存：{os.path.join(checkpoint_dir, 'best_breast_ultrasound.pth')}")
    print("="*50)
