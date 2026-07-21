#!/usr/bin/env python3
"""DPNeXt Knowledge Distillation"""
import torch, torch.nn as nn, torch.optim as optim
import json
from pathlib import Path

class Teacher(nn.Module):
    def __init__(self, n_cls=10):
        super().__init__()
        self.feat = nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.cls = nn.Linear(128, n_cls)
    def forward(self, x):
        f = self.feat(x).flatten(1)
        return self.cls(f), f

class Student(nn.Module):
    def __init__(self, n_cls=10):
        super().__init__()
        self.feat = nn.Sequential(nn.Conv2d(3,8,1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1,groups=8),nn.ReLU(),nn.Conv2d(8,16,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.cls = nn.Linear(16, n_cls)
    def forward(self, x):
        f = self.feat(x).flatten(1)
        return self.cls(f), f

class DistillLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.7):
        super().__init__()
        self.T, self.alpha = T, alpha
    def forward(self, s_out, t_out, labels):
        s_log = nn.functional.log_softmax(s_out/self.T, dim=1)
        t_soft = nn.functional.softmax(t_out/self.T, dim=1)
        distill = nn.KLDivLoss(reduction='batchmean')(s_log, t_soft) * self.T**2
        hard = nn.functional.cross_entropy(s_out, labels)
        return self.alpha * distill + (1-self.alpha) * hard

class FeatDistill(nn.Module):
    def __init__(self, s_dim, t_dim):
        super().__init__()
        self.proj = nn.Linear(s_dim, t_dim)
    def forward(self, s_f, t_f):
        return nn.functional.mse_loss(self.proj(s_f), t_f)

def main():
    print("="*60)
    print("DPNeXt Knowledge Distillation")
    print("="*60)
    
    teacher, student = Teacher(), Student()
    tp = sum(p.numel() for p in teacher.parameters())
    sp = sum(p.numel() for p in student.parameters())
    print(f"\nTeacher: {tp:,} params ({tp*4/1024:.1f} KB)")
    print(f"Student: {sp:,} params ({sp*4/1024:.1f} KB)")
    print(f"Compression: {tp/sp:.1f}x")
    
    fd = FeatDistill(16, 128)
    opt = optim.Adam(list(student.parameters()) + list(fd.parameters()), lr=1e-3)
    kd = DistillLoss(T=4.0, alpha=0.7)
    
    print(f"\nTraining (50 epochs):")
    for e in range(50):
        x = torch.randn(32,3,32,32)
        lbl = torch.randint(0,10,(32,))
        with torch.no_grad(): t_out, t_feat = teacher(x)
        s_out, s_feat = student(x)
        loss = kd(s_out, t_out, lbl) + 0.3 * fd(s_feat, t_feat)
        opt.zero_grad(); loss.backward(); opt.step()
        if (e+1)%10==0:
            acc = (s_out.argmax(1)==lbl).float().mean()
            print(f"  Epoch {e+1:2d} | Loss {loss.item():.3f} | Acc {acc:.0%}")
    
    with torch.no_grad():
        x = torch.randn(100,3,32,32)
        lbl = torch.randint(0,10,(100,))
        t_acc = (teacher(x)[0].argmax(1)==lbl).float().mean()
        s_acc = (student(x)[0].argmax(1)==lbl).float().mean()
    
    print(f"\nTeacher: {t_acc:.1%} | Student: {s_acc:.1%} | Compression: {tp/sp:.1f}x")
    Path('/root/git/mimo/paper-pipeline/reproduction/dpnext/results.json').write_text(json.dumps({'teacher':tp,'student':sp,'compression':tp/sp,'t_acc':float(t_acc),'s_acc':float(s_acc)},indent=2))
    print("Done!")

if __name__ == "__main__":
    main()
