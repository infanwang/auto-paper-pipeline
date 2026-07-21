#!/usr/bin/env python3
"""改进3: DPNeXt Knowledge Distillation"""
import torch, torch.nn as nn, torch.optim as optim
import json
from pathlib import Path

class TeacherNet(nn.Module):
    def __init__(self, n_cls=10):
        super().__init__()
        self.feat = nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.cls = nn.Linear(64, n_cls)
    def forward(self, x):
        f = self.feat(x).flatten(1)
        return self.cls(f), f

class StudentNet(nn.Module):
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

def main():
    print("="*60)
    print("改进3: DPNeXt Knowledge Distillation")
    print("="*60)
    
    teacher, student = TeacherNet(), StudentNet()
    tp = sum(p.numel() for p in teacher.parameters())
    sp = sum(p.numel() for p in student.parameters())
    print(f"\nTeacher: {tp:,} params | Student: {sp:,} params | 压缩: {tp/sp:.1f}x")
    
    opt = optim.Adam(student.parameters(), lr=1e-3)
    crit = DistillLoss(T=4.0, alpha=0.7)
    
    print("\n训练:")
    for e in range(50):
        x = torch.randn(32,3,32,32)
        lbl = torch.randint(0,10,(32,))
        with torch.no_grad(): t_out,_ = teacher(x)
        s_out, _ = student(x)
        loss = crit(s_out, t_out, lbl)
        opt.zero_grad(); loss.backward(); opt.step()
        if (e+1)%10==0:
            acc = (s_out.argmax(1)==lbl).float().mean()
            print(f"  Epoch {e+1:2d} | Loss {loss.item():.3f} | Acc {acc:.0%}")
    
    results = {'teacher': tp, 'student': sp, 'compression': tp/sp}
    Path('/root/git/mimo/paper-pipeline/reproduction/improvements/results_dpnext_distill.json').write_text(json.dumps(results, indent=2))
    print(f"\n完成: {tp/sp:.1f}x压缩比")

if __name__ == "__main__":
    main()
