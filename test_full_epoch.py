#!/usr/bin/env python3
"""Test a full epoch of training to find where it's slow"""
import torch
import torch.optim as optim
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from models.semi_siamese import Semi_siamese_
from models.losses import DiceLoss, FocalLoss, softmax_helper
from datasets.nuclei_dataset import NucleiDataset
from torch.utils.data import DataLoader
from utils.metrics import MetricTracker
import numpy as np

print("Setup...", flush=True)
model = Semi_siamese_(in_channels=3, out_channels=3, init_features=32)
device = torch.device('cuda')
model.to(device)
model.train()

dataset = NucleiDataset(
    high_dir='/root/Honors_Thesis/Patches/MoNuSeg_train_high',
    low_dir='/root/Honors_Thesis/Patches/MoNuSeg_train_low',
    mask_dir='/root/Honors_Thesis/Patches/MoNuSeg_train_mask'
)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Calculate class weights
print("Calculating class weights...", flush=True)
class_counts = torch.zeros(2, device=device)
for batch in loader:
    mask = batch['L'].to(device)
    for c in range(2):
        class_counts[c] += (mask == c).sum()
alpha = class_counts.cpu().numpy()
print(f"Class weights: {alpha}", flush=True)

focal_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)
dice_loss = DiceLoss(n_classes=2)

def criterion(pred, mask):
    return focal_loss(pred, mask) + dice_loss(pred, mask)

metric_tracker = MetricTracker(n_classes=2)

print(f"\nProcessing {len(loader)} batches...", flush=True)
epoch_start = time.time()
running_loss = 0.0

for batch_idx, batch in enumerate(loader):
    batch_start = time.time()

    img_high = batch['A'].to(device)
    img_low = batch['B'].to(device)
    mask = batch['L'].to(device)

    optimizer.zero_grad()
    pred = model(img_high, img_low)
    loss = criterion(pred, mask)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    # Convert to CPU numpy before metrics
    pred_classes = torch.argmax(pred.detach(), dim=1)
    metric_tracker.update(pred_classes.cpu().numpy(), mask.detach().cpu().numpy())

    batch_time = time.time() - batch_start

    if (batch_idx + 1) % 20 == 0:
        print(f"  Batch {batch_idx+1}/{len(loader)}: loss={loss.item():.4f}, time={batch_time:.2f}s", flush=True)

epoch_time = time.time() - epoch_start
scores = metric_tracker.get_scores()

print(f"\n✓ EPOCH COMPLETE in {epoch_time:.1f}s", flush=True)
print(f"  Loss: {running_loss/len(loader):.4f}", flush=True)
print(f"  F1: {scores['f1_per_class']}", flush=True)
print(f"  Avg batch time: {epoch_time/len(loader):.2f}s", flush=True)