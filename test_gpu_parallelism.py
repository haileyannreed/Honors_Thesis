#!/usr/bin/env python3
"""
Test if GPU can handle multiple PyTorch processes in parallel
Uses multiprocessing (not Ray Tune) to isolate whether the issue is:
1. GPU hardware limitation
2. Ray Tune specific issue
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import multiprocessing as mp
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets.nuclei_dataset import NucleiDataset
from models.semi_siamese import Semi_siamese_
from models.losses import DiceLoss, FocalLoss, softmax_helper
from utils.metrics import MetricTracker

class Config:
    BASE_DIR = Path('/root/Honors_Thesis')
    PATCHES_DIR = BASE_DIR / 'Patches'
    DATASET = 'MoNuSeg'
    HIGH_TRAIN = PATCHES_DIR / f'{DATASET}_train_high'
    LOW_TRAIN = PATCHES_DIR / f'{DATASET}_train_low'
    MASK_TRAIN = PATCHES_DIR / f'{DATASET}_train_mask'
    HIGH_TEST = PATCHES_DIR / f'{DATASET}_test_high'
    LOW_TEST = PATCHES_DIR / f'{DATASET}_test_low'
    MASK_TEST = PATCHES_DIR / f'{DATASET}_test_mask'
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    INIT_FEATURES = 32
    N_CLASSES = 2

def train_one_epoch(worker_id, alpha_list, num_batches=20):
    """Train for a few batches - called by each parallel worker"""
    print(f"[Worker {worker_id}] Starting...")
    start_time = time.time()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    print(f"[Worker {worker_id}] Creating model...")
    model = Semi_siamese_(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        init_features=Config.INIT_FEATURES
    ).to(device)

    # Create dataset
    print(f"[Worker {worker_id}] Loading dataset...")
    dataset = NucleiDataset(
        high_dir=Config.HIGH_TRAIN,
        low_dir=Config.LOW_TRAIN,
        mask_dir=Config.MASK_TRAIN
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # Create loss
    print(f"[Worker {worker_id}] Creating loss...")
    focal_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha_list, gamma=2, smooth=1e-5)
    dice_loss = DiceLoss(n_classes=Config.N_CLASSES)

    def criterion(pred, mask):
        return focal_loss(pred, mask) + dice_loss(pred, mask)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Train for a few batches
    print(f"[Worker {worker_id}] Training {num_batches} batches...")
    model.train()

    batch_times = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        batch_start = time.time()

        img_high = batch['A'].to(device)
        img_low = batch['B'].to(device)
        mask = batch['L'].to(device)

        optimizer.zero_grad()
        pred = model(img_high, img_low)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if (i + 1) % 5 == 0:
            avg_time = sum(batch_times[-5:]) / 5
            print(f"[Worker {worker_id}] Batch {i+1}/{num_batches}, loss={loss.item():.4f}, time={batch_time:.2f}s (avg {avg_time:.2f}s)")

    total_time = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times)

    print(f"[Worker {worker_id}] COMPLETE: {num_batches} batches in {total_time:.1f}s (avg {avg_batch_time:.2f}s per batch)")
    return worker_id, total_time, avg_batch_time

if __name__ == '__main__':
    print("=" * 80)
    print("GPU PARALLELISM TEST")
    print("=" * 80)
    print()

    # Pre-calculate class weights
    print("Pre-calculating class weights...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temp_dataset = NucleiDataset(
        high_dir=Config.HIGH_TRAIN,
        low_dir=Config.LOW_TRAIN,
        mask_dir=Config.MASK_TRAIN
    )
    temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=False, num_workers=0)

    class_counts = torch.zeros(Config.N_CLASSES, device=device)
    for batch in temp_loader:
        mask = batch['L'].to(device)
        for c in range(Config.N_CLASSES):
            class_counts[c] += (mask == c).sum()

    alpha = class_counts.cpu().numpy()
    alpha_list = alpha.tolist()
    print(f"Class weights: {alpha_list}")
    print()

    # TEST 1: Sequential execution (baseline)
    print("=" * 80)
    print("TEST 1: Sequential Execution (1 worker at a time) - BASELINE")
    print("=" * 80)
    print()

    sequential_start = time.time()
    for worker_id in range(2):
        train_one_epoch(worker_id, alpha_list, num_batches=20)
    sequential_total = time.time() - sequential_start

    print()
    print(f"Sequential total time: {sequential_total:.1f}s")
    print()

    # TEST 2: Parallel execution with multiprocessing
    print("=" * 80)
    print("TEST 2: Parallel Execution (2 workers simultaneously)")
    print("=" * 80)
    print()

    parallel_start = time.time()

    # Use spawn method to avoid CUDA initialization issues
    mp.set_start_method('spawn', force=True)

    with mp.Pool(processes=2) as pool:
        results = []
        for worker_id in range(2):
            result = pool.apply_async(train_one_epoch, args=(worker_id, alpha_list, 20))
            results.append(result)

        # Wait for all workers to complete
        outputs = [r.get() for r in results]

    parallel_total = time.time() - parallel_start

    print()
    print(f"Parallel total time: {parallel_total:.1f}s")
    print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    print(f"Sequential execution: {sequential_total:.1f}s")
    print(f"Parallel execution:   {parallel_total:.1f}s")
    print(f"Speedup:              {sequential_total/parallel_total:.2f}x")
    print()

    if parallel_total < sequential_total * 0.7:
        print("✓ GOOD SPEEDUP: GPU handles parallel training well!")
        print("  The issue is likely Ray Tune specific, not GPU limitation.")
    elif parallel_total < sequential_total * 1.2:
        print("⚠ MODERATE SLOWDOWN: GPU can handle parallel training but with overhead")
        print("  Parallel training is feasible but may be slower than expected.")
    else:
        print("✗ SEVERE SLOWDOWN: GPU struggles with parallel training")
        print("  The GPU cannot efficiently handle multiple PyTorch processes.")
        print("  Recommendation: Run trials sequentially (GPUS_PER_TRIAL=1)")

    print()
    print("Expected speedup for perfect parallelism: ~2x")
    print("Expected speedup with GPU overhead: ~1.3-1.5x")
    print("If speedup < 1.0x, GPU is thrashing (should run sequentially)")