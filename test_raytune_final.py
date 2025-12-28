#!/usr/bin/env python3
"""
Final comprehensive Ray Tune test with extensive debugging
Tests the EXACT setup used in hyperparam_tuning_raytune.py
"""

import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent))

from datasets.nuclei_dataset import NucleiDataset
from models.semi_siamese import Semi_siamese_
from models.losses import DiceLoss, FocalLoss, softmax_helper
from utils.metrics import MetricTracker

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

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

print("=" * 80)
print("COMPREHENSIVE RAY TUNE TEST WITH DEBUG LOGGING")
print("=" * 80)
print()

def train_epoch_debug(model, dataloader, optimizer, criterion, device, trial_id):
    """Train for one epoch with debug output"""
    print(f"[Trial {trial_id}] Starting train_epoch...")
    model.train()
    running_loss = 0.0
    metric_tracker = MetricTracker(n_classes=Config.N_CLASSES)

    batch_count = len(dataloader)
    print(f"[Trial {trial_id}] Training on {batch_count} batches")

    for batch_idx, batch in enumerate(dataloader):
        img_high = batch['A'].to(device)
        img_low = batch['B'].to(device)
        mask = batch['L'].to(device)

        optimizer.zero_grad()
        pred = model(img_high, img_low)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        metric_tracker.update(pred, mask)

        # Print progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"[Trial {trial_id}] Train batch {batch_idx+1}/{batch_count}, loss={loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    scores = metric_tracker.get_scores()
    print(f"[Trial {trial_id}] Train epoch complete: loss={epoch_loss:.4f}")

    return epoch_loss, scores

def validate_debug(model, dataloader, criterion, device, trial_id):
    """Validate with debug output"""
    print(f"[Trial {trial_id}] Starting validation...")
    model.eval()
    running_loss = 0.0
    metric_tracker = MetricTracker(n_classes=Config.N_CLASSES)

    batch_count = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            img_high = batch['A'].to(device)
            img_low = batch['B'].to(device)
            mask = batch['L'].to(device)

            pred = model(img_high, img_low)
            loss = criterion(pred, mask)

            running_loss += loss.item()
            metric_tracker.update(pred, mask)

    epoch_loss = running_loss / len(dataloader)
    scores = metric_tracker.get_scores()
    print(f"[Trial {trial_id}] Validation complete: loss={epoch_loss:.4f}")

    return epoch_loss, scores

def test_trainable(config):
    """Test trainable matching hyperparam_tuning_raytune.py exactly"""
    trial_id = config['trial_id']

    print(f"[Trial {trial_id}] ========================================")
    print(f"[Trial {trial_id}] STARTING TRIAL")
    print(f"[Trial {trial_id}] Config: lr={config['lr']:.6f}, wd={config['wd']:.6f}")
    print(f"[Trial {trial_id}] ========================================")

    # Set random seeds
    print(f"[Trial {trial_id}] Setting random seeds...")
    seed = 8888
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    gen = torch.Generator()
    gen.manual_seed(seed)

    # Create model
    print(f"[Trial {trial_id}] Creating model...")
    model = Semi_siamese_(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        init_features=Config.INIT_FEATURES
    )

    # Setup device
    print(f"[Trial {trial_id}] Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"[Trial {trial_id}] Model on {device}")

    # Create datasets
    print(f"[Trial {trial_id}] Loading datasets...")
    train_dataset = NucleiDataset(
        high_dir=Config.HIGH_TRAIN,
        low_dir=Config.LOW_TRAIN,
        mask_dir=Config.MASK_TRAIN
    )
    print(f"[Trial {trial_id}] Train dataset: {len(train_dataset)} samples")

    test_dataset = NucleiDataset(
        high_dir=Config.HIGH_TEST,
        low_dir=Config.LOW_TEST,
        mask_dir=Config.MASK_TEST
    )
    print(f"[Trial {trial_id}] Test dataset: {len(test_dataset)} samples")

    # Create dataloaders
    print(f"[Trial {trial_id}] Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=gen
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    print(f"[Trial {trial_id}] DataLoaders created: {len(train_loader)} train batches, {len(test_loader)} val batches")

    # Setup loss
    print(f"[Trial {trial_id}] Setting up loss functions...")
    alpha = config['alpha']
    print(f"[Trial {trial_id}] Alpha values: {alpha}")

    focal_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)
    dice_loss = DiceLoss(n_classes=Config.N_CLASSES)

    def criterion(pred, mask):
        return focal_loss(pred, mask) + dice_loss(pred, mask)

    print(f"[Trial {trial_id}] Loss functions created")

    # Setup optimizer
    print(f"[Trial {trial_id}] Creating optimizer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.999),
        weight_decay=config['wd']
    )
    print(f"[Trial {trial_id}] Optimizer created")

    # Test a few epochs
    print(f"[Trial {trial_id}] Starting training loop (3 epochs)...")
    for epoch in range(3):
        print(f"[Trial {trial_id}] ===== EPOCH {epoch+1}/3 =====")
        epoch_start = time.time()

        # Train
        train_loss, train_scores = train_epoch_debug(model, train_loader, optimizer, criterion, device, trial_id)

        # Validate
        val_loss, val_scores = validate_debug(model, test_loader, criterion, device, trial_id)

        # Calculate metrics
        f1_per_class = val_scores['f1_per_class']
        f1_mean = np.mean(f1_per_class[1:]) if Config.N_CLASSES > 1 else f1_per_class[0]

        metrics = {
            "loss": val_loss,
            "f1": f1_mean,
            "f1_nuclei": f1_per_class[1] if Config.N_CLASSES > 1 else 0.0,
            "precision": val_scores['precision_per_class'][1] if Config.N_CLASSES > 1 else 0.0,
            "recall": val_scores['recall_per_class'][1] if Config.N_CLASSES > 1 else 0.0,
            "accuracy": val_scores['accuracy'],
        }

        epoch_time = time.time() - epoch_start
        print(f"[Trial {trial_id}] Epoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f"[Trial {trial_id}] Metrics: loss={val_loss:.4f}, f1={f1_mean:.4f}")

        # Report to Ray Tune
        print(f"[Trial {trial_id}] Calling tune.report()...")
        try:
            tune.report(**metrics)
            print(f"[Trial {trial_id}] tune.report() SUCCESS")
        except Exception as e:
            print(f"[Trial {trial_id}] tune.report() FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise

    print(f"[Trial {trial_id}] ========================================")
    print(f"[Trial {trial_id}] TRIAL COMPLETE")
    print(f"[Trial {trial_id}] ========================================")

# Initialize Ray
print("Initializing Ray...")
os.environ["RAY_TMPDIR"] = "/tmp/ray_tmp"
ray.init(_temp_dir="/tmp/ray_tmp", ignore_reinit_error=True)
print("Ray initialized")

# Pre-calculate class weights
print("\nPre-calculating class weights...")
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
print(f"Class weights calculated: {alpha_list}")

# Define test configuration
print("\nDefining test configuration...")
config = {
    "lr": tune.grid_search([3e-4, 5e-4]),  # 2 values
    "wd": tune.grid_search([1e-6, 1e-5]),  # 2 values
    "batch_size": 16,
    "alpha": alpha_list,
    "trial_id": tune.grid_search([0, 1, 2, 3]),  # 4 trials total
}

# ASHA scheduler
scheduler = ASHAScheduler(
    metric="f1",
    mode='max',
    max_t=3,
    grace_period=2,
    reduction_factor=2,
)

print("\nStarting Ray Tune test (4 trials, 3 epochs each)...")
print("This will test PARALLEL execution with GPUS_PER_TRIAL=0.25")
print()

try:
    tuner = tune.Tuner(
        tune.with_resources(
            test_trainable,
            resources={"cpu": 1, "gpu": 0.25}  # 4 parallel trials
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=1,  # 4 trials from grid search
        ),
        param_space=config,
        run_config=tune.RunConfig(
            storage_path="/tmp/ray_tune_test",
            name="final_test",
            verbose=2,
        ),
    )

    result = tuner.fit()

    print("\n" + "=" * 80)
    print("✓ RAY TUNE TEST PASSED!")
    print("=" * 80)
    print(f"Completed {len(result)} trials")

    for i, trial_result in enumerate(result):
        print(f"\nTrial {i}:")
        print(f"  Config: {trial_result.config}")
        print(f"  Final F1: {trial_result.metrics.get('f1', 'N/A'):.4f}")
        print(f"  Final Loss: {trial_result.metrics.get('loss', 'N/A'):.4f}")

except Exception as e:
    print("\n" + "=" * 80)
    print("✗ RAY TUNE TEST FAILED!")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

ray.shutdown()

print("\n" + "=" * 80)
print("ALL TESTS PASSED - RAY TUNE WORKS WITH YOUR EXACT SETUP!")
print("=" * 80)
print()
print("If this test passes, your hyperparam_tuning_raytune.py should work.")
print("If it fails, you'll see exactly where it fails in the debug output.")