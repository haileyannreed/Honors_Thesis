#!/usr/bin/env python3
"""
Ray Tune specific diagnostic to identify why parallel workers are failing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ray
from ray import tune
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets.nuclei_dataset import NucleiDataset
from models.semi_siamese import Semi_siamese_
from models.losses import DiceLoss, FocalLoss, softmax_helper

class Config:
    BASE_DIR = Path('/root/Honors_Thesis')
    PATCHES_DIR = BASE_DIR / 'Patches'
    DATASET = 'MoNuSeg'
    HIGH_TRAIN = PATCHES_DIR / f'{DATASET}_train_high'
    LOW_TRAIN = PATCHES_DIR / f'{DATASET}_train_low'
    MASK_TRAIN = PATCHES_DIR / f'{DATASET}_train_mask'
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    INIT_FEATURES = 32
    N_CLASSES = 2

print("=" * 80)
print("RAY TUNE PARALLEL EXECUTION DIAGNOSTIC")
print("=" * 80)
print()

# TEST 1: Simple Ray Tune with dummy function
print("TEST 1: Basic Ray Tune parallel execution")
print("-" * 80)

def dummy_trainable(config):
    """Minimal trainable to test Ray Tune parallel execution"""
    print(f"Worker {config['id']} started")
    time.sleep(1)
    return {"metric": config['id']}

try:
    ray.init(ignore_reinit_error=True)

    analysis = tune.run(
        dummy_trainable,
        config={"id": tune.grid_search([0, 1, 2, 3])},
        resources_per_trial={"cpu": 1, "gpu": 0.25},
        verbose=0
    )

    print(f"✓ PASSED: {len(analysis.trials)} trials completed")
    print()
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 2: Ray Tune with CUDA initialization
print("TEST 2: Ray Tune with CUDA initialization")
print("-" * 80)

def cuda_trainable(config):
    """Test CUDA initialization in Ray worker"""
    worker_id = config['id']
    print(f"Worker {worker_id}: Initializing CUDA...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Worker {worker_id}: Device = {device}")

    # Try to allocate GPU memory
    dummy_tensor = torch.randn(100, 100).to(device)
    result = dummy_tensor.sum().item()

    print(f"Worker {worker_id}: CUDA allocation successful")
    return {"metric": result}

try:
    analysis = tune.run(
        cuda_trainable,
        config={"id": tune.grid_search([0, 1, 2, 3])},
        resources_per_trial={"cpu": 1, "gpu": 0.25},
        verbose=0
    )

    print(f"✓ PASSED: {len(analysis.trials)} trials completed")
    print()
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 3: Ray Tune with Dataset loading
print("TEST 3: Ray Tune with Dataset loading")
print("-" * 80)

def dataset_trainable(config):
    """Test dataset loading in Ray worker"""
    worker_id = config['id']
    print(f"Worker {worker_id}: Loading dataset...")

    dataset = NucleiDataset(
        high_dir=Config.HIGH_TRAIN,
        low_dir=Config.LOW_TRAIN,
        mask_dir=Config.MASK_TRAIN
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print(f"Worker {worker_id}: Dataset loaded, batch shape = {batch['A'].shape}")
    return {"metric": len(dataset)}

try:
    analysis = tune.run(
        dataset_trainable,
        config={"id": tune.grid_search([0, 1, 2, 3])},
        resources_per_trial={"cpu": 1, "gpu": 0.25},
        verbose=0
    )

    print(f"✓ PASSED: {len(analysis.trials)} trials completed")
    print()
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 4: Ray Tune with Model initialization
print("TEST 4: Ray Tune with Model initialization")
print("-" * 80)

def model_trainable(config):
    """Test model creation and GPU transfer in Ray worker"""
    worker_id = config['id']
    print(f"Worker {worker_id}: Creating model...")

    model = Semi_siamese_(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        init_features=Config.INIT_FEATURES
    )

    print(f"Worker {worker_id}: Moving model to GPU...")
    device = torch.device('cuda')
    model = model.to(device)

    # Test forward pass
    dummy_high = torch.randn(2, 3, 128, 128).to(device)
    dummy_low = torch.randn(2, 3, 128, 128).to(device)

    with torch.no_grad():
        output = model(dummy_high, dummy_low)

    print(f"Worker {worker_id}: Model forward pass successful, output shape = {output.shape}")
    return {"metric": output.sum().item()}

try:
    analysis = tune.run(
        model_trainable,
        config={"id": tune.grid_search([0, 1, 2, 3])},
        resources_per_trial={"cpu": 1, "gpu": 0.25},
        verbose=0
    )

    print(f"✓ PASSED: {len(analysis.trials)} trials completed")
    print()
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 5: Ray Tune with class weight calculation
print("TEST 5: Ray Tune with class weight calculation")
print("-" * 80)

def class_weight_trainable(config):
    """Test class weight calculation in Ray worker"""
    worker_id = config['id']
    print(f"Worker {worker_id}: Loading dataset...")

    dataset = NucleiDataset(
        high_dir=Config.HIGH_TRAIN,
        low_dir=Config.LOW_TRAIN,
        mask_dir=Config.MASK_TRAIN
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Worker {worker_id}: Calculating class weights on CPU...")
    class_counts = torch.zeros(Config.N_CLASSES)
    for i, batch in enumerate(loader):
        mask = batch['L']
        for c in range(Config.N_CLASSES):
            class_counts[c] += (mask == c).sum()
        if i >= 10:  # Only process 10 batches for speed
            break

    alpha = class_counts.numpy()
    print(f"Worker {worker_id}: Class weights calculated = {alpha}")
    return {"metric": alpha[0]}

try:
    analysis = tune.run(
        class_weight_trainable,
        config={"id": tune.grid_search([0, 1, 2, 3])},
        resources_per_trial={"cpu": 1, "gpu": 0.25},
        verbose=0
    )

    print(f"✓ PASSED: {len(analysis.trials)} trials completed")
    print()
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 6: Ray Tune with loss function creation
print("TEST 6: Ray Tune with loss function creation")
print("-" * 80)

def loss_trainable(config):
    """Test loss function creation in Ray worker"""
    worker_id = config['id']
    print(f"Worker {worker_id}: Creating loss functions...")

    alpha = [28437164., 10360151.]

    focal_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)
    dice_loss = DiceLoss(n_classes=Config.N_CLASSES)

    device = torch.device('cuda')
    dummy_pred = torch.randn(2, 2, 128, 128).to(device)
    dummy_mask = torch.randint(0, 2, (2, 128, 128)).to(device)

    loss = focal_loss(dummy_pred, dummy_mask) + dice_loss(dummy_pred, dummy_mask)

    print(f"Worker {worker_id}: Loss calculated = {loss.item():.4f}")
    return {"metric": loss.item()}

try:
    analysis = tune.run(
        loss_trainable,
        config={"id": tune.grid_search([0, 1, 2, 3])},
        resources_per_trial={"cpu": 1, "gpu": 0.25},
        verbose=0
    )

    print(f"✓ PASSED: {len(analysis.trials)} trials completed")
    print()
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 7: Ray Tune with single training step
print("TEST 7: Ray Tune with single training step")
print("-" * 80)

def training_step_trainable(config):
    """Test a single training step in Ray worker"""
    worker_id = config['id']
    print(f"Worker {worker_id}: Setting up training...")

    # Create model
    model = Semi_siamese_(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        init_features=Config.INIT_FEATURES
    )
    device = torch.device('cuda')
    model = model.to(device)

    # Load data
    dataset = NucleiDataset(
        high_dir=Config.HIGH_TRAIN,
        low_dir=Config.LOW_TRAIN,
        mask_dir=Config.MASK_TRAIN
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    # Create loss and optimizer
    alpha = [28437164., 10360151.]
    focal_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)
    dice_loss = DiceLoss(n_classes=Config.N_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"Worker {worker_id}: Running training step...")
    model.train()
    batch = next(iter(loader))
    img_high = batch['A'].to(device)
    img_low = batch['B'].to(device)
    mask = batch['L'].to(device)

    optimizer.zero_grad()
    pred = model(img_high, img_low)
    loss = focal_loss(pred, mask) + dice_loss(pred, mask)
    loss.backward()
    optimizer.step()

    print(f"Worker {worker_id}: Training step completed, loss = {loss.item():.4f}")
    return {"metric": loss.item()}

try:
    analysis = tune.run(
        training_step_trainable,
        config={"id": tune.grid_search([0, 1, 2, 3])},
        resources_per_trial={"cpu": 1, "gpu": 0.25},
        verbose=0
    )

    print(f"✓ PASSED: {len(analysis.trials)} trials completed")
    print()
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 8: Ray Tune with tune.report
print("TEST 8: Ray Tune with tune.report")
print("-" * 80)

def tune_report_trainable(config):
    """Test tune.report in Ray worker"""
    worker_id = config['id']
    print(f"Worker {worker_id}: Testing tune.report...")

    # Simulate training for 3 epochs
    for epoch in range(3):
        time.sleep(0.5)
        tune.report(epoch=epoch, loss=1.0/(epoch+1), worker_id=worker_id)

    print(f"Worker {worker_id}: Completed 3 epochs")

try:
    analysis = tune.run(
        tune_report_trainable,
        config={"id": tune.grid_search([0, 1, 2, 3])},
        resources_per_trial={"cpu": 1, "gpu": 0.25},
        verbose=0
    )

    print(f"✓ PASSED: {len(analysis.trials)} trials completed")
    print()
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

ray.shutdown()

print("=" * 80)
print("ALL RAY TUNE TESTS PASSED!")
print("=" * 80)
print()
print("This means Ray Tune parallel execution works fine with your setup.")
print("The issue must be in your actual training loop code.")
print()
print("Possible issues:")
print("1. Checkpointing code (Checkpoint.from_directory)")
print("2. Metrics reporting format")
print("3. Training loop hangs on specific batch/operation")
print("4. Memory leak causing OOM after loading data")