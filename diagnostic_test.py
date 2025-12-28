#!/usr/bin/env python3
"""
Diagnostic script to isolate the exact issue with Ray Tune training
Tests each component step by step
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import sys
from pathlib import Path

# Add project to path
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
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    INIT_FEATURES = 32
    N_CLASSES = 2

def test_1_cuda():
    """Test 1: CUDA availability"""
    print("=" * 80)
    print("TEST 1: CUDA Availability")
    print("=" * 80)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("✓ PASSED\n")

def test_2_dataset():
    """Test 2: Dataset loading"""
    print("=" * 80)
    print("TEST 2: Dataset Loading")
    print("=" * 80)
    try:
        dataset = NucleiDataset(
            high_dir=Config.HIGH_TRAIN,
            low_dir=Config.LOW_TRAIN,
            mask_dir=Config.MASK_TRAIN
        )
        print(f"Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Sample shapes: A={sample['A'].shape}, B={sample['B'].shape}, L={sample['L'].shape}")
        print("✓ PASSED\n")
        return dataset
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        raise

def test_3_dataloader(dataset):
    """Test 3: DataLoader"""
    print("=" * 80)
    print("TEST 3: DataLoader")
    print("=" * 80)
    try:
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        print(f"Batches: {len(loader)}")

        print("Loading first batch...")
        start = time.time()
        batch = next(iter(loader))
        elapsed = time.time() - start

        print(f"Batch loaded in {elapsed:.2f}s")
        print(f"Batch shapes: A={batch['A'].shape}, B={batch['B'].shape}, L={batch['L'].shape}")
        print("✓ PASSED\n")
        return loader
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        raise

def test_4_model():
    """Test 4: Model initialization and GPU transfer"""
    print("=" * 80)
    print("TEST 4: Model Initialization")
    print("=" * 80)
    try:
        model = Semi_siamese_(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.OUT_CHANNELS,
            init_features=Config.INIT_FEATURES
        )
        print(f"Model created")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Model moved to {device}")

        # Test forward pass
        dummy_high = torch.randn(2, 3, 128, 128).to(device)
        dummy_low = torch.randn(2, 3, 128, 128).to(device)

        print("Testing forward pass...")
        start = time.time()
        with torch.no_grad():
            output = model(dummy_high, dummy_low)
        elapsed = time.time() - start

        print(f"Forward pass completed in {elapsed:.3f}s")
        print(f"Output shape: {output.shape}")
        print("✓ PASSED\n")
        return model, device
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        raise

def test_5_class_weights(loader, device):
    """Test 5: Class weight calculation"""
    print("=" * 80)
    print("TEST 5: Class Weight Calculation")
    print("=" * 80)
    try:
        print("Calculating class weights...")
        start = time.time()

        class_counts = torch.zeros(Config.N_CLASSES, device=device)
        for batch in loader:
            mask = batch['L'].to(device)
            for c in range(Config.N_CLASSES):
                class_counts[c] += (mask == c).sum()

        elapsed = time.time() - start
        alpha = class_counts.cpu().numpy()

        print(f"Calculation completed in {elapsed:.2f}s")
        print(f"Class weights: {alpha}")
        print("✓ PASSED\n")
        return alpha
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        raise

def test_6_loss_functions(alpha, device):
    """Test 6: Loss function initialization"""
    print("=" * 80)
    print("TEST 6: Loss Functions")
    print("=" * 80)
    try:
        focal_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)
        dice_loss = DiceLoss(n_classes=Config.N_CLASSES)
        print("Loss functions created")

        # Test loss calculation
        dummy_pred = torch.randn(2, 2, 128, 128).to(device)
        dummy_mask = torch.randint(0, 2, (2, 128, 128)).to(device)

        print("Testing loss calculation...")
        start = time.time()
        loss1 = focal_loss(dummy_pred, dummy_mask)
        loss2 = dice_loss(dummy_pred, dummy_mask)
        total_loss = loss1 + loss2
        elapsed = time.time() - start

        print(f"Loss calculation completed in {elapsed:.3f}s")
        print(f"Focal loss: {loss1.item():.4f}")
        print(f"Dice loss: {loss2.item():.4f}")
        print(f"Total loss: {total_loss.item():.4f}")
        print("✓ PASSED\n")
        return focal_loss, dice_loss
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        raise

def test_7_single_batch_training(model, loader, focal_loss, dice_loss, device):
    """Test 7: Single batch training step"""
    print("=" * 80)
    print("TEST 7: Single Batch Training")
    print("=" * 80)
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

        batch = next(iter(loader))
        img_high = batch['A'].to(device)
        img_low = batch['B'].to(device)
        mask = batch['L'].to(device)

        print("Running forward pass...")
        start = time.time()
        optimizer.zero_grad()
        pred = model(img_high, img_low)
        loss = focal_loss(pred, mask) + dice_loss(pred, mask)

        print("Running backward pass...")
        loss.backward()
        optimizer.step()
        elapsed = time.time() - start

        print(f"Training step completed in {elapsed:.3f}s")
        print(f"Loss: {loss.item():.4f}")
        print("✓ PASSED\n")
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        raise

def test_8_full_epoch(model, loader, focal_loss, dice_loss, device):
    """Test 8: Full epoch training"""
    print("=" * 80)
    print("TEST 8: Full Epoch Training")
    print("=" * 80)
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

        print(f"Training on {len(loader)} batches...")
        start = time.time()
        running_loss = 0.0

        for i, batch in enumerate(loader):
            img_high = batch['A'].to(device)
            img_low = batch['B'].to(device)
            mask = batch['L'].to(device)

            optimizer.zero_grad()
            pred = model(img_high, img_low)
            loss = focal_loss(pred, mask) + dice_loss(pred, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 20 == 0:
                print(f"  Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}")

        elapsed = time.time() - start
        avg_loss = running_loss / len(loader)

        print(f"\nEpoch completed in {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Time per batch: {elapsed/len(loader):.3f}s")
        print("✓ PASSED\n")
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        raise

def test_9_ray_tune_compatibility():
    """Test 9: Ray Tune import and basic functionality"""
    print("=" * 80)
    print("TEST 9: Ray Tune Compatibility")
    print("=" * 80)
    try:
        import ray
        from ray import tune

        print("Ray Tune imported successfully")

        # Test Ray initialization
        ray.init(ignore_reinit_error=True)
        print("Ray initialized")

        # Test simple tune.run
        def dummy_trainable(config):
            import time
            time.sleep(0.1)
            return {"metric": 1.0}

        print("Running dummy trial...")
        analysis = tune.run(
            dummy_trainable,
            config={"dummy": 1},
            num_samples=1,
            verbose=0
        )
        print("Dummy trial completed")

        ray.shutdown()
        print("✓ PASSED\n")
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        raise

def main():
    print("\n" + "=" * 80)
    print("DIAGNOSTIC TEST SUITE")
    print("=" * 80 + "\n")

    try:
        test_1_cuda()
        dataset = test_2_dataset()
        loader = test_3_dataloader(dataset)
        model, device = test_4_model()
        alpha = test_5_class_weights(loader, device)
        focal_loss, dice_loss = test_6_loss_functions(alpha, device)
        test_7_single_batch_training(model, loader, focal_loss, dice_loss, device)
        test_8_full_epoch(model, loader, focal_loss, dice_loss, device)
        test_9_ray_tune_compatibility()

        print("=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print("\nIf all tests passed, the issue is likely with:")
        print("1. Ray Tune's parallel execution setup")
        print("2. Resource allocation conflicts")
        print("3. Ray worker communication/serialization")
        print("\nThe core training code itself works fine.")

    except Exception as e:
        print("\n" + "=" * 80)
        print("DIAGNOSTIC FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()