"""
5-Fold Cross-Validation script mirroring train_kfold.py
Uses best hyperparameters found by hyperparam_tuning_simple.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
import numpy as np
import json
from pathlib import Path
import os

from models.semi_siamese import Semi_siamese_
from datasets.nuclei_dataset import NucleiDataset
from utils.metrics import MetricTracker
from models.losses import DiceLoss, FocalLoss, get_alpha
from models.nets import get_scheduler


# Configuration
class Config:
    # Paths - UPDATE THESE FOR EACH DATASET
    BASE_DIR = Path('/Users/haileyreed/Desktop/Honors_Thesis')
    PATCHES_DIR = BASE_DIR / 'Patches'

    # Choose dataset: 'MoNuSeg', 'GlaS', or 'TNBC'
    DATASET = 'MoNuSeg'

    HIGH_TRAIN = PATCHES_DIR / f'{DATASET}_train_high'
    LOW_TRAIN = PATCHES_DIR / f'{DATASET}_train_low'
    MASK_TRAIN = PATCHES_DIR / f'{DATASET}_train_mask'

    HIGH_TEST = PATCHES_DIR / f'{DATASET}_test_high'
    LOW_TEST = PATCHES_DIR / f'{DATASET}_test_low'
    MASK_TEST = PATCHES_DIR / f'{DATASET}_test_mask'

    # Load best hyperparameters from tuning
    HYPERPARAM_FILE = BASE_DIR / f'best_hyperparams_{DATASET}.json'

    # Training settings
    IMG_SIZE = 128
    NUM_EPOCHS = 500  # Full training
    N_SPLITS = 5  # 5-fold CV

    # Model settings
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    INIT_FEATURES = 32
    N_CLASSES = 2

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Checkpoints
    CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / DATASET
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Seed for reproducibility
    SEED = 8888


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_alpha_weights(dataloader):
    """Calculate class weights from training data"""
    print("Calculating class weights from training data...")
    class_counts = np.zeros(2)

    for batch in dataloader:
        masks = batch['mask'].numpy()
        unique, counts = np.unique(masks, return_counts=True)
        for u, c in zip(unique, counts):
            if u < 2:
                class_counts[int(u)] += c

    total = class_counts.sum()
    alpha = class_counts / total
    print(f"alpha-0 (background)={alpha[0]:.4f}, alpha-1 (nuclei)={alpha[1]:.4f}")
    return alpha


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    metric_tracker = MetricTracker(n_classes=2)

    for batch in dataloader:
        img_high = batch['high'].to(device)
        img_low = batch['low'].to(device)
        mask = batch['mask'].to(device).long()

        # Forward pass
        pred = model(img_high, img_low)
        loss = criterion(pred, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        pred_class = torch.argmax(pred.detach(), dim=1)
        metric_tracker.update(pred_class.cpu().numpy(), mask.cpu().numpy())
        total_loss += loss.item()

    scores = metric_tracker.get_scores()
    return total_loss / len(dataloader), scores


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    metric_tracker = MetricTracker(n_classes=2)

    with torch.no_grad():
        for batch in dataloader:
            img_high = batch['high'].to(device)
            img_low = batch['low'].to(device)
            mask = batch['mask'].to(device).long()

            # Forward pass
            pred = model(img_high, img_low)
            loss = criterion(pred, mask)

            # Metrics
            pred_class = torch.argmax(pred, dim=1)
            metric_tracker.update(pred_class.cpu().numpy(), mask.cpu().numpy())
            total_loss += loss.item()

    scores = metric_tracker.get_scores()
    return total_loss / len(dataloader), scores


def train_fold(fold, train_idx, val_idx, full_dataset, config, hyperparams):
    """Train a single fold"""
    print(f"\n{'='*80}")
    print(f"Starting Fold {fold+1}/{config.N_SPLITS}")
    print(f"{'='*80}")

    # Create train/val subsets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Calculate alpha weights
    alpha = get_alpha_weights(train_loader)

    # Create model
    model = Semi_siamese_(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        init_features=config.INIT_FEATURES
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(config.DEVICE)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparams['lr'],
        betas=(0.9, 0.999),
        weight_decay=hyperparams['weight_decay']
    )

    # Learning rate scheduler
    scheduler = get_scheduler(optimizer, config.NUM_EPOCHS)

    # Loss function - using FocalLoss + DiceLoss combined (like train_semi_ray.py)
    focal_loss = FocalLoss(alpha=alpha, gamma=2)
    dice_loss = DiceLoss(n_classes=2)

    # Combined loss function
    def criterion(pred, target):
        return focal_loss(pred, target) + dice_loss(pred, target)

    # Training loop
    best_f1_1 = 0.0
    best_epoch = 0

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_scores = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        val_loss, val_scores = validate(model, val_loader, criterion, config.DEVICE)

        scheduler.step()

        # Get metrics
        train_f1_macro = train_scores['mean_f1']
        val_f1_0 = val_scores['f1_per_class'][0]
        val_f1_1 = val_scores['f1_per_class'][1]
        val_f1_macro = val_scores['mean_f1']

        # Print progress
        print(f"[Epoch {epoch+1}/{config.NUM_EPOCHS}] "
              f"train_loss: {train_loss:.4f} | train_mf1: {train_f1_macro:.4f}")
        print(f"[Epoch {epoch+1}/{config.NUM_EPOCHS}] "
              f"val_loss: {val_loss:.4f} | val_F1_0: {val_f1_0:.4f} | "
              f"val_F1_1: {val_f1_1:.4f} | val_mf1: {val_f1_macro:.4f}")

        # Save best model for this fold
        if val_f1_1 > best_f1_1:
            best_f1_1 = val_f1_1
            best_epoch = epoch

            # Save checkpoint
            checkpoint_path = config.CHECKPOINT_DIR / f'fold_{fold+1}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1_1': val_f1_1,
                'val_f1_0': val_f1_0,
                'val_loss': val_loss,
                'hyperparams': hyperparams
            }, checkpoint_path)

    print(f"\nFold {fold+1} completed!")
    print(f"Best F1_1: {best_f1_1:.4f} at epoch {best_epoch+1}")

    return {
        'fold': fold + 1,
        'best_f1_1': best_f1_1,
        'best_epoch': best_epoch,
        'final_val_loss': val_loss,
        'final_val_f1_0': val_f1_0,
        'final_val_f1_1': val_f1_1,
        'final_val_f1_macro': val_f1_macro
    }


def main():
    # Set seed
    set_seed(Config.SEED)

    print("="*80)
    print("5-FOLD CROSS-VALIDATION")
    print(f"Dataset: {Config.DATASET}")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs per fold: {Config.NUM_EPOCHS}")
    print("="*80)

    # Load best hyperparameters
    if not Config.HYPERPARAM_FILE.exists():
        print(f"\nERROR: Hyperparameter file not found: {Config.HYPERPARAM_FILE}")
        print("Please run hyperparam_tuning_simple.py first!")
        return

    with open(Config.HYPERPARAM_FILE, 'r') as f:
        hyperparams = json.load(f)

    print("\nLoaded hyperparameters:")
    print(f"  Learning Rate: {hyperparams['lr']}")
    print(f"  Weight Decay: {hyperparams['weight_decay']}")
    print(f"  Batch Size: {hyperparams['batch_size']}")
    print(f"  (Found with F1_1: {hyperparams['best_f1_1']:.4f})")

    # Create full training dataset (we'll split it with k-fold)
    full_dataset = NucleiDataset(
        Config.HIGH_TRAIN,
        Config.LOW_TRAIN,
        Config.MASK_TRAIN,
        img_size=Config.IMG_SIZE,
        is_train=True
    )

    print(f"\nTotal training samples: {len(full_dataset)}")

    # Create k-fold splitter
    kf = KFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)

    # Store results for each fold
    fold_results = []

    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"\nFold {fold+1}: {len(train_idx)} train samples, {len(val_idx)} val samples")

        result = train_fold(fold, train_idx, val_idx, full_dataset, Config, hyperparams)
        fold_results.append(result)

    # Print summary
    print("\n" + "="*80)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("="*80)

    for r in fold_results:
        print(f"Fold {r['fold']}: "
              f"best_F1_1={r['best_f1_1']:.4f} (epoch {r['best_epoch']+1}), "
              f"final_loss={r['final_val_loss']:.4f}")

    # Calculate mean and std
    f1_1_scores = [r['best_f1_1'] for r in fold_results]
    mean_f1_1 = np.mean(f1_1_scores)
    std_f1_1 = np.std(f1_1_scores)

    print("\n" + "="*80)
    print(f"Mean F1_1 over {Config.N_SPLITS} folds: {mean_f1_1:.4f} ± {std_f1_1:.4f}")
    print("="*80)

    # Save results
    results_file = Config.CHECKPOINT_DIR / 'cv_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': Config.DATASET,
            'hyperparams': hyperparams,
            'fold_results': fold_results,
            'mean_f1_1': mean_f1_1,
            'std_f1_1': std_f1_1
        }, f, indent=4)

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
