"""
Hyperparameter tuning script for ANY model architecture
Supports: Semi-Siamese, ConcatUNet, DiffUNet, LatentDiffUNet

Usage:
    Change MODEL_TYPE to test different architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from pathlib import Path
import json
import numpy as np

from datasets.nuclei_dataset import NucleiDataset
from utils.metrics import MetricTracker
from models.losses import DiceLoss, FocalLoss

# Import all model architectures
from models.semi_siamese import Semi_siamese_
from models.flexible_unet import ConcatUNet, DiffUNet
from models.latent_diff_unet import LatentDiffUNet


# ============= CHANGE THIS TO TEST DIFFERENT MODELS =============
MODEL_TYPE = 'semi_siamese'  # Options: 'semi_siamese', 'concat_unet', 'diff_unet', 'latent_diff'
# =================================================================


# Configuration
class Config:
    # Paths
    BASE_DIR = Path('/root/Honors_Thesis')
    PATCHES_DIR = BASE_DIR / 'Patches'

    # Dataset
    DATASET = 'MoNuSeg'

    HIGH_TRAIN = PATCHES_DIR / f'{DATASET}_train_high'
    LOW_TRAIN = PATCHES_DIR / f'{DATASET}_train_low'
    MASK_TRAIN = PATCHES_DIR / f'{DATASET}_train_mask'

    HIGH_TEST = PATCHES_DIR / f'{DATASET}_test_high'
    LOW_TEST = PATCHES_DIR / f'{DATASET}_test_low'
    MASK_TEST = PATCHES_DIR / f'{DATASET}_test_mask'

    # Training settings
    IMG_SIZE = 128
    NUM_EPOCHS = 50  # Shorter for tuning
    NUM_TRIALS = 20

    # Model settings
    IN_CHANNELS = 3
    OUT_CHANNELS = 2
    INIT_FEATURES = 32
    N_CLASSES = 2

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Output
    OUTPUT_FILE = BASE_DIR / f'best_hyperparams_{DATASET}_{MODEL_TYPE}.json'


def get_model(model_type, skip_mode='high'):
    """Get model based on type and skip_mode (for latent_diff)"""
    if model_type == 'semi_siamese':
        return Semi_siamese_(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.OUT_CHANNELS,
            init_features=Config.INIT_FEATURES
        )
    elif model_type == 'concat_unet':
        return ConcatUNet(
            out_channels=Config.OUT_CHANNELS,
            init_features=Config.INIT_FEATURES
        )
    elif model_type == 'diff_unet':
        return DiffUNet(
            out_channels=Config.OUT_CHANNELS,
            init_features=Config.INIT_FEATURES
        )
    elif model_type == 'latent_diff':
        return LatentDiffUNet(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.OUT_CHANNELS,
            init_features=Config.INIT_FEATURES,
            skip_mode=skip_mode  # Support configurable skip connections
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_alpha_weights(dataloader):
    """Calculate class weights from training data"""
    print("Calculating class weights from training data...")
    class_counts = np.zeros(2)

    for batch in dataloader:
        masks = batch['L'].numpy()
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
        img_high = batch['A'].to(device)
        img_low = batch['B'].to(device)
        mask = batch['L'].to(device).long()

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
            img_high = batch['A'].to(device)
            img_low = batch['B'].to(device)
            mask = batch['L'].to(device).long()

            # Forward pass
            pred = model(img_high, img_low)
            loss = criterion(pred, mask)

            # Metrics
            pred_class = torch.argmax(pred, dim=1)
            metric_tracker.update(pred_class.cpu().numpy(), mask.cpu().numpy())
            total_loss += loss.item()

    scores = metric_tracker.get_scores()
    return total_loss / len(dataloader), scores


def objective(trial):
    """Optuna objective function"""

    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])

    # Add skip_mode tuning for latent_diff model
    skip_mode = 'high'  # default for other models
    if MODEL_TYPE == 'latent_diff':
        skip_mode = trial.suggest_categorical('skip_mode', ['high', 'low', 'both', 'avg'])
        print(f"\nTrial {trial.number}: lr={lr:.6f}, wd={weight_decay:.6f}, batch_size={batch_size}, skip_mode={skip_mode}")
    else:
        print(f"\nTrial {trial.number}: lr={lr:.6f}, wd={weight_decay:.6f}, batch_size={batch_size}")

    # Create datasets
    train_dataset = NucleiDataset(
        Config.HIGH_TRAIN, Config.LOW_TRAIN, Config.MASK_TRAIN,
        img_size=Config.IMG_SIZE, is_train=True
    )

    test_dataset = NucleiDataset(
        Config.HIGH_TEST, Config.LOW_TEST, Config.MASK_TEST,
        img_size=Config.IMG_SIZE, is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Calculate alpha weights
    alpha = get_alpha_weights(train_loader)

    # Create model with skip_mode parameter
    model = get_model(MODEL_TYPE, skip_mode=skip_mode)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(Config.DEVICE)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    # Loss function - FocalLoss + DiceLoss combined
    focal_loss = FocalLoss(alpha=alpha, gamma=2)
    dice_loss = DiceLoss(n_classes=2)

    def criterion(pred, target):
        return focal_loss(pred, target) + dice_loss(pred, target)

    # Training loop
    best_f1 = 0.0

    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_scores = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        val_loss, val_scores = validate(model, test_loader, criterion, Config.DEVICE)

        f1_1 = val_scores['f1_per_class'][1]

        if f1_1 > best_f1:
            best_f1 = f1_1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{Config.NUM_EPOCHS}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1_1={f1_1:.4f}, best_f1_1={best_f1:.4f}")

        trial.report(f1_1, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_f1


def main():
    print("="*80)
    print(f"HYPERPARAMETER TUNING - {MODEL_TYPE.upper()}")
    print(f"Dataset: {Config.DATASET}")
    print(f"Device: {Config.DEVICE}")
    print(f"Number of trials: {Config.NUM_TRIALS}")
    print(f"Epochs per trial: {Config.NUM_EPOCHS}")
    print("="*80)

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=Config.NUM_TRIALS)

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1_1 score: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_params = {
        'model_type': MODEL_TYPE,
        'lr': study.best_params['lr'],
        'weight_decay': study.best_params['wd'],
        'batch_size': study.best_params['batch_size'],
        'best_f1_1': study.best_value,
        'trial_number': study.best_trial.number
    }

    # Add skip_mode to saved params if using latent_diff
    if MODEL_TYPE == 'latent_diff':
        best_params['skip_mode'] = study.best_params['skip_mode']

    with open(Config.OUTPUT_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"\nBest hyperparameters saved to: {Config.OUTPUT_FILE}")


if __name__ == '__main__':
    main()
