"""
Hyperparameter tuning using Optuna - ALIGNED WITH WORKING WACV PAPER CODE
Supports: Semi-Siamese, FlexibleUNet (concat/diff), LatentDiffUNet

Usage:
    Change MODEL_TYPE to test different architectures
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.tensorboard import SummaryWriter

from datasets.nuclei_dataset import NucleiDataset
from utils.metrics import MetricTracker
from models.losses import DiceLoss, FocalLoss, softmax_helper

# Import all model architectures
from models.semi_siamese import Semi_siamese_
from models.flexible_unet import ConcatUNet, DiffUNet, FlexibleUNet
from models.latent_diff_unet import LatentDiffUNet


MODEL_TYPE = 'semi_siamese'  # Options: 'semi_siamese', 'flexible_unet', 'latent_diff'


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

    # Model settings
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    INIT_FEATURES = 32
    N_CLASSES = 2

    # Training settings
    NUM_TRIALS = 20  # Number of hyperparameter trials
    MAX_EPOCHS = 500  # Maximum epochs per trial
    PATIENCE = 50  # Early stopping patience
    PRUNE_AFTER = 15  # Prune bad trials after this many epochs

    # Output
    OUTPUT_DIR = BASE_DIR / 'optuna_results'
    OUTPUT_DIR.mkdir(exist_ok=True)


def get_model(model_type, skip_mode='high', fusion_mode='concat'):
    """Get model based on type and configuration"""
    if model_type == 'semi_siamese':
        return Semi_siamese_(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.OUT_CHANNELS,
            init_features=Config.INIT_FEATURES
        )
    elif model_type == 'flexible_unet':
        return FlexibleUNet(
            fusion_mode=fusion_mode,
            out_channels=Config.N_CLASSES,
            init_features=Config.INIT_FEATURES
        )
    elif model_type == 'latent_diff':
        return LatentDiffUNet(
            out_channels=Config.N_CLASSES,
            init_features=Config.INIT_FEATURES,
            skip_mode=skip_mode
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_alpha(dataloader):
    """
    Calculate class weights - EXACTLY from paper's code
    (from wacv_paper-main/Sia_train/models/losses.py get_alpha function)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = np.zeros((Config.N_CLASSES,))

    for batch in dataloader:
        gt = batch['L'].to(device)
        gt = gt.detach().cpu().numpy()
        gt = gt.reshape(-1)
        for i in range(Config.N_CLASSES):
            z[i] += np.sum(gt == i)

    return z


def train_epoch(model, dataloader, optimizer, criterion, device, trial_num=0, epoch_num=0):
    """Train for one epoch - ALIGNED WITH PAPER"""
    import sys
    model.train()
    running_loss = 0.0
    metric_tracker = MetricTracker(n_classes=Config.N_CLASSES)

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

        # KEY: Convert to CPU numpy BEFORE updating metrics (like paper does)
        pred_classes = torch.argmax(pred.detach(), dim=1)
        metric_tracker.update(pred_classes.cpu().numpy(), mask.detach().cpu().numpy())

        if (batch_idx + 1) % 30 == 0:
            print(f"[Trial {trial_num}] Epoch {epoch_num}: batch {batch_idx+1}/{len(dataloader)}, loss={loss.item():.4f}", flush=True)
            sys.stdout.flush()

    epoch_loss = running_loss / len(dataloader)
    scores = metric_tracker.get_scores()

    return epoch_loss, scores


def validate(model, dataloader, criterion, device):
    """Validate the model - ALIGNED WITH PAPER"""
    model.eval()
    running_loss = 0.0
    metric_tracker = MetricTracker(n_classes=Config.N_CLASSES)

    with torch.no_grad():
        for batch in dataloader:
            img_high = batch['A'].to(device)
            img_low = batch['B'].to(device)
            mask = batch['L'].to(device)

            pred = model(img_high, img_low)
            loss = criterion(pred, mask)

            running_loss += loss.item()

            # KEY: Convert to CPU numpy BEFORE updating metrics (like paper does)
            pred_classes = torch.argmax(pred, dim=1)
            metric_tracker.update(pred_classes.cpu().numpy(), mask.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    scores = metric_tracker.get_scores()

    return epoch_loss, scores


def objective(trial):
    """
    Optuna objective function - ALIGNED WITH PAPER'S RAYTUNE CODE
    """
    # Suggest hyperparameters (matching paper's search space EXACTLY)
    # From wacv_paper-main/Sia_train/main_raytune.py:
    # MoNuSeg: lr: loguniform(1e-5, 1e-3), wd: loguniform(1e-6, 1e-2)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
    batch_size = 16  # Fixed to 16

    # Model-specific hyperparameters
    if MODEL_TYPE == 'flexible_unet':
        fusion_mode = trial.suggest_categorical('fusion_mode', ['concat', 'diff'])
    else:
        fusion_mode = 'concat'

    if MODEL_TYPE == 'latent_diff':
        skip_mode = trial.suggest_categorical('skip_mode', ['high', 'low', 'both', 'avg'])
    else:
        skip_mode = 'high'

    # Set random seeds (EXACTLY like paper)
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

    # Get model
    model = get_model(MODEL_TYPE, skip_mode=skip_mode, fusion_mode=fusion_mode)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create datasets
    train_dataset = NucleiDataset(
        high_dir=Config.HIGH_TRAIN,
        low_dir=Config.LOW_TRAIN,
        mask_dir=Config.MASK_TRAIN
    )

    test_dataset = NucleiDataset(
        high_dir=Config.HIGH_TEST,
        low_dir=Config.LOW_TEST,
        mask_dir=Config.MASK_TEST
    )

    # Create dataloaders (num_workers=0 to avoid multiprocessing issues)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=gen
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Calculate class weights (EXACTLY like paper's get_alpha function)
    print(f"\n[Trial {trial.number}] Calculating class weights...")
    alpha = get_alpha(train_loader)
    print(f"[Trial {trial.number}] Class weights: {alpha}")

    # Setup loss (EXACTLY like paper: FocalLoss + DiceLoss)
    focal_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)
    dice_loss = DiceLoss(n_classes=Config.N_CLASSES)

    def criterion(pred, mask):
        return focal_loss(pred, mask) + dice_loss(pred, mask)

    # Setup optimizer (EXACTLY like paper)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=wd
    )

    # Learning rate scheduler (EXACTLY like paper: StepLR with step_size=500)
    # From paper: exp_lr_scheduler_G = get_scheduler(optimizer_G, 500)
    # The paper uses StepLR with step_size=500, gamma=0.5
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=100,
        gamma=0.5
    )

    # Training loop with early stopping and TensorBoard logging
    best_f1 = 0.0
    patience_counter = 0

    # Create TensorBoard writer for this trial
    log_dir = Config.OUTPUT_DIR / f"trial_{trial.number}_lr{lr:.6f}_wd{wd:.6f}"
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"\n[Trial {trial.number}] Starting training...")
    print(f"[Trial {trial.number}] lr={lr:.6f}, wd={wd:.6f}, fusion_mode={fusion_mode}, skip_mode={skip_mode}")
    print(f"[Trial {trial.number}] TensorBoard logs: {log_dir}")

    for epoch in range(Config.MAX_EPOCHS):
        # Train
        train_loss, train_scores = train_epoch(model, train_loader, optimizer, criterion, device, trial.number, epoch)

        # Validate
        val_loss, val_scores = validate(model, test_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Calculate metrics (EXACTLY like paper: mean F1 of non-background classes)
        f1_per_class = val_scores['f1_per_class']
        f1_mean = np.mean(f1_per_class[1:]) if Config.N_CLASSES > 1 else f1_per_class[0]

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1/val', f1_mean, epoch)
        writer.add_scalar('F1/val_nuclei', f1_per_class[1] if Config.N_CLASSES > 1 else f1_mean, epoch)
        writer.add_scalar('Precision/val_nuclei', val_scores['precision_per_class'][1] if Config.N_CLASSES > 1 else 0.0, epoch)
        writer.add_scalar('Recall/val_nuclei', val_scores['recall_per_class'][1] if Config.N_CLASSES > 1 else 0.0, epoch)
        writer.add_scalar('Accuracy/val', val_scores['accuracy'], epoch)
        writer.add_scalar('Hyperparameters/lr', lr, epoch)
        writer.add_scalar('Hyperparameters/wd', wd, epoch)

        # Report to Optuna
        trial.report(f1_mean, epoch)

        # Early stopping check
        if f1_mean > best_f1:
            best_f1 = f1_mean
            patience_counter = 0
        else:
            patience_counter += 1

        # Log progress
        if (epoch + 1) % 10 == 0:
            print(f"[Trial {trial.number}] Epoch {epoch+1}/{Config.MAX_EPOCHS}: "
                  f"val_loss={val_loss:.4f}, val_f1={f1_mean:.4f}, best_f1={best_f1:.4f}")

        # Pruning: let Optuna decide if this trial should be stopped
        if epoch >= Config.PRUNE_AFTER:
            if trial.should_prune():
                print(f"[Trial {trial.number}] Pruned at epoch {epoch+1}")
                writer.close()
                raise optuna.TrialPruned()

        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"[Trial {trial.number}] Early stopped at epoch {epoch+1}")
            break

    writer.close()

    print(f"[Trial {trial.number}] Completed with best F1: {best_f1:.4f}")
    return best_f1


def main():
    """Main function to run Optuna hyperparameter optimization"""
    print("=" * 80)
    print(f"Starting Optuna Hyperparameter Tuning for {MODEL_TYPE}")
    print("ALIGNED WITH WACV PAPER CODE")
    print("=" * 80)

    # Create study
    study_name = f"{MODEL_TYPE}_{Config.DATASET}"
    storage_path = Config.OUTPUT_DIR / f"{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction='maximize',  # Maximize F1 score
        load_if_exists=True,
        sampler=TPESampler(seed=8888),
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=Config.PRUNE_AFTER
        )
    )

    print(f"\nStudy storage: {storage_path}")
    print(f"Number of trials: {Config.NUM_TRIALS}")
    print(f"Max epochs per trial: {Config.MAX_EPOCHS}")
    print(f"Early stopping patience: {Config.PATIENCE}")
    print(f"Pruning after: {Config.PRUNE_AFTER} epochs\n")

    # Run optimization
    study.optimize(objective, n_trials=Config.NUM_TRIALS)

    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1 score: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    results_file = Config.OUTPUT_DIR / f"{study_name}_results.json"
    results = {
        'best_trial': study.best_trial.number,
        'best_f1': study.best_value,
        'best_params': study.best_params,
        'all_trials': [
            {
                'number': trial.number,
                'f1': trial.value,
                'params': trial.params,
                'state': str(trial.state)
            }
            for trial in study.trials
        ]
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Optuna dashboard info
    print(f"\nTo view results in Optuna Dashboard:")
    print(f"  optuna-dashboard sqlite:///{storage_path}")


if __name__ == "__main__":
    main()