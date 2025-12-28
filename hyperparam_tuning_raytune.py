"""
Hyperparameter tuning script using Ray Tune (following paper's approach)
Supports: Semi-Siamese, FlexibleUNet (concat/diff), LatentDiffUNet

Usage:
    Change MODEL_TYPE to test different architectures:
    - 'semi_siamese': Baseline Semi-Siamese architecture
    - 'flexible_unet': Tunes both concat and diff fusion modes
    - 'latent_diff': Tunes all 4 skip connection modes (high/low/both/avg)
"""

import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json

import ray
from ray import tune, train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger import TBXLoggerCallback
from functools import partial

from datasets.nuclei_dataset import NucleiDataset
from utils.metrics import MetricTracker
from models.losses import DiceLoss, FocalLoss, softmax_helper  # DiceLoss used in combined loss

# Import all model architectures
from models.semi_siamese import Semi_siamese_
from models.flexible_unet import ConcatUNet, DiffUNet, FlexibleUNet
from models.latent_diff_unet import LatentDiffUNet


MODEL_TYPE = 'semi_siamese'  # Options: 'semi_siamese', 'concat_unet', 'diff_unet', 'latent_diff'

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
    OUT_CHANNELS = 3  # For semi_siamese decoder output
    INIT_FEATURES = 32
    N_CLASSES = 2

    # Ray Tune settings
    NUM_SAMPLES = 20  # Number of trials
    MAX_EPOCHS = 500  # Max epochs per trial
    GRACE_PERIOD = 10  # Reduced from 15 for faster pruning
    GPUS_PER_TRIAL = 0.25  # Run 4 trials in parallel on 1 GPU

    # Output
    OUTPUT_DIR = BASE_DIR / 'ray_tune_results'
    OUTPUT_DIR.mkdir(exist_ok=True)


# torch.use_deterministic_algorithms(True)  # DISABLED: Can cause extreme slowdowns/hangs


def get_model(model_type, skip_mode='high', fusion_mode='concat'):
    """Get model based on type, skip_mode (for latent_diff), and fusion_mode (for flexible_unet)"""
    if model_type == 'semi_siamese':
        return Semi_siamese_(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.OUT_CHANNELS,
            init_features=Config.INIT_FEATURES
        )
    elif model_type == 'concat_unet':
        return ConcatUNet(
            out_channels=Config.N_CLASSES,
            init_features=Config.INIT_FEATURES
        )
    elif model_type == 'diff_unet':
        return DiffUNet(
            out_channels=Config.N_CLASSES,
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


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    metric_tracker = MetricTracker(n_classes=Config.N_CLASSES)

    for batch_idx, batch in enumerate(dataloader):
        img_high = batch['A'].to(device)
        img_low = batch['B'].to(device)
        mask = batch['L'].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred = model(img_high, img_low)
        loss = criterion(pred, mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        metric_tracker.update(pred, mask)

    epoch_loss = running_loss / len(dataloader)
    scores = metric_tracker.get_scores()

    return epoch_loss, scores


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    metric_tracker = MetricTracker(n_classes=Config.N_CLASSES)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            img_high = batch['A'].to(device)
            img_low = batch['B'].to(device)
            mask = batch['L'].to(device)

            # Forward pass
            pred = model(img_high, img_low)
            loss = criterion(pred, mask)

            # Track metrics
            running_loss += loss.item()
            metric_tracker.update(pred, mask)

    epoch_loss = running_loss / len(dataloader)
    scores = metric_tracker.get_scores()

    return epoch_loss, scores


def calculate_class_weights(dataloader, device):
    """Calculate class weights from training data (matching paper's get_alpha)"""
    print("Calculating class weights from training data...")
    class_counts = torch.zeros(Config.N_CLASSES, device=device)

    for batch in dataloader:
        mask = batch['L'].to(device)
        for c in range(Config.N_CLASSES):
            class_counts[c] += (mask == c).sum()

    # Return RAW pixel counts (not normalized) - FocalLoss will normalize internally
    alpha = class_counts.cpu().numpy()

    for i, count in enumerate(alpha):
        print(f"alpha-{i} {'(background)' if i == 0 else '(nuclei)'}={count:.0f} pixels")

    return alpha


def CDTrainer(config, checkpoint_dir=None):
    """
    Ray Tune training function
    Follows the paper's training approach
    """
    # Set random seeds
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

    # Get model with skip_mode and fusion_mode if needed
    skip_mode = config.get('skip_mode', 'high')
    fusion_mode = config.get('fusion_mode', 'concat')
    model = get_model(MODEL_TYPE, skip_mode=skip_mode, fusion_mode=fusion_mode)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
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

    # Create dataloaders
    # Use num_workers=0 for parallel trials to avoid CPU contention
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

    # Calculate class weights on CPU to avoid GPU contention
    # Each worker calculates independently to avoid serialization issues
    class_counts = torch.zeros(Config.N_CLASSES)
    for batch in train_loader:
        mask = batch['L']
        for c in range(Config.N_CLASSES):
            class_counts[c] += (mask == c).sum()
    alpha = class_counts.numpy()

    # Setup loss function and optimizer (matching paper's approach)
    # CRITICAL: Must pass apply_nonlin=softmax_helper to apply softmax to raw logits!
    focal_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)
    dice_loss = DiceLoss(n_classes=Config.N_CLASSES)

    def criterion(pred, mask):
        return focal_loss(pred, mask) + dice_loss(pred, mask)

    # same as in train_semi_ray.py
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.999),
        weight_decay=config['wd']
    )

    # Learning rate scheduler --> nonlinear (a bit faster than train_semi_ray)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=100,
        gamma=0.5
    )

    # Start training from scratch (Ray Tune handles checkpointing automatically)
    start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, Config.MAX_EPOCHS):
        # Train
        train_loss, train_scores = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_scores = validate(model, test_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Calculate mean F1 score (excluding background class)
        f1_per_class = val_scores['f1_per_class']
        f1_mean = np.mean(f1_per_class[1:]) if Config.N_CLASSES > 1 else f1_per_class[0]

        # Report metrics to Ray Tune
        metrics = {
            "loss": val_loss,
            "f1": f1_mean,
            "f1_nuclei": f1_per_class[1] if Config.N_CLASSES > 1 else 0.0,
            "precision": val_scores['precision_per_class'][1] if Config.N_CLASSES > 1 else 0.0,
            "recall": val_scores['recall_per_class'][1] if Config.N_CLASSES > 1 else 0.0,
            "accuracy": val_scores['accuracy'],
        }

        # Save checkpoint to temporary directory (Ray Tune API)
        import tempfile
        import ray.cloudpickle as pickle

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            # Use tune.report instead of train.report when called from Ray Tune
            tune.report(metrics, checkpoint=checkpoint)


def main(num_samples=20, max_num_epochs=500, gpus_per_trial=1):
    """
    Main function to run Ray Tune hyperparameter optimization
    """
    # Set Ray temp directory
    os.environ["RAY_TMPDIR"] = "/tmp/ray_tmp"
    ray.init(_temp_dir="/tmp/ray_tmp", ignore_reinit_error=True)

    # Calculate class weights ONCE before starting trials
    # This avoids GPU contention from 4 trials all calculating simultaneously
    print("Pre-calculating class weights...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temp_dataset = NucleiDataset(
        high_dir=Config.HIGH_TRAIN,
        low_dir=Config.LOW_TRAIN,
        mask_dir=Config.MASK_TRAIN
    )
    temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=False, num_workers=0)
    alpha = calculate_class_weights(temp_loader, device)
    alpha_list = alpha.tolist()  # Convert to list for Ray Tune config
    print(f"Class weights calculated: {alpha_list}")

    # Define search space (NARROWED based on paper's best results)
    if MODEL_TYPE == 'latent_diff':
        # Add skip_mode tuning for latent_diff
        config = {
            "lr": tune.loguniform(2e-4, 8e-4),  # NARROWED: Focus on optimal range
            "wd": tune.loguniform(1e-6, 1e-4),  # NARROWED: Avoid over-regularization
            "batch_size": tune.choice([16]),     # FIXED: Use optimal batch size only
            "skip_mode": tune.choice(['high', 'low', 'both', 'avg']),
            "alpha": alpha_list,  # Pre-calculated class weights
        }
    elif MODEL_TYPE == 'flexible_unet':
        # Add fusion_mode tuning for flexible_unet
        config = {
            "lr": tune.loguniform(2e-4, 8e-4),  # NARROWED: Focus on optimal range
            "wd": tune.loguniform(1e-6, 1e-4),  # NARROWED: Avoid over-regularization
            "batch_size": tune.choice([16]),     # FIXED: Use optimal batch size only
            "fusion_mode": tune.choice(['concat', 'diff']),
            "alpha": alpha_list,  # Pre-calculated class weights
        }
    else:
        config = {
            "lr": tune.loguniform(2e-4, 8e-4),  # NARROWED: Focus on optimal range (paper's best: 3.6e-4)
            "wd": tune.loguniform(1e-6, 1e-4),  # NARROWED: Avoid over-regularization (paper's best: 1.4e-6)
            "batch_size": tune.choice([16]),     # FIXED: Use optimal batch size only (skip slow 4 and 8)
            "alpha": alpha_list,  # Pre-calculated class weights
        }

    # ASHA scheduler (early stopping)
    scheduler = ASHAScheduler(
        metric="f1",
        mode='max',
        max_t=max_num_epochs,
        grace_period=Config.GRACE_PERIOD,
        reduction_factor=2,
    )

    # Run tuning
    tuner = tune.Tuner(
        tune.with_resources(
            CDTrainer,
            resources={"cpu": 1, "gpu": gpus_per_trial}  # Reduced CPU to allow 4 parallel trials
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=tune.RunConfig(
            storage_path=str(Config.OUTPUT_DIR),
            name=f"{MODEL_TYPE}_{Config.DATASET}",
            callbacks=[TBXLoggerCallback()],
        ),
    )

    print("=" * 80)
    print(f"HYPERPARAMETER TUNING - {MODEL_TYPE.upper()}")
    print(f"Dataset: {Config.DATASET}")
    print(f"Number of trials: {num_samples}")
    print(f"Max epochs per trial: {max_num_epochs}")
    print(f"Grace period: {Config.GRACE_PERIOD} epochs")
    print("=" * 80)

    result = tuner.fit()

    # Get best trial
    best_trial = result.get_best_result("f1", "max")

    print("\n" + "=" * 80)
    print("BEST TRIAL RESULTS")
    print("=" * 80)
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.metrics['loss']:.4f}")
    print(f"Best trial final validation F1 (mean): {best_trial.metrics['f1']:.4f}")
    print(f"Best trial final validation F1 (nuclei): {best_trial.metrics['f1_nuclei']:.4f}")
    print(f"Best trial final validation Precision: {best_trial.metrics['precision']:.4f}")
    print(f"Best trial final validation Recall: {best_trial.metrics['recall']:.4f}")
    print(f"Best trial final validation Accuracy: {best_trial.metrics['accuracy']:.4f}")

    # Save best hyperparameters
    output_file = Config.BASE_DIR / f'best_hyperparams_{Config.DATASET}_{MODEL_TYPE}_raytune.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model_type': MODEL_TYPE,
            'dataset': Config.DATASET,
            'best_config': best_trial.config,
            'best_metrics': {
                'loss': best_trial.metrics['loss'],
                'f1_mean': best_trial.metrics['f1'],
                'f1_nuclei': best_trial.metrics['f1_nuclei'],
                'precision': best_trial.metrics['precision'],
                'recall': best_trial.metrics['recall'],
                'accuracy': best_trial.metrics['accuracy'],
            }
        }, f, indent=4)

    print(f"\nBest hyperparameters saved to: {output_file}")

    ray.shutdown()


if __name__ == "__main__":
    main(
        num_samples=Config.NUM_SAMPLES,
        max_num_epochs=Config.MAX_EPOCHS,
        gpus_per_trial=Config.GPUS_PER_TRIAL
    )
