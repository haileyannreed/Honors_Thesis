"""
Hyperparameter tuning script using Ray Tune (following paper's approach)
Supports: Semi-Siamese, ConcatUNet, DiffUNet, LatentDiffUNet

Usage:
    Change MODEL_TYPE to test different architectures
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
from functools import partial

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

    # Model settings
    IN_CHANNELS = 3
    OUT_CHANNELS = 3  # For semi_siamese decoder output
    INIT_FEATURES = 32
    N_CLASSES = 2

    # Ray Tune settings
    NUM_SAMPLES = 20  # Number of trials
    MAX_EPOCHS = 500
    GRACE_PERIOD = 15  # Train each trial at least for n epochs
    GPUS_PER_TRIAL = 1

    # Output
    OUTPUT_DIR = BASE_DIR / 'ray_tune_results'
    OUTPUT_DIR.mkdir(exist_ok=True)


torch.use_deterministic_algorithms(True)


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
            out_channels=Config.N_CLASSES,
            init_features=Config.INIT_FEATURES
        )
    elif model_type == 'diff_unet':
        return DiffUNet(
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

    for batch_idx, (img_high, img_low, mask) in enumerate(dataloader):
        img_high = img_high.to(device)
        img_low = img_low.to(device)
        mask = mask.to(device)

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
        for batch_idx, (img_high, img_low, mask) in enumerate(dataloader):
            img_high = img_high.to(device)
            img_low = img_low.to(device)
            mask = mask.to(device)

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
    """Calculate class weights from training data"""
    print("Calculating class weights from training data...")
    class_counts = torch.zeros(Config.N_CLASSES, device=device)

    for _, _, mask in dataloader:
        mask = mask.to(device)
        for c in range(Config.N_CLASSES):
            class_counts[c] += (mask == c).sum()

    total_pixels = class_counts.sum()
    class_weights = total_pixels / (Config.N_CLASSES * class_counts)
    alpha = class_weights / class_weights.sum()

    for i, a in enumerate(alpha):
        print(f"alpha-{i} {'(background)' if i == 0 else '(nuclei)'}={a:.4f}")

    return alpha.cpu().numpy()


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

    # Get model with skip_mode if needed
    skip_mode = config.get('skip_mode', 'high')
    model = get_model(MODEL_TYPE, skip_mode=skip_mode)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # Create datasets
    train_dataset = NucleiDataset(
        high_fidelity_dir=Config.HIGH_TRAIN,
        low_fidelity_dir=Config.LOW_TRAIN,
        mask_dir=Config.MASK_TRAIN
    )

    test_dataset = NucleiDataset(
        high_fidelity_dir=Config.HIGH_TEST,
        low_fidelity_dir=Config.LOW_TEST,
        mask_dir=Config.MASK_TEST
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=gen
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Calculate class weights
    alpha = calculate_class_weights(train_loader, device)

    # Setup loss function and optimizer
    criterion = FocalLoss(alpha=alpha, gamma=2)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.999),
        weight_decay=config['wd']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=100,
        gamma=0.5
    )

    # Load checkpoint if resuming
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, Config.MAX_EPOCHS):
        # Train
        train_loss, train_scores = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_scores = validate(model, test_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Calculate mean F1 score
        f1_mean = np.mean([val_scores['f1_score'][i] for i in range(1, Config.N_CLASSES)])

        # Report metrics to Ray Tune
        metrics = {
            "loss": val_loss,
            "f1": f1_mean,
            "f1_1": val_scores['f1_score'][1] if Config.N_CLASSES > 1 else 0.0,
            "f1_2": val_scores['f1_score'][1] if Config.N_CLASSES > 1 else 0.0,  # Same as f1_1 for binary
            "iou": val_scores['iou'][1] if Config.N_CLASSES > 1 else 0.0,
            "precision": val_scores['precision'][1] if Config.N_CLASSES > 1 else 0.0,
            "recall": val_scores['recall'][1] if Config.N_CLASSES > 1 else 0.0,
        }

        # Save checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        checkpoint = Checkpoint.from_dict(checkpoint_data)
        train.report(metrics, checkpoint=checkpoint)


def main(num_samples=20, max_num_epochs=500, gpus_per_trial=1):
    """
    Main function to run Ray Tune hyperparameter optimization
    """
    # Set Ray temp directory
    os.environ["RAY_TMPDIR"] = "/tmp/ray_tmp"
    ray.init(_temp_dir="/tmp/ray_tmp", ignore_reinit_error=True)

    # Define search space
    if MODEL_TYPE == 'latent_diff':
        # Add skip_mode tuning for latent_diff
        config = {
            "lr": tune.loguniform(1e-5, 1e-2),
            "wd": tune.loguniform(1e-6, 1e-2),
            "batch_size": tune.choice([4, 8, 16]),
            "skip_mode": tune.choice(['high', 'low', 'both', 'avg']),
        }
    else:
        config = {
            "lr": tune.loguniform(1e-5, 1e-2),
            "wd": tune.loguniform(1e-6, 1e-2),
            "batch_size": tune.choice([4, 8, 16]),
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
            resources={"cpu": 4, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=train.RunConfig(
            storage_path=str(Config.OUTPUT_DIR),
            name=f"{MODEL_TYPE}_{Config.DATASET}",
            checkpoint_config=train.CheckpointConfig(
                checkpoint_score_attribute="f1",
                num_to_keep=2,
            ),
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
    print(f"Best trial final validation F1: {best_trial.metrics['f1']:.4f}")
    print(f"Best trial final validation IoU: {best_trial.metrics['iou']:.4f}")
    print(f"Best trial final validation Precision: {best_trial.metrics['precision']:.4f}")
    print(f"Best trial final validation Recall: {best_trial.metrics['recall']:.4f}")

    # Save best hyperparameters
    output_file = Config.BASE_DIR / f'best_hyperparams_{Config.DATASET}_{MODEL_TYPE}_raytune.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model_type': MODEL_TYPE,
            'dataset': Config.DATASET,
            'best_config': best_trial.config,
            'best_metrics': {
                'loss': best_trial.metrics['loss'],
                'f1': best_trial.metrics['f1'],
                'iou': best_trial.metrics['iou'],
                'precision': best_trial.metrics['precision'],
                'recall': best_trial.metrics['recall'],
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
