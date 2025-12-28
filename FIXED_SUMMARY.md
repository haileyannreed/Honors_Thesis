# HYPERPARAMETER TUNING - FIXED AND ALIGNED WITH PAPER

## What Was Wrong

After 24+ hours of debugging, the root cause was found:

**MetricTracker.update() was iterating over GPU tensors element-by-element**

This caused 524,288 GPU→CPU transfers per batch (16 × 2 × 128 × 128), resulting in catastrophic slowdown that eventually hung the process.

## The Fix

The paper's working code does this:
```python
# Convert to CPU numpy FIRST
pred_classes = torch.argmax(pred.detach(), dim=1)
metric_tracker.update(pred_classes.cpu().numpy(), mask.cpu().numpy())
```

## Changes Made

1. **hyperparam_tuning_optuna.py** - Completely rewritten to match paper's working code:
   - Uses paper's `get_alpha()` function for class weights
   - Converts tensors to CPU numpy before passing to MetricTracker
   - Matches all hyperparameters (seed=8888, lr/wd ranges, etc.)
   - Uses same loss: FocalLoss + DiceLoss
   - Same optimizer: AdamW with paper's settings

2. **utils/metrics.py** - Fixed to handle numpy arrays efficiently:
   - Now accepts numpy arrays (preferred) or tensors
   - Converts tensors to CPU numpy in bulk before iteration
   - No longer causes GPU→CPU transfer slowdown

## Verification Against Paper

✅ Hyperparameter ranges match paper exactly:
   - lr: loguniform(1e-5, 1e-3)
   - wd: loguniform(1e-6, 1e-2)
   - batch_size: 16

✅ Loss function matches: FocalLoss(alpha, gamma=2) + DiceLoss

✅ Optimizer matches: AdamW(lr, betas=(0.9, 0.999), weight_decay=wd)

✅ Seeds match: seed=8888

✅ Metric calculation matches: mean F1 of non-background classes

## How to Run

```bash
# SSH to RunPod
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519

# Start hyperparameter tuning
cd /root/Honors_Thesis
nohup python hyperparam_tuning_optuna.py > tuning.log 2>&1 &

# Monitor progress
tail -f tuning.log

# Start TensorBoard (in separate terminal)
ssh -L 6006:localhost:6006 root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
tensorboard --logdir=/root/Honors_Thesis/optuna_results --port=6006
# Then open: http://localhost:6006
```

## Expected Performance

- Each epoch should take ~1-2 minutes (148 batches)
- First trial should complete 15-20 epochs before potential pruning
- TensorBoard should show metrics updating every epoch
- 20 trials with early stopping should complete in ~12-24 hours

## Paper's Optimal Hyperparameters for MoNuSeg

- lr: 0.00023586175632493136 (2.36e-4)
- wd: 3.214183772365648e-05 (3.21e-5)

Your tuning should find hyperparameters in the same range to prove alignment.
