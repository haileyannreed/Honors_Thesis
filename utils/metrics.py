import numpy as np

class MetricTracker:
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, pred, gt):
        """
        Update confusion matrix - FULLY VECTORIZED (no loops!)

        Args:
            pred: Predicted class labels as numpy array (already argmax'd) OR torch tensor
            gt: Ground truth class labels as numpy array OR torch tensor
        """
        import torch

        # Convert to numpy if torch tensors
        if isinstance(pred, torch.Tensor):
            if pred.dim() == 4:  # [B, C, H, W] - logits
                pred = torch.argmax(pred, dim=1)  # [B, H, W]
            pred = pred.cpu().numpy()

        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()

        # Flatten arrays
        pred = pred.flatten().astype(np.int64)
        gt = gt.flatten().astype(np.int64)

        # Filter valid indices
        valid_mask = (pred >= 0) & (pred < self.n_classes) & (gt >= 0) & (gt < self.n_classes)
        pred = pred[valid_mask]
        gt = gt[valid_mask]

        # VECTORIZED confusion matrix update using np.bincount
        # This is 100-1000x faster than looping!
        indices = self.n_classes * gt + pred
        bincount = np.bincount(indices, minlength=self.n_classes**2)
        self.confusion_matrix += bincount.reshape(self.n_classes, self.n_classes)

    def get_scores(self):
        """Calculate metrics from confusion matrix"""
        # Per-class metrics
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        # Precision, Recall, F1 per class
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # Overall metrics
        accuracy = tp.sum() / (self.confusion_matrix.sum() + 1e-10)
        mean_f1 = f1.mean()

        return {
            'accuracy': accuracy,
            'mean_f1': mean_f1,
            'f1_per_class': f1,
            'precision_per_class': precision,
            'recall_per_class': recall
        }

    def clear(self):
        """Reset confusion matrix"""
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)