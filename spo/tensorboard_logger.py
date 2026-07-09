"""
TensorBoard Logger for SPO Training.

This module provides comprehensive logging utilities for monitoring SPO training,
including loss metrics, gradient statistics, SPM scores, and training health indicators.

Key features:
- Automatic detection of training anomalies (NaN, gradient explosion, etc.)
- SPM score distribution tracking
- Win/lose pair statistics
- Learning rate and parameter change monitoring
- Multi-step aggregation metrics
"""

import os
import math
from typing import Dict, Optional, Any, List, Tuple, Union
from collections import deque, defaultdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class SPOTensorBoardLogger:
    """
    Comprehensive TensorBoard logger for SPO training.

    This logger tracks:
    1. Loss metrics (DPO loss, KL loss, total loss)
    2. Probability ratio statistics (win/lose ratios, clip rates)
    3. DPO logits and accuracy
    4. Gradient statistics (norm, max, histogram)
    5. SPM score distributions
    6. Win/lose pair statistics
    7. Learning rate schedule
    8. Training health indicators

    Usage:
        logger = SPOTensorBoardLogger(log_dir='runs/spo_exp1')

        # Log metrics from loss function
        logger.log_loss_metrics(metrics, global_step)

        # Log gradient info
        logger.log_gradients(model, global_step)

        # Log SPM scores
        logger.log_spm_scores(win_scores, lose_scores, all_scores, global_step)

        # Check training health
        is_healthy, issues = logger.check_training_health()
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        flush_secs: int = 60,
        history_window: int = 100,
        log_histograms: bool = True,
        log_param_changes: bool = True
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Optional experiment name (appended to log_dir)
            flush_secs: How often to flush to disk
            history_window: Window size for rolling statistics
            log_histograms: Whether to log histograms (more expensive)
            log_param_changes: Whether to track parameter changes
        """
        # Create log directory
        if experiment_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join(log_dir, f'{experiment_name}_{timestamp}')

        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)

        # Settings
        self.log_histograms = log_histograms
        self.log_param_changes = log_param_changes
        self.history_window = history_window

        # Rolling history for health checks
        self.loss_history = deque(maxlen=history_window)
        self.accuracy_history = deque(maxlen=history_window)
        self.grad_norm_history = deque(maxlen=history_window)
        self.clip_rate_history = deque(maxlen=history_window)
        self.logits_history = deque(maxlen=history_window)

        # SPM score tracking
        self.spm_win_scores = deque(maxlen=history_window)
        self.spm_lose_scores = deque(maxlen=history_window)
        self.spm_score_gaps = deque(maxlen=history_window)

        # Parameter change tracking
        self.prev_params: Optional[Dict[str, torch.Tensor]] = None

        # Anomaly counters
        self.nan_count = 0
        self.inf_count = 0
        self.gradient_explosion_count = 0

        # Thresholds for anomaly detection
        self.grad_norm_threshold = 100.0
        self.loss_spike_threshold = 10.0
        self.min_accuracy_threshold = 0.4
        self.max_clip_rate_threshold = 0.9

        print(f"[SPOTensorBoardLogger] Logging to: {log_dir}")

    def log_loss_metrics(
        self,
        metrics: Dict[str, float],
        global_step: int,
        prefix: str = 'train'
    ) -> None:
        """
        Log metrics from SPOLoss.

        Args:
            metrics: Dictionary from SPOLoss.forward()
            global_step: Current training step
            prefix: Prefix for metric names ('train' or 'val')
        """
        # Core loss metrics
        if 'loss' in metrics:
            self.writer.add_scalar(f'{prefix}/loss', metrics['loss'], global_step)
            self.loss_history.append(metrics['loss'])

        if 'dpo_loss' in metrics:
            self.writer.add_scalar(f'{prefix}/dpo_loss', metrics['dpo_loss'], global_step)

        if 'kl_loss' in metrics:
            self.writer.add_scalar(f'{prefix}/kl_loss', metrics['kl_loss'], global_step)
            self.writer.add_scalar(f'{prefix}/kl_win', metrics.get('kl_win', 0), global_step)
            self.writer.add_scalar(f'{prefix}/kl_lose', metrics.get('kl_lose', 0), global_step)

        # Probability ratio metrics
        self.writer.add_scalar(f'{prefix}/ratio/win_mean', metrics.get('ratio_win_mean', 1.0), global_step)
        self.writer.add_scalar(f'{prefix}/ratio/win_std', metrics.get('ratio_win_std', 0.0), global_step)
        self.writer.add_scalar(f'{prefix}/ratio/lose_mean', metrics.get('ratio_lose_mean', 1.0), global_step)
        self.writer.add_scalar(f'{prefix}/ratio/lose_std', metrics.get('ratio_lose_std', 0.0), global_step)

        # Clipping metrics (important for training stability)
        win_clip = metrics.get('ratio_win_clip_rate', 0.0)
        lose_clip = metrics.get('ratio_lose_clip_rate', 0.0)
        avg_clip = (win_clip + lose_clip) / 2

        self.writer.add_scalar(f'{prefix}/clip/win_rate', win_clip, global_step)
        self.writer.add_scalar(f'{prefix}/clip/lose_rate', lose_clip, global_step)
        self.writer.add_scalar(f'{prefix}/clip/avg_rate', avg_clip, global_step)
        self.clip_rate_history.append(avg_clip)

        # DPO logits (key indicator of preference learning)
        if 'logits_mean' in metrics:
            self.writer.add_scalar(f'{prefix}/logits/mean', metrics['logits_mean'], global_step)
            self.writer.add_scalar(f'{prefix}/logits/std', metrics.get('logits_std', 0.0), global_step)
            self.logits_history.append(metrics['logits_mean'])

        # Accuracy (most important metric for preference learning)
        if 'accuracy' in metrics:
            self.writer.add_scalar(f'{prefix}/accuracy', metrics['accuracy'], global_step)
            self.accuracy_history.append(metrics['accuracy'])

        # Log probability metrics
        if 'log_prob_win_mean' in metrics:
            self.writer.add_scalar(f'{prefix}/log_prob/win_mean', metrics['log_prob_win_mean'], global_step)
            self.writer.add_scalar(f'{prefix}/log_prob/lose_mean', metrics.get('log_prob_lose_mean', 0.0), global_step)
            self.writer.add_scalar(f'{prefix}/log_prob/diff_mean', metrics.get('log_prob_diff_mean', 0.0), global_step)

        # Multi-step specific metrics
        if 'num_pairs' in metrics:
            self.writer.add_scalar(f'{prefix}/num_pairs', metrics['num_pairs'], global_step)

        # Check for NaN/Inf
        for key, value in metrics.items():
            if math.isnan(value):
                self.nan_count += 1
                print(f"[WARNING] NaN detected in {key} at step {global_step}")
            if math.isinf(value):
                self.inf_count += 1
                print(f"[WARNING] Inf detected in {key} at step {global_step}")

    def log_gradients(
        self,
        model: nn.Module,
        global_step: int,
        log_per_layer: bool = False
    ) -> Dict[str, float]:
        """
        Log gradient statistics.

        Args:
            model: The model being trained
            global_step: Current training step
            log_per_layer: Whether to log per-layer gradient stats

        Returns:
            Dictionary of gradient statistics
        """
        total_norm = 0.0
        max_grad = 0.0
        grad_count = 0
        all_grads = []

        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()

                # Compute statistics
                param_norm = grad.norm(2).item()
                total_norm += param_norm ** 2
                max_grad = max(max_grad, grad.abs().max().item())
                grad_count += 1

                if log_per_layer:
                    layer_norms[name] = param_norm

                if self.log_histograms:
                    all_grads.append(grad.flatten())

        total_norm = math.sqrt(total_norm)
        avg_norm = total_norm / max(grad_count, 1)

        # Log global gradient stats
        self.writer.add_scalar('gradient/total_norm', total_norm, global_step)
        self.writer.add_scalar('gradient/max', max_grad, global_step)
        self.writer.add_scalar('gradient/avg_norm', avg_norm, global_step)

        self.grad_norm_history.append(total_norm)

        # Log histogram of all gradients
        if self.log_histograms and all_grads:
            all_grads_tensor = torch.cat(all_grads)
            self.writer.add_histogram('gradient/distribution', all_grads_tensor, global_step)

        # Log per-layer norms
        if log_per_layer and layer_norms:
            for name, norm in layer_norms.items():
                clean_name = name.replace('.', '/')
                self.writer.add_scalar(f'gradient/layers/{clean_name}', norm, global_step)

        # Check for gradient explosion
        if total_norm > self.grad_norm_threshold:
            self.gradient_explosion_count += 1
            print(f"[WARNING] Gradient explosion detected: norm={total_norm:.2f} at step {global_step}")

        return {
            'total_norm': total_norm,
            'max_grad': max_grad,
            'avg_norm': avg_norm,
            'grad_count': grad_count
        }

    def log_spm_scores(
        self,
        win_scores: torch.Tensor,
        lose_scores: torch.Tensor,
        all_scores: Optional[torch.Tensor] = None,
        global_step: int = 0,
        timestep: Optional[int] = None
    ) -> None:
        """
        Log SPM score statistics.

        Args:
            win_scores: Scores of selected winning candidates [B]
            lose_scores: Scores of selected losing candidates [B]
            all_scores: All candidate scores [num_candidates, B] (optional)
            global_step: Current training step
            timestep: Diffusion timestep (for timestep-specific logging)
        """
        prefix = 'spm'
        if timestep is not None:
            prefix = f'spm/t{timestep}'

        # Convert to numpy for statistics
        win_np = win_scores.detach().cpu().numpy()
        lose_np = lose_scores.detach().cpu().numpy()

        # Basic statistics
        win_mean = float(np.mean(win_np))
        lose_mean = float(np.mean(lose_np))
        score_gap = win_mean - lose_mean

        self.writer.add_scalar(f'{prefix}/win_score_mean', win_mean, global_step)
        self.writer.add_scalar(f'{prefix}/lose_score_mean', lose_mean, global_step)
        self.writer.add_scalar(f'{prefix}/score_gap', score_gap, global_step)
        self.writer.add_scalar(f'{prefix}/win_score_std', float(np.std(win_np)), global_step)
        self.writer.add_scalar(f'{prefix}/lose_score_std', float(np.std(lose_np)), global_step)

        # Track history
        self.spm_win_scores.append(win_mean)
        self.spm_lose_scores.append(lose_mean)
        self.spm_score_gaps.append(score_gap)

        # Log histograms
        if self.log_histograms:
            self.writer.add_histogram(f'{prefix}/win_scores', win_np, global_step)
            self.writer.add_histogram(f'{prefix}/lose_scores', lose_np, global_step)

        # All candidates statistics
        if all_scores is not None:
            all_np = all_scores.detach().cpu().numpy()
            self.writer.add_scalar(f'{prefix}/all_mean', float(np.mean(all_np)), global_step)
            self.writer.add_scalar(f'{prefix}/all_std', float(np.std(all_np)), global_step)
            self.writer.add_scalar(f'{prefix}/all_max', float(np.max(all_np)), global_step)
            self.writer.add_scalar(f'{prefix}/all_min', float(np.min(all_np)), global_step)

            if self.log_histograms:
                self.writer.add_histogram(f'{prefix}/all_scores', all_np.flatten(), global_step)

    def log_candidate_selection(
        self,
        decisions: List[Dict],
        global_step: int
    ) -> None:
        """
        Log candidate selection statistics.

        Args:
            decisions: List of decision dictionaries from sample_trajectory_multi
            global_step: Current training step
        """
        if not decisions:
            return

        # Selection diversity (how often different candidates are selected)
        selected_indices = [d['selected_idx'] for d in decisions]
        unique_selections = len(set(selected_indices))
        total_selections = len(selected_indices)
        selection_diversity = unique_selections / max(total_selections, 1)

        self.writer.add_scalar('selection/diversity', selection_diversity, global_step)
        self.writer.add_scalar('selection/num_decisions', total_selections, global_step)

        # Score statistics across decisions
        all_scores = []
        score_gaps = []

        for d in decisions:
            scores = d.get('scores', None)
            if scores is not None:
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                all_scores.extend(scores.flatten().tolist())
                score_gaps.append(float(np.max(scores) - np.min(scores)))

        if all_scores:
            self.writer.add_scalar('selection/avg_score_gap', np.mean(score_gaps), global_step)
            self.writer.add_scalar('selection/max_score_gap', np.max(score_gaps), global_step)

            if self.log_histograms:
                self.writer.add_histogram('selection/score_gaps', np.array(score_gaps), global_step)

        # Timestep distribution of selections
        timesteps = [d['timestep'] for d in decisions]
        if timesteps and self.log_histograms:
            self.writer.add_histogram('selection/timesteps', np.array(timesteps), global_step)

    def log_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        global_step: int
    ) -> None:
        """Log learning rate from optimizer."""
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            if i == 0:
                self.writer.add_scalar('lr/main', lr, global_step)
            else:
                self.writer.add_scalar(f'lr/group_{i}', lr, global_step)

    def log_parameter_changes(
        self,
        model: nn.Module,
        global_step: int
    ) -> Optional[Dict[str, float]]:
        """
        Track how much parameters have changed.

        Args:
            model: The model being trained
            global_step: Current training step

        Returns:
            Dictionary of parameter change statistics
        """
        if not self.log_param_changes:
            return None

        current_params = {}
        for name, param in model.named_parameters():
            current_params[name] = param.detach().clone()

        if self.prev_params is None:
            self.prev_params = current_params
            return None

        # Compute changes
        total_change = 0.0
        max_change = 0.0
        layer_changes = {}

        for name, current in current_params.items():
            if name in self.prev_params:
                prev = self.prev_params[name]
                diff = (current - prev).norm().item()
                total_change += diff ** 2
                max_change = max(max_change, diff)
                layer_changes[name] = diff

        total_change = math.sqrt(total_change)

        self.writer.add_scalar('param_change/total', total_change, global_step)
        self.writer.add_scalar('param_change/max', max_change, global_step)

        # Update previous params
        self.prev_params = current_params

        return {
            'total_change': total_change,
            'max_change': max_change
        }

    def log_timestep_metrics(
        self,
        timestep: int,
        metrics: Dict[str, float],
        global_step: int
    ) -> None:
        """
        Log metrics specific to a diffusion timestep.

        Args:
            timestep: Diffusion timestep
            metrics: Dictionary of metrics
            global_step: Current training step
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f'timestep/{timestep}/{key}', value, global_step)

    def log_reference_model_divergence(
        self,
        log_prob_current: torch.Tensor,
        log_prob_ref: torch.Tensor,
        global_step: int
    ) -> None:
        """
        Log divergence between current and reference model.

        Args:
            log_prob_current: Log probs under current model
            log_prob_ref: Log probs under reference model
            global_step: Current training step
        """
        diff = (log_prob_current - log_prob_ref).detach()

        self.writer.add_scalar('divergence/mean', diff.mean().item(), global_step)
        self.writer.add_scalar('divergence/std', diff.std().item(), global_step)
        self.writer.add_scalar('divergence/max', diff.max().item(), global_step)
        self.writer.add_scalar('divergence/min', diff.min().item(), global_step)

        # KL approximation
        kl_approx = diff.mean().item()
        self.writer.add_scalar('divergence/kl_approx', kl_approx, global_step)

        if self.log_histograms:
            self.writer.add_histogram('divergence/distribution', diff.cpu().numpy(), global_step)

    def check_training_health(self) -> Tuple[bool, List[str]]:
        """
        Check if training is progressing healthily.

        Returns:
            is_healthy: Boolean indicating if training is healthy
            issues: List of detected issues
        """
        issues = []

        # Check for NaN/Inf
        if self.nan_count > 0:
            issues.append(f"NaN values detected ({self.nan_count} occurrences)")
        if self.inf_count > 0:
            issues.append(f"Inf values detected ({self.inf_count} occurrences)")

        # Check gradient explosions
        if self.gradient_explosion_count > 5:
            issues.append(f"Frequent gradient explosions ({self.gradient_explosion_count} occurrences)")

        # Check accuracy trend
        if len(self.accuracy_history) >= 20:
            recent_acc = np.mean(list(self.accuracy_history)[-20:])
            if recent_acc < self.min_accuracy_threshold:
                issues.append(f"Low accuracy: {recent_acc:.3f} (threshold: {self.min_accuracy_threshold})")

        # Check clip rate
        if len(self.clip_rate_history) >= 20:
            recent_clip = np.mean(list(self.clip_rate_history)[-20:])
            if recent_clip > self.max_clip_rate_threshold:
                issues.append(f"High clip rate: {recent_clip:.3f} (threshold: {self.max_clip_rate_threshold})")

        # Check loss trend
        if len(self.loss_history) >= 50:
            early_loss = np.mean(list(self.loss_history)[:25])
            recent_loss = np.mean(list(self.loss_history)[-25:])

            # Loss should generally decrease or stay stable
            if recent_loss > early_loss * 1.5:
                issues.append(f"Loss increasing: early={early_loss:.4f}, recent={recent_loss:.4f}")

        # Check logits (should be positive on average for good preferences)
        if len(self.logits_history) >= 20:
            recent_logits = np.mean(list(self.logits_history)[-20:])
            if recent_logits < -1.0:
                issues.append(f"Negative logits trend: {recent_logits:.3f} (preferences might be reversed)")

        # Check SPM score gap
        if len(self.spm_score_gaps) >= 20:
            recent_gap = np.mean(list(self.spm_score_gaps)[-20:])
            if recent_gap < 0.01:
                issues.append(f"Small SPM score gap: {recent_gap:.4f} (win/lose pairs may be too similar)")

        is_healthy = len(issues) == 0
        return is_healthy, issues

    def log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log end-of-epoch summary.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics for the epoch
            val_metrics: Optional validation metrics
        """
        # Log epoch-level metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'epoch/train/{key}', value, epoch)

        if val_metrics:
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'epoch/val/{key}', value, epoch)

        # Log health check
        is_healthy, issues = self.check_training_health()
        self.writer.add_scalar('epoch/is_healthy', float(is_healthy), epoch)

        if issues:
            print(f"\n[Epoch {epoch}] Training Health Issues:")
            for issue in issues:
                print(f"  - {issue}")

    def log_custom_scalar(
        self,
        tag: str,
        value: float,
        global_step: int
    ) -> None:
        """Log a custom scalar value."""
        self.writer.add_scalar(tag, value, global_step)

    def log_custom_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.ndarray],
        global_step: int
    ) -> None:
        """Log a custom histogram."""
        if self.log_histograms:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            self.writer.add_histogram(tag, values, global_step)

    def log_text(
        self,
        tag: str,
        text: str,
        global_step: int
    ) -> None:
        """Log text for notes/debugging."""
        self.writer.add_text(tag, text, global_step)

    def log_hparams(
        self,
        hparams: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> None:
        """
        Log hyperparameters and final metrics.

        Args:
            hparams: Dictionary of hyperparameters
            metrics: Final metrics for this run
        """
        # Convert non-scalar hparams to strings
        cleaned_hparams = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                cleaned_hparams[k] = v
            else:
                cleaned_hparams[k] = str(v)

        self.writer.add_hparams(cleaned_hparams, metrics)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics from training history.

        Returns:
            Dictionary of summary statistics
        """
        summary = {}

        if self.loss_history:
            summary['loss_mean'] = np.mean(list(self.loss_history))
            summary['loss_std'] = np.std(list(self.loss_history))
            summary['loss_final'] = list(self.loss_history)[-1]

        if self.accuracy_history:
            summary['accuracy_mean'] = np.mean(list(self.accuracy_history))
            summary['accuracy_final'] = list(self.accuracy_history)[-1]

        if self.grad_norm_history:
            summary['grad_norm_mean'] = np.mean(list(self.grad_norm_history))
            summary['grad_norm_max'] = np.max(list(self.grad_norm_history))

        if self.spm_score_gaps:
            summary['spm_gap_mean'] = np.mean(list(self.spm_score_gaps))

        summary['nan_count'] = self.nan_count
        summary['inf_count'] = self.inf_count
        summary['gradient_explosion_count'] = self.gradient_explosion_count

        return summary

    def flush(self) -> None:
        """Flush all pending writes to disk."""
        self.writer.flush()

    def close(self) -> None:
        """Close the logger."""
        self.flush()
        self.writer.close()
        print(f"[SPOTensorBoardLogger] Closed. Logs saved to: {self.log_dir}")


class TrainingHealthMonitor:
    """
    Real-time training health monitor with automatic alerts.

    This class monitors training metrics and can trigger callbacks
    when anomalies are detected.
    """

    def __init__(
        self,
        logger: SPOTensorBoardLogger,
        check_interval: int = 100,
        alert_callback: Optional[callable] = None
    ):
        """
        Args:
            logger: SPOTensorBoardLogger instance
            check_interval: How often to check health (in steps)
            alert_callback: Function to call when issues detected
        """
        self.logger = logger
        self.check_interval = check_interval
        self.alert_callback = alert_callback
        self.last_check_step = 0

    def maybe_check(self, global_step: int) -> Optional[List[str]]:
        """
        Check training health if interval has passed.

        Returns:
            List of issues if any, None if not checked
        """
        if global_step - self.last_check_step < self.check_interval:
            return None

        self.last_check_step = global_step
        is_healthy, issues = self.logger.check_training_health()

        if not is_healthy and self.alert_callback:
            self.alert_callback(global_step, issues)

        return issues if not is_healthy else None


def create_spo_logger(
    experiment_name: str,
    base_dir: str = 'runs',
    **kwargs
) -> SPOTensorBoardLogger:
    """
    Factory function to create SPO logger with standard settings.

    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for logs
        **kwargs: Additional arguments to SPOTensorBoardLogger

    Returns:
        Configured SPOTensorBoardLogger instance
    """
    return SPOTensorBoardLogger(
        log_dir=base_dir,
        experiment_name=experiment_name,
        **kwargs
    )
