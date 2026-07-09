"""
SPO (Step-aware Preference Optimization) Training Framework for PMDM

This module provides components for fine-tuning PMDM using DPO (Direct Preference Optimization)
with a trained Ligand SPM (Step-aware Preference Model).

Components:
    - log_prob_pmdm: Log probability calculation for PMDM diffusion steps
    - losses: DPO/SPO loss functions
    - reference_model: Reference model management
    - multi_sample_pmdm: Multi-candidate sampling utilities
    - spm_wrapper: SPM model wrapper for integration
    - tensorboard_logger: TensorBoard logging and training monitoring
"""

from .log_prob_pmdm import (
    compute_log_prob_pmdm,
    compute_log_prob_position,
    compute_log_prob_atom,
    gaussian_log_prob,
)
from .losses import SPOLoss, SPOLossWithKL, MultiStepSPOLoss
from .reference_model import ReferenceModelManager, MemoryEfficientReferenceManager
from .multi_sample_pmdm import MultiSamplePMDM, SamplingCandidate, sample_trajectory_multi
from .spm_wrapper import SPMWrapper, DummySPMWrapper, load_spm_model
from .tensorboard_logger import (
    SPOTensorBoardLogger,
    TrainingHealthMonitor,
    create_spo_logger,
)

__all__ = [
    # Log probability
    'compute_log_prob_pmdm',
    'compute_log_prob_position',
    'compute_log_prob_atom',
    'gaussian_log_prob',
    # Losses
    'SPOLoss',
    'SPOLossWithKL',
    'MultiStepSPOLoss',
    # Reference model
    'ReferenceModelManager',
    'MemoryEfficientReferenceManager',
    # Sampling
    'MultiSamplePMDM',
    'SamplingCandidate',
    'sample_trajectory_multi',
    # SPM
    'SPMWrapper',
    'DummySPMWrapper',
    'load_spm_model',
    # TensorBoard
    'SPOTensorBoardLogger',
    'TrainingHealthMonitor',
    'create_spo_logger',
]
