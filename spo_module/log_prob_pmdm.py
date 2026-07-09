"""
Log probability computation for PMDM diffusion steps.

This module provides functions to compute the log probability of transitions
in the PMDM diffusion process, which is essential for DPO/SPO loss computation.

IMPORTANT NOTES ABOUT PMDM SAMPLING:
====================================

1. PMDM is NOT standard DDIM:
   - PMDM uses a hybrid sampling strategy with min(LD_step_size, generalized_step_size)
   - eps_pos is a position gradient direction, NOT noise prediction
   - The update formula is: pos_next = pos_t - et * step_size_pos + noise * step_size_noise
     where et = -eps_pos

2. Center_pos operation:
   - After each denoising step, PMDM applies center_pos_pl() which centers
     the ligand positions around their center of mass
   - This is a deterministic transformation that doesn't change the log probability
     (Jacobian determinant = 1 for translation)
   - However, when computing log prob, we need to ensure mean and target are in
     the same coordinate frame

3. Log probability formula:
   For the transition x_t -> x_{t-1}:
   - Raw sampling: pos_raw = mean + noise * sigma
   - After center: pos_next = pos_raw - mean(pos_raw)

   The log prob is computed as:
   log p(x_{t-1}|x_t) = -||x_{t-1} - mean_centered||^2 / (2*sigma^2) - d/2 * log(2*pi*sigma^2)

   where mean_centered = mean - mean(pos_raw) ≈ mean - mean(mean) for small noise
   Since PMDM centers the target (pos_next), we should also center the predicted mean.
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Union


def center_pos_by_batch(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    Center positions by subtracting the center of mass for each batch.

    This replicates PMDM's center_pos_pl operation for the ligand.

    Args:
        pos: Positions [N, 3]
        batch: Batch indices [N]

    Returns:
        centered_pos: Centered positions [N, 3]
    """
    from torch_scatter import scatter_mean
    com = scatter_mean(pos, batch, dim=0)  # [B, 3]
    return pos - com[batch]


def gaussian_log_prob(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    reduce_dims: Tuple[int, ...] = (-1,)
) -> torch.Tensor:
    """
    Compute log probability under a Gaussian distribution.

    Args:
        x: Sample tensor
        mean: Mean of the Gaussian
        std: Standard deviation (must be positive)
        reduce_dims: Dimensions to sum over (default: last dim only)

    Returns:
        log_prob: Log probability, summed over reduce_dims
    """
    # Ensure std is positive
    std = std.clamp(min=1e-8)

    # Log probability: -0.5 * [(x-mean)^2/std^2 + log(2*pi*std^2)]
    var = std ** 2
    log_prob = -0.5 * (
        ((x - mean) ** 2) / var +
        torch.log(2 * math.pi * var)
    )

    # Sum over specified dimensions
    for dim in sorted(reduce_dims, reverse=True):
        log_prob = log_prob.sum(dim=dim)

    return log_prob


def compute_variance_ddim(
    alphas_cumprod: torch.Tensor,
    timestep: int,
    prev_timestep: int,
    eta: float = 1.0
) -> torch.Tensor:
    """
    Compute DDIM variance for the transition from timestep to prev_timestep.

    sigma = eta * sqrt((1-alpha_{t-1})/(1-alpha_t)) * sqrt(1-alpha_t/alpha_{t-1})

    Args:
        alphas_cumprod: Cumulative product of alphas [T]
        timestep: Current timestep t
        prev_timestep: Previous timestep t-1 (can be -1 for t=0)
        eta: DDIM eta parameter (0=deterministic, 1=DDPM-like)

    Returns:
        variance: Scalar variance tensor
    """
    alpha_t = alphas_cumprod[timestep]
    alpha_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)

    # Ensure tensors are on the same device
    if not isinstance(alpha_t_prev, torch.Tensor):
        alpha_t_prev = torch.tensor(alpha_t_prev, device=alpha_t.device, dtype=alpha_t.dtype)

    # Variance formula from DDIM paper
    # sigma^2 = eta^2 * (1-alpha_{t-1})/(1-alpha_t) * (1-alpha_t/alpha_{t-1})
    variance = (
        eta ** 2 *
        (1 - alpha_t_prev) / (1 - alpha_t).clamp(min=1e-8) *
        (1 - alpha_t / alpha_t_prev.clamp(min=1e-8))
    )

    return variance.clamp(min=1e-8)


def compute_log_prob_position(
    eps_pred: torch.Tensor,
    pos_t: torch.Tensor,
    pos_t_prev: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_t_prev: torch.Tensor,
    eta: float = 1.0,
    batch: Optional[torch.Tensor] = None,
    sampling_type: str = 'generalized',
    sigmas_t: Optional[torch.Tensor] = None,
    step_lr: float = 1e-6,
    apply_center: bool = True
) -> torch.Tensor:
    """
    Compute log probability for position transition in PMDM.

    IMPORTANT: PMDM's position prediction is NOT standard DDPM noise.
    In PMDM:
        - eps_pos is a position correction vector (gradient-like direction)
        - The network predicts: pos_next = pos_t - (-eps_pos) * step_size + noise * noise_scale
        - This is equivalent to: pos_next = pos_t + eps_pos * step_size + noise * noise_scale
        - For generalized sampling: et = -eps_pos, then pos_next = pos_t - et * step_size + noise

    For generalized/DDIM sampling (used in PMDM):
        et = -eps_pos
        c1 = eta * sqrt((1-alpha_t/alpha_{t-1}) * (1-alpha_{t-1}) / (1-alpha_t))
        c2 = sqrt((1-alpha_{t-1}) - c1^2)
        step_size_pos_gen = 3 * (sqrt(1-alpha_t)/sqrt(alpha_t) - c2/sqrt(alpha_{t-1}))
        step_size_noise_gen = 5 * c1 / sqrt(alpha_{t-1})
        step_size_pos_ld = step_lr * (sigma/0.01)^2 / sigma
        step_size_noise_ld = sqrt(step_lr * (sigma/0.01)^2 * 2)
        step_size_pos = min(step_size_pos_ld, step_size_pos_gen)
        step_size_noise = min(step_size_noise_ld, step_size_noise_gen)
        pos_next = pos_t - et * step_size_pos + noise * step_size_noise

    IMPORTANT: PMDM applies center_pos after sampling. If apply_center=True,
    we assume pos_t_prev is already centered and will center the predicted mean
    to match the coordinate frame.

    Args:
        eps_pred: Predicted position correction [N, 3] (PMDM's eps_pos)
        pos_t: Current positions [N, 3]
        pos_t_prev: Next positions [N, 3] (what we want probability of, may be centered)
        alpha_t: Alpha at timestep t (scalar or per-sample)
        alpha_t_prev: Alpha at timestep t-1 (scalar or per-sample)
        eta: DDIM eta parameter
        batch: Optional batch indices [N] for per-sample computation
        sampling_type: 'generalized', 'ddpm_noisy', or 'ld'
        sigmas_t: Sigma at timestep t (required for generalized sampling)
        step_lr: Step learning rate (for LD sampling)
        apply_center: If True, center the predicted mean to match PMDM's center_pos operation

    Returns:
        log_prob: Log probability [B] if batch provided, else [1]
    """
    # Expand alphas to match shape
    if alpha_t.dim() == 0:
        alpha_t = alpha_t.view(1, 1).expand(pos_t.size(0), 1)
    if alpha_t_prev.dim() == 0:
        alpha_t_prev = alpha_t_prev.view(1, 1).expand(pos_t.size(0), 1)

    if sampling_type == 'generalized':
        # PMDM generalized sampling (DDIM-like)
        # et = -eps_pos (the network output is negated for position update)
        et = -eps_pred

        # Compute sigma_t if not provided
        if sigmas_t is None:
            sigmas_t = torch.sqrt(1 - alpha_t) / torch.sqrt(alpha_t).clamp(min=1e-8)
        if sigmas_t.dim() == 0:
            sigmas_t = sigmas_t.view(1, 1).expand(pos_t.size(0), 1)

        # Compute DDIM coefficients
        c1 = eta * torch.sqrt(
            (1 - alpha_t / alpha_t_prev.clamp(min=1e-8)) *
            (1 - alpha_t_prev) /
            (1 - alpha_t).clamp(min=1e-8)
        )
        c2 = torch.sqrt((1 - alpha_t_prev - c1**2).clamp(min=0))

        # Compute LD step sizes
        step_size_pos_ld = step_lr * (sigmas_t / 0.01) ** 2 / sigmas_t.clamp(min=1e-8)
        step_size_noise_ld = torch.sqrt((step_lr * (sigmas_t / 0.01) ** 2) * 2)

        # Compute generalized step sizes
        step_size_pos_gen = 3 * (
            torch.sqrt(1 - alpha_t) / torch.sqrt(alpha_t).clamp(min=1e-8) -
            c2 / torch.sqrt(alpha_t_prev).clamp(min=1e-8)
        )
        step_size_noise_gen = 5 * c1 / torch.sqrt(alpha_t_prev).clamp(min=1e-8)

        # Take minimum of LD and generalized step sizes (following PMDM)
        step_size_pos = torch.minimum(step_size_pos_ld, step_size_pos_gen)
        step_size_noise = torch.minimum(step_size_noise_ld, step_size_noise_gen)

        # Compute mean: pos_next = pos_t - et * step_size_pos
        mean = pos_t - et * step_size_pos

        # Variance: sigma = step_size_noise
        sigma = step_size_noise.clamp(min=1e-8)

    elif sampling_type == 'ddpm_noisy':
        # DDPM-style sampling for positions
        beta_t = 1 - alpha_t / alpha_t_prev.clamp(min=1e-8)
        e = -eps_pred  # PMDM negates eps_pos

        # Mean: (pos_t - beta_t * e) / sqrt(1 - beta_t)
        mean = (pos_t - beta_t * e) / torch.sqrt((1 - beta_t).clamp(min=1e-8))

        # Variance: exp(0.5 * log(beta_t)) = sqrt(beta_t)
        sigma = torch.sqrt(beta_t.clamp(min=1e-8))

    elif sampling_type == 'ld':
        # Langevin dynamics sampling
        if sigmas_t is None:
            # Compute sigma from alpha
            sigmas_t = torch.sqrt(1 - alpha_t) / torch.sqrt(alpha_t).clamp(min=1e-8)

        if sigmas_t.dim() == 0:
            sigmas_t = sigmas_t.view(1, 1).expand(pos_t.size(0), 1)

        step_size = step_lr * (sigmas_t / 0.01) ** 2

        # Mean: pos_t + step_size * eps_pos / sigma
        mean = pos_t + step_size * eps_pred / sigmas_t.clamp(min=1e-8)

        # Variance: sqrt(step_size * 2)
        sigma = torch.sqrt(step_size * 2).clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown sampling type: {sampling_type}")

    # Apply center operation to mean if requested
    # PMDM applies center_pos_pl after sampling, which centers the ligand positions
    # around their center of mass. To compute correct log prob, we need to center
    # the predicted mean to match the coordinate frame of pos_t_prev.
    if apply_center and batch is not None:
        mean = center_pos_by_batch(mean, batch)

    # Compute log probability
    log_prob = gaussian_log_prob(pos_t_prev, mean, sigma, reduce_dims=(-1,))

    # Aggregate by batch if provided
    if batch is not None:
        # Sum log probs for each sample
        from torch_scatter import scatter_add
        log_prob = scatter_add(log_prob, batch, dim=0)
    else:
        log_prob = log_prob.sum(dim=0, keepdim=True)

    return log_prob


def compute_log_prob_atom(
    eps_pred: torch.Tensor,
    atom_t: torch.Tensor,
    atom_t_prev: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_t_prev: torch.Tensor,
    eta: float = 1.0,
    batch: Optional[torch.Tensor] = None,
    sampling_type: str = 'generalized',
    sigmas_t: Optional[torch.Tensor] = None,
    step_lr: float = 1e-6
) -> torch.Tensor:
    """
    Compute log probability for atom type transition in PMDM.

    IMPORTANT: In PMDM, atom type uses standard DDPM-style noise prediction:
        - eps_node predicts the noise added to atom features
        - For generalized sampling: eps_node is normalized by sqrt(1-alpha_t)
        - atom_next = atom_t - eps_node_normalized * step_size + noise * step_size_noise

    Args:
        eps_pred: Predicted noise [N, D] (PMDM's eps_node = node_score_global + node_score_local)
        atom_t: Current atom features [N, D]
        atom_t_prev: Next atom features [N, D]
        alpha_t: Alpha at timestep t
        alpha_t_prev: Alpha at timestep t-1
        eta: DDIM eta parameter
        batch: Optional batch indices [N]
        sampling_type: 'generalized', 'ddpm_noisy', or 'ld'
        sigmas_t: Sigma at timestep t (for LD sampling)
        step_lr: Step learning rate (for LD sampling)

    Returns:
        log_prob: Log probability [B] if batch provided, else [1]
    """
    # Expand alphas
    if alpha_t.dim() == 0:
        alpha_t = alpha_t.view(1, 1).expand(atom_t.size(0), 1)
    if alpha_t_prev.dim() == 0:
        alpha_t_prev = alpha_t_prev.view(1, 1).expand(atom_t.size(0), 1)

    if sampling_type == 'generalized':
        # PMDM generalized sampling for atom types
        # eps_node is normalized by sqrt(1-alpha_t) before use
        eps_normalized = eps_pred / torch.sqrt(1 - alpha_t).clamp(min=1e-8)

        # Compute sigma_t if not provided
        if sigmas_t is None:
            sigmas_t = torch.sqrt(1 - alpha_t) / torch.sqrt(alpha_t).clamp(min=1e-8)
        if sigmas_t.dim() == 0:
            sigmas_t = sigmas_t.view(1, 1).expand(atom_t.size(0), 1)

        # Compute DDIM coefficients (same as position)
        c1 = eta * torch.sqrt(
            (1 - alpha_t / alpha_t_prev.clamp(min=1e-8)) *
            (1 - alpha_t_prev) /
            (1 - alpha_t).clamp(min=1e-8)
        )
        c2 = torch.sqrt((1 - alpha_t_prev - c1**2).clamp(min=0))

        # Compute LD step sizes
        step_size_pos_ld = step_lr * (sigmas_t / 0.01) ** 2 / sigmas_t.clamp(min=1e-8)
        step_size_noise_ld = torch.sqrt((step_lr * (sigmas_t / 0.01) ** 2) * 2)

        # Compute generalized step sizes
        step_size_pos_gen = 3 * (
            torch.sqrt(1 - alpha_t) / torch.sqrt(alpha_t).clamp(min=1e-8) -
            c2 / torch.sqrt(alpha_t_prev).clamp(min=1e-8)
        )
        step_size_noise_gen = 5 * c1 / torch.sqrt(alpha_t_prev).clamp(min=1e-8)

        # Take minimum of LD and generalized step sizes (following PMDM)
        step_size_pos = torch.minimum(step_size_pos_ld, step_size_pos_gen)
        step_size_noise = torch.minimum(step_size_noise_ld, step_size_noise_gen)

        # Mean: atom_next = atom_t - eps_normalized * step_size_pos
        mean = atom_t - eps_normalized * step_size_pos

        # Variance
        sigma = step_size_noise.clamp(min=1e-8)

    elif sampling_type == 'ddpm_noisy':
        # DDPM-style sampling for atom types
        beta_t = 1 - alpha_t / alpha_t_prev.clamp(min=1e-8)
        e = eps_pred  # atom uses standard noise (NOT negated like position)

        # Predict x0 from noise
        node0_from_e = (
            torch.sqrt(1.0 / alpha_t) * atom_t -
            torch.sqrt(1.0 / alpha_t - 1) * e
        )

        # Mean
        mean_eps = (
            (torch.sqrt(alpha_t_prev) * beta_t) * node0_from_e +
            (torch.sqrt(1 - beta_t) * (1 - alpha_t_prev)) * atom_t
        ) / (1.0 - alpha_t)
        mean = mean_eps

        # Variance
        sigma = torch.sqrt(beta_t.clamp(min=1e-8))

    elif sampling_type == 'ld':
        # Langevin dynamics sampling
        if sigmas_t is None:
            sigmas_t = torch.sqrt(1 - alpha_t) / torch.sqrt(alpha_t).clamp(min=1e-8)

        if sigmas_t.dim() == 0:
            sigmas_t = sigmas_t.view(1, 1).expand(atom_t.size(0), 1)

        step_size = step_lr * (sigmas_t / 0.01) ** 2

        # Normalize eps_node
        eps_normalized = eps_pred / torch.sqrt(1 - alpha_t).clamp(min=1e-8)

        # Mean: atom_t - step_size * eps_normalized / sigma
        mean = atom_t - step_size * eps_normalized / sigmas_t.clamp(min=1e-8)

        # Variance
        sigma = torch.sqrt(step_size * 2).clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown sampling type: {sampling_type}")

    # Compute log probability (sum over feature dimension)
    log_prob = gaussian_log_prob(atom_t_prev, mean, sigma, reduce_dims=(-1,))

    # Aggregate by batch
    if batch is not None:
        from torch_scatter import scatter_add
        log_prob = scatter_add(log_prob, batch, dim=0)
    else:
        log_prob = log_prob.sum(dim=0, keepdim=True)

    return log_prob


def compute_log_prob_pmdm(
    model_output: Tuple[torch.Tensor, torch.Tensor],
    current_state: Tuple[torch.Tensor, torch.Tensor],
    next_state: Tuple[torch.Tensor, torch.Tensor],
    timestep: int,
    prev_timestep: int,
    alphas_cumprod: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    eta: float = 1.0,
    pos_weight: float = 1.0,
    atom_weight: float = 1.0,
    sampling_type: str = 'generalized',
    sigmas: Optional[torch.Tensor] = None,
    step_lr: float = 1e-6,
    apply_center: bool = True
) -> torch.Tensor:
    """
    Compute combined log probability for PMDM diffusion step.

    This is the main function for DPO loss computation.

    Args:
        model_output: Tuple of (eps_pos, eps_atom) from model
        current_state: Tuple of (pos_t, atom_t) at timestep t
        next_state: Tuple of (pos_t_prev, atom_t_prev) at timestep t-1
        timestep: Current timestep index
        prev_timestep: Previous timestep index (t-1)
        alphas_cumprod: Cumulative product of alphas
        batch: Optional batch indices
        eta: DDIM eta parameter
        pos_weight: Weight for position log prob
        atom_weight: Weight for atom type log prob
        sampling_type: 'generalized', 'ddpm_noisy', or 'ld'
        sigmas: Optional sigma values for LD sampling
        step_lr: Step learning rate for LD sampling
        apply_center: If True, center the predicted mean to match PMDM's center_pos operation

    Returns:
        log_prob: Combined log probability [B]
    """
    eps_pos, eps_atom = model_output
    pos_t, atom_t = current_state
    pos_t_prev, atom_t_prev = next_state

    # Get alpha values
    alpha_t = alphas_cumprod[timestep]
    alpha_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=alpha_t.device)

    # Get sigma at timestep if provided
    sigmas_t = sigmas[timestep] if sigmas is not None else None

    # Compute position log prob
    log_prob_pos = compute_log_prob_position(
        eps_pred=eps_pos,
        pos_t=pos_t,
        pos_t_prev=pos_t_prev,
        alpha_t=alpha_t,
        alpha_t_prev=alpha_t_prev,
        eta=eta,
        batch=batch,
        sampling_type=sampling_type,
        sigmas_t=sigmas_t,
        step_lr=step_lr,
        apply_center=apply_center
    )

    # Compute atom type log prob
    log_prob_atom = compute_log_prob_atom(
        eps_pred=eps_atom,
        atom_t=atom_t,
        atom_t_prev=atom_t_prev,
        alpha_t=alpha_t,
        alpha_t_prev=alpha_t_prev,
        eta=eta,
        batch=batch,
        sampling_type=sampling_type,
        sigmas_t=sigmas_t,
        step_lr=step_lr
    )

    # Combine with weights
    log_prob = pos_weight * log_prob_pos + atom_weight * log_prob_atom

    return log_prob


def compute_log_prob_from_model_step(
    model,
    ligand_pos_t: torch.Tensor,
    ligand_atom_t: torch.Tensor,
    ligand_pos_next: torch.Tensor,
    ligand_atom_next: torch.Tensor,
    ligand_batch: torch.Tensor,
    protein_ctx: torch.Tensor,
    protein_pos: torch.Tensor,
    protein_atom_type: torch.Tensor,
    protein_batch: torch.Tensor,
    timestep: int,
    prev_timestep: int,
    eta: float = 1.0,
    **model_kwargs
) -> torch.Tensor:
    """
    Convenience function that calls model and computes log prob in one step.

    This is useful for computing log probabilities during SPO training.

    Args:
        model: PMDM model instance
        ligand_pos_t: Current ligand positions [N, 3]
        ligand_atom_t: Current ligand atom features [N, D]
        ligand_pos_next: Target ligand positions [N, 3]
        ligand_atom_next: Target ligand atom features [N, D]
        ligand_batch: Batch indices for ligand [N]
        protein_ctx: Protein context embeddings
        protein_pos: Protein positions
        protein_atom_type: Protein atom features
        protein_batch: Batch indices for protein
        timestep: Current timestep
        prev_timestep: Target timestep
        eta: DDIM eta parameter
        **model_kwargs: Additional arguments for model

    Returns:
        log_prob: Log probability [B]
    """
    num_graphs = ligand_batch.max().item() + 1
    device = ligand_pos_t.device

    # Create timestep tensor
    t = torch.full((num_graphs,), timestep, dtype=torch.long, device=device)

    # Get model prediction
    net_out = model.net(
        ligand_atom_type=ligand_atom_t,
        ligand_pos=ligand_pos_t,
        ligand_batch=ligand_batch,
        protein_embeddings=protein_ctx,
        time_step=t,
        protein_atom_feature=protein_atom_type,
        protein_pos=protein_pos,
        protein_batch=protein_batch,
        return_edges=True,
        **model_kwargs
    )

    # Extract predictions
    # PMDM returns (pos_eq_global, pos_eq_local, node_score_global, node_score_local, ...)
    if model.vae_context:
        (pos_eq_global, pos_eq_local, node_score_global, node_score_local,
         edge_index, edge_type, edge_length, local_edge_mask) = net_out[:-1]
    else:
        (pos_eq_global, pos_eq_local, node_score_global, node_score_local,
         edge_index, edge_type, edge_length, local_edge_mask) = net_out

    # Combine global and local predictions (using default weights)
    eps_pos = pos_eq_global + pos_eq_local
    eps_atom = node_score_global + node_score_local

    # Compute log probability
    log_prob = compute_log_prob_pmdm(
        model_output=(eps_pos, eps_atom),
        current_state=(ligand_pos_t, ligand_atom_t),
        next_state=(ligand_pos_next, ligand_atom_next),
        timestep=timestep,
        prev_timestep=prev_timestep,
        alphas_cumprod=model.alphas,
        batch=ligand_batch,
        eta=eta
    )

    return log_prob
