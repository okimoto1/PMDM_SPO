"""
Diffusion utilities for adding noise to ligand coordinates.

Adapted from PMDM's diffusion process to simulate noisy intermediate states
during molecule generation.
"""

import math
import numpy as np
import torch


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    """
    Get beta schedule for diffusion process.

    Args:
        beta_schedule: Type of schedule ('linear', 'cosine', 'quad', etc.)
        beta_start: Starting beta value
        beta_end: Ending beta value
        num_diffusion_timesteps: Number of timesteps

    Returns:
        betas: [T] numpy array of beta values
    """
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(beta_schedule)

    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function.

    Args:
        num_diffusion_timesteps: Number of betas to produce
        alpha_bar: Function that takes t from 0 to 1 and produces cumulative product
        max_beta: Maximum beta value

    Returns:
        betas: [T] numpy array
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class DiffusionNoiseScheduler:
    """
    Noise scheduler for ligand diffusion process.

    Manages the noise schedule and provides utilities for adding noise
    to ligand coordinates and atom features.
    """

    def __init__(
        self,
        num_diffusion_timesteps=1000,
        beta_schedule="sigmoid",  # Default matches PMDM training config
        beta_start=1e-7,
        beta_end=2e-3,
    ):
        """
        Args:
            num_diffusion_timesteps: Total number of diffusion steps
            beta_schedule: Type of noise schedule
            beta_start: Starting beta value
            beta_end: Ending beta value
        """
        self.num_diffusion_timesteps = num_diffusion_timesteps

        # Get beta schedule
        betas = get_beta_schedule(
            beta_schedule, beta_start, beta_end, num_diffusion_timesteps
        )

        # Compute alphas
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        # Convert to torch tensors
        self.betas = torch.from_numpy(betas).float()
        self.alphas = torch.from_numpy(alphas).float()
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod).float()

    def to(self, device):
        """Move tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def add_noise_to_pos(self, pos, noise, timesteps, batch):
        """
        Add noise to ligand coordinates following DDPM forward process.

        Following PMDM's implementation:
        pos_noisy = pos + noise * sqrt((1 - alpha_t) / alpha_t)

        Args:
            pos: [N, 3] original coordinates
            noise: [N, 3] sampled Gaussian noise
            timesteps: [B] timestep indices
            batch: [N] batch assignment

        Returns:
            pos_noisy: [N, 3] noised coordinates
        """
        # Get alpha values for each timestep
        alpha_t = self.alphas_cumprod[timesteps]  # [B]

        # Broadcast to atoms
        alpha_t_per_atom = alpha_t[batch].unsqueeze(-1)  # [N, 1]

        # Add noise: pos_noisy = pos + noise * sqrt((1-a)/a)
        pos_noisy = pos + noise * torch.sqrt((1.0 - alpha_t_per_atom) / alpha_t_per_atom)

        return pos_noisy

    def add_noise_to_atom_features(self, atom_features, noise, timesteps, batch):
        """
        Add noise to atom type features.

        Following PMDM's implementation:
        atom_noisy = sqrt(alpha_t) * atom + sqrt(1 - alpha_t) * noise

        Args:
            atom_features: [N, D] original atom features
            noise: [N, D] sampled Gaussian noise
            timesteps: [B] timestep indices
            batch: [N] batch assignment

        Returns:
            atom_features_noisy: [N, D] noised features
        """
        # Get alpha values
        alpha_t = self.alphas_cumprod[timesteps]  # [B]

        # Broadcast to atoms
        alpha_t_per_atom = alpha_t[batch].unsqueeze(-1)  # [N, 1]

        # Add noise
        atom_features_noisy = (
            torch.sqrt(alpha_t_per_atom) * atom_features +
            torch.sqrt(1.0 - alpha_t_per_atom) * noise
        )

        return atom_features_noisy

    def center_pos(self, pos, batch):
        """
        Center coordinates to center of mass for each molecule.

        Args:
            pos: [N, 3] coordinates
            batch: [N] batch indices

        Returns:
            pos_centered: [N, 3] centered coordinates
        """
        num_graphs = batch.max().item() + 1
        pos_centered = pos.clone()

        for i in range(num_graphs):
            mask = (batch == i)
            com = pos[mask].mean(dim=0)
            pos_centered[mask] = pos[mask] - com

        return pos_centered

    def add_noise(self, pos, atom_features, timesteps, batch, center=True):
        """
        Add noise to both coordinates and atom features.

        Args:
            pos: [N, 3] coordinates
            atom_features: [N, D] atom features
            timesteps: [B] timestep indices
            batch: [N] batch assignment
            center: Whether to center coordinates before and after adding noise

        Returns:
            pos_noisy: [N, 3] noised coordinates
            atom_features_noisy: [N, D] noised features
        """
        # Center to COM if requested
        if center:
            pos = self.center_pos(pos, batch)

        # Sample noise
        pos_noise = torch.randn_like(pos)
        atom_noise = torch.randn_like(atom_features)

        # Add noise
        pos_noisy = self.add_noise_to_pos(pos, pos_noise, timesteps, batch)
        atom_features_noisy = self.add_noise_to_atom_features(
            atom_features, atom_noise, timesteps, batch
        )

        # Re-center after adding noise
        if center:
            pos_noisy = self.center_pos(pos_noisy, batch)

        return pos_noisy, atom_features_noisy
