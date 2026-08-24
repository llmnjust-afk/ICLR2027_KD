"""
TAGS: Timestep-Adaptive Guidance Scheduling

Determines the guidance weight at each diffusion timestep, creating a smooth
schedule that varies the guidance strength across the denoising process.

Key insight:
  - High-noise timesteps (early denoising): Strong guidance to set global
    semantic structure and align with target modes
  - Mid-noise timesteps: Gradually decreasing guidance for smooth transition
  - Low-noise timesteps (late denoising): Weak/no guidance to let the diffusion
    prior refine fine details and image quality

This prevents the artifacts that arise from constant strong guidance (as in MGD3)
while maintaining the benefits of mode-guided generation.
"""

import numpy as np
import torch
import math


class TimestepAdaptiveSchedule:
    """
    Computes adaptive guidance weights for each diffusion timestep.

    The schedule ensures:
      1. Strong guidance at high-noise steps (structure formation)
      2. Smooth decay through mid-noise steps (transition)
      3. Minimal guidance at low-noise steps (detail refinement)
    """

    SCHEDULE_TYPES = ["cosine", "linear", "exponential", "step", "warmup_cosine", "adaptive"]

    def __init__(
        self,
        w_max=0.5,
        w_min=0.0,
        schedule_type="cosine",
        warmup_steps=0,
        decay_rate=3.0,
        n_steps=5,
    ):
        """
        Args:
            w_max: Maximum guidance weight (scaled by CAGS per class)
            w_min: Minimum guidance weight (typically 0)
            schedule_type: Type of schedule function
                - 'cosine': Cosine annealing from w_max to w_min
                - 'linear': Linear decay from w_max to w_min
                - 'exponential': Exponential decay
                - 'step': Step function (constant then drop)
                - 'warmup_cosine': Warmup then cosine decay
                - 'adaptive': Uses noise schedule to determine weight
            warmup_steps: Number of warmup steps for warmup_cosine schedule
            decay_rate: Decay rate for exponential schedule
            n_steps: Number of transition steps for step schedule
        """
        self.w_max = w_max
        self.w_min = w_min
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.n_steps = n_steps

    def get_weight(self, t, t_start, t_stop, w_max=None):
        """
        Compute guidance weight at timestep t.

        Args:
            t: Current timestep (0 = lowest noise, t_max = highest noise)
            t_start: Low-noise boundary of guidance window (IAST stop point)
            t_stop: High-noise boundary of guidance window (typically t_max)
            w_max: Maximum weight (from CAGS, overrides self.w_max if provided)

        Returns:
            weight: float in [w_min, w_max]
        """
        if w_max is None:
            w_max = self.w_max

        # No guidance outside the guidance window
        if t < t_start or t > t_stop:
            return self.w_min

        if t_stop == t_start:
            return w_max

        # Progress: 0 at high-noise end (t_stop, strongest guidance),
        #           1 at low-noise end (t_start, weakest guidance)
        progress = (t_stop - t) / (t_stop - t_start)

        if self.schedule_type == "cosine":
            weight = w_max * math.cos(math.pi / 2 * progress) + self.w_min

        elif self.schedule_type == "linear":
            weight = w_max * (1 - progress) + self.w_min

        elif self.schedule_type == "exponential":
            weight = w_max * math.exp(-self.decay_rate * progress) + self.w_min

        elif self.schedule_type == "step":
            if progress < (1 - 1.0 / self.n_steps):
                weight = w_max
            else:
                weight = self.w_min

        elif self.schedule_type == "warmup_cosine":
            if progress < self.warmup_steps / (t_stop - t_start):
                # Warmup phase: linear ramp up
                warmup_progress = progress * (t_stop - t_start) / self.warmup_steps
                weight = w_max * warmup_progress
            else:
                # Cosine decay phase
                decay_progress = (progress - self.warmup_steps / (t_stop - t_start)) / (
                    1 - self.warmup_steps / (t_stop - t_start)
                )
                weight = w_max * math.cos(math.pi / 2 * decay_progress) + self.w_min

        elif self.schedule_type == "adaptive":
            # Based on signal-to-noise ratio: stronger guidance when SNR is low
            # (early denoising), weaker when SNR is high (late denoising)
            # This mimics the natural information flow in diffusion models
            snr_approx = 1.0 / (1.0 + progress * 10)  # Approximate SNR
            weight = w_max * snr_approx + self.w_min

        else:
            weight = w_max

        return float(weight)

    def get_weight_tensor(self, t_tensor, t_start, t_stop, w_max=None, device="cuda"):
        """
        Compute guidance weight as a tensor for batch processing.

        Args:
            t_tensor: torch tensor of timesteps
            t_start, t_stop: guidance window boundaries
            w_max: maximum weight
            device: torch device

        Returns:
            weight_tensor: torch tensor of weights
        """
        if w_max is None:
            w_max = self.w_max

        weights = []
        for t_val in t_tensor:
            w = self.get_weight(t_val.item(), t_start, t_stop, w_max)
            weights.append(w)

        return torch.tensor(weights, device=device, dtype=torch.float32)

    def get_schedule_curve(self, t_start, t_stop, w_max=None, n_points=50):
        """
        Generate the full schedule curve for visualization.

        Args:
            t_start, t_stop: guidance window
            w_max: maximum weight
            n_points: number of points

        Returns:
            timesteps: numpy array
            weights: numpy array
        """
        if w_max is None:
            w_max = self.w_max

        timesteps = np.linspace(t_start, max(t_stop, 1), n_points)
        weights = np.array([
            self.get_weight(t, t_start, t_stop, w_max) for t in timesteps
        ])

        return timesteps, weights

    def compare_schedules(self, t_start, t_stop, w_max=None):
        """
        Compare all schedule types for ablation study.

        Returns:
            dict: {schedule_type: (timesteps, weights)}
        """
        if w_max is None:
            w_max = self.w_max

        results = {}
        original_type = self.schedule_type

        for sched_type in self.SCHEDULE_TYPES:
            self.schedule_type = sched_type
            results[sched_type] = self.get_schedule_curve(t_start, t_stop, w_max)

        self.schedule_type = original_type
        return results
