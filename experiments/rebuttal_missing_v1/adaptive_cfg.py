"""Beta-CFG guidance rule adapted to DiT's existing three-channel CFG.

Reference: gmum/beta-CFG, latent_diffusion.py, BetaDDIMGeo.sample.
The original SD/DDIM backbone is replaced by the common DiT/DDPM protocol;
this is explicitly an adaptation, not a reproduction of published SD scores.
"""
import math
import torch


def beta_weight(progress, a=2.0, b=3.0):
    if not 0 <= progress <= 1 or a <= 1 or b <= 1:
        raise ValueError('Expected progress in [0,1] and beta shapes > 1.')
    if progress in (0, 1):
        return 0.0
    return math.exp((a - 1) * math.log(progress) + (b - 1) * math.log1p(-progress)
                    + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))


def guided_noise(conditional, unconditional, progress, scale=7.5, a=2., b=3., gamma=1.):
    delta = conditional - unconditional
    norm = torch.linalg.vector_norm(delta, dim=(1, 2, 3), keepdim=True)
    # Official code has no zero guard; zero difference must produce zero guidance.
    normalized = torch.where(norm > 0, delta / norm.clamp_min(torch.finfo(delta.dtype).tiny).pow(gamma),
                             torch.zeros_like(delta))
    return unconditional + scale * beta_weight(progress, a, b) * normalized


def make_forward(model, timestep_map, specification):
    # SpacedDiffusion passes ORIGINAL DiT timesteps to the model wrapper.
    progress = {int(t): (len(timestep_map) - 1 - i) / (len(timestep_map) - 1)
                for i, t in enumerate(timestep_map)}

    def forward(x, t, y, cfg_scale):
        half = x[:len(x) // 2]
        result = model.forward(torch.cat([half, half]), t, y)
        eps, rest = result[:, :3], result[:, 3:]
        conditional, unconditional = eps.chunk(2)
        guided = guided_noise(conditional, unconditional, progress[int(t[0].item())],
                              **specification)
        return torch.cat([torch.cat([guided, guided]), rest], dim=1)
    return forward
