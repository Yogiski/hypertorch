import torch
from torch import Tensor


def lorentz_norm(x: Tensor):
    return (-x[..., :1]**2 + (x[..., 1:]**2)).sum(dim=-1, keepdim=True)


def lortenz_normalize(x: Tensor, eps: float = 1e-5):
    norm = lorentz_norm(x)
    return x / torch.sqrt(torch.clamp(norm.abs()), min=eps)


def lorentz_dot(x: Tensor, y: Tensor) -> Tensor:
    x_o = -(x[..., :1] * y[..., :1])
    x_s = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    return x_o + x_s


def lortenz_distance(x: Tensor, y: Tensor, eps: float = 1e-5):
    dot = -lorentz_dot(x, y)
    dot = torch.clamp(dot, min=1.0 + eps)
    return torch.arcosh(dot)


def lorentz_to_klein(x: Tensor, eps: float = 1e-8):
    return x[..., :1] / (x[..., 1:] + eps)


def klein_to_lorentz(x: Tensor, eps: float = 1e-8):
    norm_sqr = torch.sum(x**2, dim=-1, keepdim=True)
    denom = torch.sqrt(1.0 - norm_sqr + eps)
    x_o = 1 / denom
    x_s = x / denom
    return torch.cat([x_o, x_s], dim=-1)
