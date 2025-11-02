import torch
from torch import nn
from torch import Tensor


class LorentzLinear(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, bias=True):
        super().__init__()
        # ignore time-like dimension
        self.weights = nn.Parameter(torch.randn(out_dim - 1, in_dim - 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_dim - 1))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:

        # spatial dims only
        x_s = (x[..., 1:] @ self.weights.T)
        if self.bias is not None:
            x_s = x_s + self.bias
        # compute new time dim
        x_o_new = torch.sqrt(1.0 + torch.sum(x_s**2, dim=-1, keepdim=True))

        return torch.cat([x_o_new, x_s], dim=-1)


class LorentzLinearParallel(nn.Module):

    def __init__(self, n_proj, in_dim, out_dim, bias=True):

        self.weights = nn.Parameter(
            torch.randn(n_proj, out_dim - 1, in_dim - 1)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(n_proj, out_dim - 1)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:

        x_s = x[..., 1:]
        x_s = torch.einsum("...i,pji->...pj", x_s, self.weights)

        if self.bias is not None:
            x_s = x_s + self.bias

        x_o_new = torch.sqrt(1.0 + torch.sum(x_s**2, dim=-1, keepdim=True))
        return torch.cat([x_o_new, x_s], dim=-1)
