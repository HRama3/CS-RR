import torch
from torch import autograd, nn, Tensor

__all__ = ['compute_grad_penalty']


def compute_grad_penalty(discriminator: nn.Module, interpolated: Tensor) -> Tensor:
    identity = torch.ones(interpolated.shape[0], device=interpolated.device)
    gradients = autograd.grad(outputs=discriminator.forward(interpolated), inputs=interpolated, grad_outputs=identity)[0]

    gradients = gradients.view((interpolated.shape[0], -1))
    gradient_penalty = torch.pow(torch.norm(gradients, dim=1) - 1.0, 2.0)

    return gradient_penalty
