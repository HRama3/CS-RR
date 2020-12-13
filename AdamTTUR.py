import math
import torch
from torch.optim import Adam
from typing import Iterator

# Code adapted from https://github.com/pytorch/pytorch/blob/1.6/torch/optim/adam.py


class AdamTTUR(Adam):
    def __init__(self, parameters: Iterator[torch.nn.parameter.Parameter], lr: float,
                 tau: float, memory: float, alpha: float = 1.0, **kwargs) -> None:
        if not 0.0 < tau <= 1.0:
            raise ValueError('Damping coefficient, {:f} should be between 0 exclusive and 1 inclusive'
                             .format(tau))
        if memory <= 0.0:
            raise ValueError('Memory parameter, {:f} should be should be positive'.format(memory))
        if alpha <= 0.0:
            raise ValueError('Alpha parameter for beta2, {:f} must be positive'.format(alpha))

        super(AdamTTUR, self).__init__(parameters, lr, **kwargs)

        self.alpha = alpha
        self.tau = tau
        self.memory = memory

    @torch.no_grad()
    def step(self):
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1
                beta_decay = (group['lr'] / ((state['step'] + 1) ** self.tau)) * self.memory
                beta1 = 1.0 - beta_decay
                beta2 = 1.0 - self.alpha * beta_decay

                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']

                if group['weight_decay'] != 0.0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
