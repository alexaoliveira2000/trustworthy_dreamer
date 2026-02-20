import torch
from torch.optim import Optimizer

class LaProp(Optimizer):
    def __init__(
        self,
        params,
        lr=4e-5,
        betas=(0.9, 0.99),
        eps=1e-20,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['nu'] = torch.zeros_like(p)  # RMS accumulator
                    state['mu'] = torch.zeros_like(p)  # momentum

                state['step'] += 1

                # ---- RMS normalization (like RMSprop) ----
                nu = state['nu']
                nu.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                nu_hat = nu / (1 - beta2 ** state['step'])
                grad_normed = grad / (nu_hat.sqrt() + eps)

                # ---- Momentum on normalized gradients ----
                mu = state['mu']
                mu.mul_(beta1).add_(grad_normed, alpha=1 - beta1)

                mu_hat = mu / (1 - beta1 ** state['step'])

                # ---- Parameter update ----
                p.add_(mu_hat, alpha=-lr)

        return loss
