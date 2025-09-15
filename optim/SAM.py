from collections import defaultdict
import torch, logging
from torch.optim import Adam, SGD

class SAM:
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0, base_optim='adam',
                       rho=0.05, adaptive=False, **kwargs):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        logging.info(f'Using SAM optimizer with lr={lr} weight_decay={weight_decay} rho={rho}')
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.state = defaultdict(dict)
        self.params = params
        self.rho = rho
        self.adaptive = adaptive
        if base_optim == 'adam':
            self.optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        scale = self.rho / grad_norm
        for p in self.params:
            if p.grad is None: continue
            self.state[p]["old_p"] = p.data.clone()
            scale *= torch.pow(p, 2) if self.adaptive else 1.0
            p.data.add_(p.grad.data * scale.to(p))
        if zero_grad: self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for p in self.params:
            p.data.copy_(self.state[p]["old_p"])
        self.optimizer.step()
        if zero_grad: self.optimizer.zero_grad()

    def _grad_norm(self):
        shared_device = self.params[0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                for p in self.params if p.grad is not None
            ]), p=2
        )
        return norm + 1e-12

    def zero_grad(self):
        self.optimizer.zero_grad()