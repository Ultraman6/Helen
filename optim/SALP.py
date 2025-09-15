import math, torch
from collections import defaultdict
from torch.optim import Adam, SGD

class SALP:
    def __init__(self, embed_params, net_params=None, lr=1e-3, momentum=0.9, weight_decay=0, base_optim='adam',
                       rho=0.05, rho_min=0.0, rho_max=1.0, rho_lr=1.0, net_pert=0, adaptive=False, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.embed_params = embed_params
        self.net_params = net_params
        self.params = embed_params + net_params
        self.state = defaultdict(dict)
        self.net_pert = net_pert
        self.rho = rho
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho_lr = rho_lr
        self.adaptive = adaptive
        if base_optim == 'adam':
            self.optimizer = Adam(self.params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = SGD(self.params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        for p in self.embed_params:
            self.state[p]["rho"] = torch.full_like(p, rho)
        if net_pert != 0:
            for p in self.net_params:
                self.state[p]["rho"] = torch.full_like(p, rho)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for p in self.embed_params:
            if p.grad is None: continue
            self.state[p]["old_p"] = p.data.clone()
            eps = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad.data * self.state[p]['rho'] / grad_norm
            p.data.add_(eps)
        if self.net_pert == 1:
            scale = self.rho / grad_norm
            for p in self.net_params:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.data.add_(e_w)
        elif self.net_pert == 2:
            for p in self.net_params:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad.data * self.state[p]['rho'] / grad_norm
                p.data.add_(e_w)
        else:
            for p in self.net_params:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()

        if zero_grad: self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for p in self.embed_params:
            if p.grad is None: continue
            p.data.copy_(self.state[p]["old_p"])
        for p in self.net_params:
            if p.grad is None: continue
            if self.net_pert:
                p.data.copy_(self.state[p]["old_p"])
            else:
                p.grad.data.copy_(self.state[p]["old_g"])
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
