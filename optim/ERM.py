import logging
from torch.optim import Adam, SGD

class ERM:
    """Basic Adam optimizer (minimal implementation).

    Supports per-parameter groups and optional weight decay. This mirrors the
    update logic used in the project's optimizers (Helen/SAM/SALP).
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0, base_optim='adam', **kwargs):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        logging.info(f'Using ERM optimizer with lr={lr} weight_decay={weight_decay}')
        if base_optim == 'adam':
            self.optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


# Alias to satisfy dynamic import pattern in optim/__init__.py
erm = ERM


