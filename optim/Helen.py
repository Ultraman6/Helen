import torch, logging
from collections import defaultdict
from torch.optim import Adam, SGD

class Helen:
    def __init__(self, embed_params, net_params, lr=1e-3, momentum=0.9, weight_decay=0, base_optim='adam',
                       rho=0.05, net_pert=True, bound=0.3, adaptive=False, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        logging.info(f'Using Helen optimizer with lr={lr} weight_decay={weight_decay}'
                     f'rho={rho} bound={bound} net_pert={net_pert}')
        self.state = defaultdict(dict)
        self.embed_params = embed_params
        self.net_params = net_params
        self.params = embed_params + net_params
        self.rho = rho
        self.adaptive = adaptive
        self.net_pert = net_pert
        self.bound = bound
        if base_optim == 'adam':
            self.optimizer = Adam(self.params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = SGD(self.params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for p in self.embed_params:
            if p.grad is None: continue
            self.state[p]["old_p"] = p.data.clone()
            g_norm = torch.norm(p.grad)
            scale = self.rho / g_norm
            unique_ids = self.state[p]["unique_ids"]
            unique_ids_count = self.state[p]["unique_ids_count"].float()
            freq_scale = torch.scatter(torch.zeros(p.shape[0], device=p.device), 0, unique_ids,
                                       unique_ids_count)
            freq_scale = torch.clamp(freq_scale / torch.max(freq_scale), self.bound)
            e_w = p.grad * scale.to(p)
            e_w = e_w * freq_scale.unsqueeze(1)
            p.data.add_(e_w)
        if self.net_pert:
            grad_norm = self._grad_norm()
            scale = self.rho / grad_norm
            for p in self.net_params:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
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

    def count_feature_occurrence(self, X, feature_params_map, feature_specs):
        """
        Count the occurrence of each feature in the batch
        add result to state of optimizer
        :param X: batch data
        :param feature_params_map: map from feature name to its parameters
        :param feature_specs: feature specs
        """
        X = X.long()
        for feature, feature_spec in feature_specs.items():
            feature_idx = feature_spec["index"]
            feature_field_batch = X[:, feature_idx]
            unique_features, feature_count = torch.unique(feature_field_batch, return_counts=True, sorted=True)

            feature_params = feature_params_map[feature]
            for param in feature_params:
                self.state[param]["unique_ids"] = unique_features
                self.state[param]["unique_ids_count"] = feature_count

    def zero_grad(self):
        self.optimizer.zero_grad()
