import torch


class SALP(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        base_optimizer,
        rho=0.05,
        rho_min=0.01,
        rho_max=1.0,
        rho_lr=1.0,
        adaptive=False,
        **kwargs
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(
            rho=rho,
            adaptive=adaptive,
            rho_min=rho_min,
            rho_max=rho_max,
            rho_lr=rho_lr,
            **kwargs
        )
        super(SALP, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.rho_lr = rho_lr
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.g_0_norm = None
        self.g_1_loss = None

        # Initialize per-parameter rho tensors
        for group in self.param_groups:
            init_rho = group.get("rho", rho)
            for p in group["params"]:
                state = self.state[p]
                if "rho" not in state:
                    state["rho"] = torch.full_like(p, init_rho)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # Compute ||g_0||
        self.g_0_norm = self._grad_norm()

        for group in self.param_groups:
            adaptive = group.get("adaptive", False)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["old_p"] = p.data.clone()
                state["g_0"] = p.grad.data.clone()
                rho_t = state["rho"]

                scale = 1.0 / (self.g_0_norm + 1e-12)
                weight = torch.pow(p, 2) if adaptive else 1.0
                e_w = weight * p.grad * (rho_t * scale)

                p.add_(e_w)  # climb to local max w + e(w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False, g1_loss=None):
        # Optionally set/update g_1_loss (scalar tensor on the right device)
        if g1_loss is not None:
            if not torch.is_tensor(g1_loss):
                # make it a tensor on the same device/dtype as parameters
                device = self.param_groups[0]["params"][0].device
                dtype = self.param_groups[0]["params"][0].dtype
                self.g_1_loss = torch.tensor(g1_loss, device=device, dtype=dtype)
            else:
                self.g_1_loss = g1_loss.detach()
        if self.g_1_loss is None:
            # Fallback if not provided; this keeps code running but disables rho learning this step
            self.g_1_loss = torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)

        g0_norm = (self.g_0_norm if self.g_0_norm is not None else torch.tensor(1.0, device=self.param_groups[0]["params"][0].device))
        g0_norm = g0_norm + 1e-12  # numeric stability

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                # grads at perturbed point (g_1)
                g_1 = p.grad.data
                g_0 = state.get("g_0", None)
                if g_0 is not None:
                    h_0 = g_1 - g_0
                    rho_t = state["rho"]

                    # rho_g = (g1 * g0 / ||g0||^2) - (h0 * g0) * (g1_loss / ||g0||^3)
                    rho_g = (g_1 * g_0) / (g0_norm ** 2) - (h_0 * g_0) * (self.g_1_loss / (g0_norm ** 3))

                    rho_t.add_(rho_g, alpha=self.rho_lr)
                    rho_t.clamp_(self.rho_min, self.rho_max)

                # restore to original weights
                p.data = state["old_p"]

        # actual optimizer step at w
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        # Optional SAM-style step with closure. Expects closure to:
        # - do a full forward/backward and return the loss tensor.
        assert closure is not None, "SALP requires a closure for step(), or use first_step/second_step explicitly."

        # First pass: compute g_0 and perturb
        loss_origin = closure()        # forward at w
        loss_origin.backward()         # compute g_0
        self.first_step(zero_grad=True)

        # Second pass: compute g_1 at w + e(w), update rho, and step
        loss_perturbed = closure()     # forward at w + e(w)
        loss_perturbed.backward()      # compute g_1
        self.second_step(zero_grad=True, g1_loss=loss_perturbed.detach())

        return loss_origin, loss_perturbed

    def _grad_norm(self):
        # Same as SAM: global L2 norm over all parameter gradients
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            adaptive = group.get("adaptive", False)
            for p in group["params"]:
                if p.grad is None:
                    continue
                weight = torch.abs(p) if adaptive else 1.0
                norms.append((weight * p.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    # --------- Utilities for inspecting learned rho ---------
    @torch.no_grad()
    def get_rho(self, flatten=False):
        rhos = []
        for group in self.param_groups:
            for p in group["params"]:
                if "rho" in self.state[p]:
                    rp = self.state[p]["rho"]
                    rhos.append(rp.clone().flatten() if flatten else rp.clone())
        return rhos

    @torch.no_grad()
    def get_rho_stats(self):
        rhos = self.get_rho(flatten=True)
        if not rhos:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        all_rho = torch.cat(rhos, dim=0)
        return {
            "mean": all_rho.mean().item(),
            "std": all_rho.std(unbiased=False).item(),
            "min": all_rho.min().item(),
            "max": all_rho.max().item(),
        }

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
