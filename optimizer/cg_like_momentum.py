import torch
from .optimizer import Optimizer, required
from .gamma import lr_fn_dict


class CGLikeMomentum(Optimizer):
    r"""Implements Conjugate Gradient-like stochastic gradient descent with momentum.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(self, params, alpha_type=required, beta_type=required, gamma_type=required,
                 dampening=0, weight_decay=0, nesterov=False) -> None:
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(alpha_type=alpha_type, beta_type=beta_type, gamma_type=gamma_type,
                        dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and dampening != 0:
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CGLikeMomentum, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CGLikeMomentum, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']
            alpha_fn = lr_fn_dict[group['alpha_type']]
            beta_fn = lr_fn_dict[group['beta_type']]
            gamma_fn = lr_fn_dict[group['gamma_type']]

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                param_state = self.state[p]
                if 'n' in param_state:
                    param_state['n'] += 1
                else:
                    param_state['n'] = 1
                n = param_state['n']
                if 'd_buffer' in param_state:
                    d_buf = param_state['d_buffer']
                    gamma = gamma_fn(n)
                    d_p.add_(d_buf, alpha=-gamma)
                param_state['d_buffer'] = torch.clone(d_p).detach()

                beta = beta_fn(n)
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(beta).add_(d_p, alpha=1 - dampening)
                if nesterov:
                    d_p = d_p.add(buf, alpha=beta)
                else:
                    d_p = buf
                alpha = alpha_fn(n)
                p.add_(d_p, alpha=-alpha)

        return loss
