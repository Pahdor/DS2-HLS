import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class MagnitudeAwarePCGrad:
    def __init__(self, optimizer, beta=0.9, eps=1e-10):
        """
        Args:
            optimizer: torch.optim.Optimizer
            beta: decay factor for EMA (0 < beta < 1)
            eps: small constant to avoid zero division
        """
        self.optimizer = optimizer
        self.beta = beta
        self.eps = eps
        # will hold EMA of gradient magnitudes per task
        self.ema_lambdas = None

    def zero_grad(self):
        return self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        return self.optimizer.step()

    def pc_backward(self, objectives):
        # pack gradients for each task
        grads, shapes, has_grads = self._pack_grad(objectives)
        # project and merge with EMA
        merged = self._project_with_ema(grads, has_grads)
        # unflatten and set back
        unflat = self._unflatten_grad(merged, shapes[0])
        self._set_grad(unflat)

    def _project_with_ema(self, grads, has_grads):
        device = grads[0].device
        num_tasks = len(grads)

        # compute norms for each task
        norms = torch.tensor([g.norm().item() for g in grads], device=device)
        # init or update ema_lambdas
        if self.ema_lambdas is None:
            self.ema_lambdas = norms.clone()
        else:
            self.ema_lambdas = self.beta * self.ema_lambdas + (1 - self.beta) * norms

        # normalize gradients
        normalized = []
        for g, n in zip(grads, norms):
            if n < self.eps:
                normalized.append(torch.zeros_like(g))
            else:
                normalized.append(g / (n + self.eps))

        # copy for projection
        proj_norms = copy.deepcopy(normalized)
        for grad_i in proj_norms:
            # shuffle other normalized grads
            others = normalized.copy()
            random.shuffle(others)
            for grad_j in others:
                dot = torch.dot(grad_i, grad_j)
                if dot < 0:
                    grad_i.sub_(dot * grad_j)

        # merge: scale each projected unit-vector by its EMA magnitude
        merged = torch.zeros_like(grads[0], device=device)
        for lam, g_pc in zip(self.ema_lambdas, proj_norms):
            merged.add_(lam * g_pc)

        return merged

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self.optimizer.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            g_list, shape_list, has_list = self._retrieve_grad()
            grads.append(self._flatten_grad(g_list, shape_list))
            has_grads.append(self._flatten_grad(has_list, shape_list))
            shapes.append(shape_list)
        return grads, shapes, has_grads

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    grad.append(torch.zeros_like(p))
                    has_grad.append(torch.zeros_like(p))
                    shape.append(p.shape)
                else:
                    grad.append(p.grad.clone())
                    has_grad.append(torch.ones_like(p))
                    shape.append(p.grad.shape)
        return grad, shape, has_grad

    def _flatten_grad(self, grad_list, shape_list):
        return torch.cat([g.flatten() for g in grad_list], dim=0)

    def _unflatten_grad(self, flat_grad, shapes):
        unflat, idx = [], 0
        for shape in shapes:
            length = int(np.prod(shape))
            chunk = flat_grad[idx: idx + length].view(shape).clone()
            unflat.append(chunk)
            idx += length
        return unflat

    def _set_grad(self, grads):
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1