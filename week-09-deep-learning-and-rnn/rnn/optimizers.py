"""
optimizers.py
Optimizers for training RNN with gradient clipping.
"""

import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate

    def clip_gradients(self, grad_clip_val):
        pass


class OptimizerSGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def update_params_layer(self, params_dict):
        for name, (param, grad) in params_dict.items():
            if self.momentum > 0:
                if name not in self.velocity:
                    self.velocity[name] = np.zeros_like(param)
                self.velocity[name] = (
                    self.momentum * self.velocity[name]
                    - self.current_learning_rate * grad
                )
                param += self.velocity[name]
            else:
                param -= self.current_learning_rate * grad

    def clip_gradients(self, grad_clip_val, params_dict):
        total_norm = 0.0
        for name, (param, grad) in params_dict.items():
            grad_norm = np.sum(grad**2)
            total_norm += grad_norm
        total_norm = np.sqrt(total_norm)

        clip_coef = grad_clip_val / (total_norm + 1e-6)
        if clip_coef < 1:
            for name, (param, grad) in params_dict.items():
                grad[:] *= clip_coef


class OptimizerAdam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0
        self.m = {}
        self.v = {}

    def update_params_layer(self, params_dict):
        self.iterations += 1

        for name, (param, grad) in params_dict.items():
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)

            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad**2)

            m_corrected = self.m[name] / (1 - self.beta1**self.iterations)
            v_corrected = self.v[name] / (1 - self.beta2**self.iterations)

            param -= (
                self.current_learning_rate
                * m_corrected
                / (np.sqrt(v_corrected) + self.epsilon)
            )

    def clip_gradients(self, grad_clip_val, params_dict):
        total_norm = 0.0
        for name, (param, grad) in params_dict.items():
            grad_norm = np.sum(grad**2)
            total_norm += grad_norm
        total_norm = np.sqrt(total_norm)

        clip_coef = grad_clip_val / (total_norm + 1e-6)
        if clip_coef < 1:
            for name, (param, grad) in params_dict.items():
                grad[:] *= clip_coef
