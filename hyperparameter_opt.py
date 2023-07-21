import torch
import numpy as np
import copy


class EtaOpt(object):

    def __init__(self, **kwargs):
        self.hyper_lr = kwargs['hyper_lr']
        self.T = kwargs['n_steps_T']
        self.n_steps = kwargs['n_steps']
        self.gamma = kwargs['gamma']
        self.dt = kwargs['dt']
        self.loss_constant_size = kwargs['loss_constant_size']
        self.eta_upper_bound = kwargs['eta_upper_bound']
        self.eta_lower_bound = kwargs['eta_lower_bound']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.time_span = torch.arange(0, self.n_steps, device=self.device, dtype=self.dtype)*self.dt
        self._init_params()

    def _init_params(self):
        self.eta = torch.ones(self.n_steps, device=self.device, dtype=self.dtype, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.eta], lr=self.hyper_lr)
        self.init_overlap_rho = 0.0
        self.init_magnitude_Q = 1.0

    def _bound_eta(self, eta):
        eta.data.clamp_(min=self.eta_lower_bound, max=self.eta_upper_bound)
        return eta

    def compute_ode(self):
        self.roh_t = []
        self.Q_t = []
        current_rho, current_Q = torch.tensor(self.init_overlap_rho, device=self.device, dtype=self.dtype), \
            torch.tensor(self.init_magnitude_Q, device=self.device, dtype=self.dtype)
        for t_index, t in enumerate(self.time_span):
            self.roh_t.append(current_rho)
            self.Q_t.append(current_Q)
            current_rho, current_Q = self.ode_single_step(t, current_rho, current_Q, t_index)
        self.roh_t = torch.stack(self.roh_t, dim=0)
        self.Q_t = torch.stack(self.Q_t, dim=0)
        return self.roh_t, self.Q_t

    def ode_single_step(self, t, current_rho, current_Q, t_index = None):
        current_eta = self.eta[t_index]

        d_roh = current_eta/torch.sqrt(2 * np.pi * current_Q) * (1 - current_rho**2)\
                * torch.pow(1 - 1/(np.pi)*torch.arccos(current_rho), self.T - 1)\
                - current_eta**2/(2*self.T*current_Q) * current_rho * \
                torch.pow(1 - 1/(np.pi)*torch.arccos(current_rho), self.T)

        d_Q = current_eta * torch.sqrt(2*current_Q/np.pi) * (1+current_rho) \
              * torch.pow(1 - 1/(np.pi)*torch.arccos(current_rho), self.T - 1) \
              + current_eta**2/(self.T) * torch.pow(1 - 1/(np.pi)*torch.arccos(current_rho), self.T)
        return d_roh, d_Q

    def get_current_loss(self):
        rho, Q = self.compute_ode()
        gamma_t = self.gamma**self.time_span
        loss = torch.sum(rho * gamma_t)*self.dt
        """loss = self.n_steps
        for index, ro in enumerate(rho):
            if 1 - rho < 1e-5:
                loss = index
                break"""
        return loss

    def hyper_update(self):
        loss = -self.get_current_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.eta = self._bound_eta(self.eta)
        return loss

    def hyper_cost(self):
        return 0


if __name__ == '__main__':
    hyper_lr = 0.1
    n_steps_T = 10
    n_steps = 5000
    gamma = 1.0
    dt = 1/100
    loss_constant_size = 1.0
    eta_upper_bound = 5
    eta_lower_bound = 0
    eta_opt = EtaOpt(hyper_lr=hyper_lr, n_steps_T=n_steps_T, n_steps=n_steps, gamma=gamma, dt=dt,
                     loss_constant_size=loss_constant_size, eta_upper_bound=eta_upper_bound,
                     eta_lower_bound=eta_lower_bound)
    eta_opt.hyper_update()

    iters = 100
    print(eta_opt.get_current_loss())
    for i in range(iters):
        print(eta_opt.hyper_update())