from hyperparameter_opt import EtaOpt
import numpy as np
import os

def run(hyper_lr, n_steps_T, n_steps, gamma, dt, loss_constant_size, eta_upper_bound, eta_lower_bound, iters=100, path):
	eta_opt = EtaOpt(hyper_lr=hyper_lr, n_steps_T=n_steps_T, n_steps=n_steps, gamma=gamma, dt=dt,
					 loss_constant_size=loss_constant_size, eta_upper_bound=eta_upper_bound,
					 eta_lower_bound=eta_lower_bound)
	eta_opt.hyper_update()

	print(eta_opt.get_current_loss())
	for i in range(iters):
		print(eta_opt.hyper_update())

	file_path = os.path.join(path, 'etas.npz')
	np.save(eta_opt.eta, file_path)