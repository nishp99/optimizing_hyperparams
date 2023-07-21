import datetime
import os
from run_func import run
import submitit



#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)


#connect to results folder
results_run = os.path.join('results', run_timestamp)
os.makedirs(results_run, exist_ok=True)

#make data folder inside timestamp

"""#os.path.join(results, unique identifier)
training_path = os.path.join(data, "training_data")
os.makedirs(training_path, exist_ok = True)

evaluation_path = os.path.join(data, "evaluation_data")
os.makedirs(evaluation_path, exist_ok = True)"""

executor = submitit.AutoExecutor(folder=outputpath)
executor.update_parameters(timeout_min=1000, mem_gb=3, gpus_per_node=0, cpus_per_task=8, slurm_array_parallelism=1, slurm_partition="cpu")

jobs = []

hyper_lr = 0.1
n_steps_T = 10
n_steps = 5000
gamma = 1.0
dt = 1/100
loss_constant_size = 1.0
eta_upper_bound = 5
eta_lower_bound = 0



with executor.batch():
	job = executor.submit(run, path=results_run, hyper_lr=hyper_lr, n_steps_T=n_steps_T, n_steps=n_steps, gamma=gamma, dt=dt, loss_constant_size=loss_constant_size, eta_upper_bound=eta_upper_bound, eta_lower_bound=eta_lower_bound, path=results_run)
	#jobs.append(job)