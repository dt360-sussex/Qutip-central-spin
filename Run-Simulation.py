import matplotlib.pyplot as plt
from qutip import *
import numpy as np
import random
import time
import os
from scipy.constants import pi
import dask
from dask.distributed import Client, progress
from Simulation_library import *
import pickle 
###############################################################################################################################################################################################################################
# Define parameters
# Using units where hbar = 1, all parameters below are in Hz

N = 8 # Number of nuclear spins in the bath
A= 200e3    #Hyperfine coupling energy
nu_e = 15e6   # Electron Larmor frequency
nu_n = 2e6   # Nuclear Larmor frequency

nu_RF = 1.9e6   # RF carrier frequency
nu_RF_Amplitude=2.5e4   # RF amplitude

t_max = 20e-6  # Simulation time
n_steps = 5000  # Number of time steps

#Define the parameter sweeps for each simulation 
num_params = 1                   # Number of parameters to sweep over (0, 1, or 2)
param_names = ['nu_RF', 'nu_RF_Amplitude']     # Names of parameters to sweep over, will only sweep over the first num_params parameters
param_ranges = [(1.5e6, 2.5e6, 10), (2.4e4, 2.6e4, 20)]  # List of tuples, where each tuple defines the range (start, stop, num) for a parameter

# Decoherence parameters
unitary_evolution = True # Set to True to use sesolve for unitary evolution, False to use mcsolve for non-unitary evolution
n_traj = 1e2 # Number of trajectories for Monte Carlo simulation
gamma_relax_e       =   1e4 
gamme_dephase_e     =   1e4
gamma_relax_n       =   1e4
gamma_dephase_n     =   1e4

use_dask=False

###############################################################################################################################################################################################################################
# define time array for simulation
tlist = np.linspace(0, t_max, n_steps)
Sx,Sy,Sz=define_electron_spin_operators(N)

def main():
    t1=time.time()

    psi0=define_initial_state(N)
    observables=define_observables(N)

    base_params = {
        'nu_RF_Amplitude': nu_RF_Amplitude,
        'A': A,
        'nu_e': nu_e,
        'nu_n': nu_n,
        'nu_RF': nu_RF,
        'N': N,
        'psi0': psi0,
        't_max': t_max,
        'n_traj': n_traj,
        'gamma_dephase_e': gamme_dephase_e,
        'gamma_relax_e': gamma_relax_e,
        'gamma_dephase_n': gamma_dephase_n,
        'gamma_relax_n': gamma_relax_n,
        'swept_params': param_names
    }

    params_list = define_params_list(base_params, num_params, param_names, param_ranges)

    for i in range(num_params):
        print(f'Sweeping {param_names[i]} over range {param_ranges[i]}')

    print(f'{time.strftime("%H:%M:%S")}: Starting new run')

    if use_dask:
        # Start a local Dask client
        dask.config.set(scheduler='threads') 
        client = Client("tcp://192.168.1.101:8676", )

        # Use client.map to parallelize the computation
        futures = client.map(lambda params: calculate_expectation_values(params, tlist, observables, unitary_evolution), params_list)# Gather the results
        progress(futures)  # Display progress bar for futures

        results = client.gather(futures)
    else:
        # results = [calculate_expectation_values(params, tlist, observables, unitary_evolution) for params in params_list]
        results = loky_pmap(calculate_expectation_values, params_list,task_args=(tlist, observables), progress_bar=True)

    t2=time.time()
    print("Time taken: ", t2-t1)
    print("Time taken per simulation: ", (t2-t1)/len(params_list))
    timestamp = time.strftime("%d%m_%H%M")
    os.makedirs('Simulation Results', exist_ok=True)
    pickle.dump([results, tlist,base_params, num_params, param_names], open(f'Simulation Results/results_{timestamp}.pkl', 'wb'))
    print('Results saved to file:')
    print(f'results_{timestamp}.pkl')

if __name__ == '__main__':
    main()

#To do
#Dipole dipole interactions
#Vary hyperfine constants
#Explore more interesting initial states
#Iterate over many cycles of mcsolve where the electron spin is reset, in order to simulate optical pumping of the spins.