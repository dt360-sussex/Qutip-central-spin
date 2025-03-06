import matplotlib.pyplot as plt
from qutip import *
import numpy as np
import random
import time
from scipy.constants import pi
import dask
from dask.distributed import Client, progress

Ix = 0.5*sigmax()
Iy = 0.5*sigmay()
Iz = 0.5*sigmaz()

def define_electron_spin_operators(N):
    Sx = tensor(0.5*sigmax(), qeye([2]*(N)))
    Sy = tensor(0.5*sigmay(), qeye([2]*(N)))
    Sz = tensor(0.5*sigmaz(), qeye([2]*(N)))
    return Sx,Sy,Sz

def RF_pulse_envelope(tlist,t_max, t_mean=10e-6, t_width=4e-6):
    # return 1e5*np.exp(-((tlist-t_mean)/t_width)**2)
    return (1-np.cos(2*pi*tlist/(t_max)))

def bath_op(spin_index, op,N):
    """
    Creates a tensor product operator for a single nuclear spin in the bath.

    Parameters:
        spin_index (int): Index of the nuclear spin to apply the operator to.
        op (Qobj): QuTiP operator to apply to the specified spin.

    Returns:
        Qobj: Tensor product operator.
    """
    return tensor([qeye(2)] + [op if i == spin_index else qeye(2) for i in range(N)])

def define_observables(N):
    """
    Define observables to track during the simulation.
    
    Parameters:
        N (int): Number of nuclear spins in the bath.
    
    Returns:
        list: List of observables to track.
    """
    Sx,Sy,Sz=define_electron_spin_operators(N)
    # Define observables to track
    observables = [
        Sx,
        Sy,
        Sz,  # Electron spin z-component
        sum(bath_op(i, Ix, N) for i in range(N)),
        sum(bath_op(i, Iy, N) for i in range(N)),
        sum(bath_op(i, Iz, N) for i in range(N))  
    ]

    return observables

def define_initial_state(N, initial_electron_spin=0, initial_nuclear_spin=0, random_nuclear_spins=True, random_seed=22):
    """
    Define the initial state of the system.
    
    Parameters: 
        N (int): Number of nuclear spins in the bath.
    
    Returns: Qobj: Initial state of the system.
    """

    electron_initial = basis(2, initial_electron_spin)  # Spin up state
    
    if random_nuclear_spins:    
        random.seed(random_seed)
        nuclear_initial = [basis(2, random.randint(0, 1)) for _ in range(N)]  # Random nuclear spins up/down
    else:
        nuclear_initial = [basis(2, initial_nuclear_spin) for _ in range(N)]  # All nuclear spins up

    return tensor([electron_initial] + nuclear_initial)

def define_hamiltonian(nu_RF_Amplitude,A, nu_e, nu_n, N,nu_RF, tlist, t_max):
    """
    Define the Hamiltonian for the system as a QobjEvo.
    
    Parameters:
        A (float): Hyperfine coupling strength.
        B (float): External magnetic field.
        gamma_e (float): Electron gyromagnetic ratio.
        gamma_n (float): Nuclear gyromagnetic ratio.
        N (int): Number of nuclear spins in the bath.
        tlist (array): Time array for simulation. (Currently unused, but can be used for time-dependent Hamiltonians)
    """
    Sx,Sy,Sz=define_electron_spin_operators(N)
    # Zeeman interactions for electron spin
    H_zeeman_e =2*pi * nu_e * Sz

    # Zeeman interactions for nuclear spins
    H_zeeman_n = 0
    for i in range(N): 
        H_zeeman_n -= 2 * pi *nu_n * bath_op(i, Iz, N)
    
    # Hyperfine interactions between electron and nuclear spins
    H_hyperfine = 0
    for i in range(N):
        H_hyperfine += (2*pi)*A*(
            Sx * bath_op(i, Ix, N) +
            Sy * bath_op(i, Iy, N) +
            Sz * bath_op(i, Iz, N)
        )

    # H_dipole = 0
    # # Dipole-dipole interactions between nuclear spins
    # for i in range(N):
    #     for j in range(i+1, N):
    #         H_dipole += A/100 * (
    #             bath_op(i, sigmax(), N) * bath_op(j, sigmax()) +
    #             bath_op(i, sigmay(), N) * bath_op(j, sigmay()) +
    #             bath_op(i, sigmaz(), N) * bath_op(j, sigmaz())
    #         )

    # RF interactions
    H_RF_x = 0
    for i in range(N):
        H_RF_x += (bath_op(i, Ix, N))

    H_RF_y = 0
    for i in range(N):
        H_RF_y += (bath_op(i, Iy, N))

    # Define the Hamiltonian as a time dependent Quantum Object, QobjEvo. 
    # QobjEvo is defined as a list of quantum objects (which must all be the same dimensions), tuples must be of the form (Qobj, list of coefficients)
    H = QobjEvo(
        [H_zeeman_e + H_zeeman_n + H_hyperfine, 
         [H_RF_x, 2*pi*nu_RF_Amplitude*RF_pulse_envelope(tlist, t_max)*np.cos(2*pi*nu_RF*tlist)], 
         [H_RF_y, -2*pi*nu_RF_Amplitude*RF_pulse_envelope(tlist, t_max)*np.sin(2*pi*nu_RF*tlist)]],
        tlist=tlist
    )

    return  H

def define_collapse_operators(N, gamma_relax_e=0, gamma_dephase_e=0, gamma_relax_n=0, gamma_dephase_n=0):
    """
    Define collapse operators for the system.
    
    Parameters:
        N (int): Number of nuclear spins in the bath.
        gamma_relax_e (float): Electron relaxation rate.
        gamma_dephase_e (float): Electron Dephasing rate.
        gamma_relax_n (float): Nuclear relaxation rate.
        gamma_dephase_n (float): Nuclear Dephasing rate.
    
    Returns:
        list: List of collapse operators.
    """
    Sx,Sy,Sz=define_electron_spin_operators(N)
    # Define relaxation and dephasing collapse operators for the electron spin
    c_ops = []
    c_ops.append(np.sqrt(gamma_relax_e) * Sz)
    c_ops.append(np.sqrt(gamma_dephase_e) * Sx)
    c_ops.append(np.sqrt(gamma_dephase_e) * Sy)
    c_ops.append(np.sqrt(gamma_dephase_e) * Sz)
    
    # Define relaxation collapse operators for the nuclear spins
    for i in range(N):
        c_ops.append(np.sqrt(gamma_relax_n) * bath_op(i, Iz, N))
        c_ops.append(np.sqrt(gamma_dephase_n) * bath_op(i, Ix, N))
        c_ops.append(np.sqrt(gamma_dephase_n) * bath_op(i, Iy, N))
        c_ops.append(np.sqrt(gamma_dephase_n) * bath_op(i, Iz, N))
    return c_ops

def define_params_list(base_params, num_params, param_names, param_ranges):
    """
    Define the parameters for each simulation, sweeping over a specified number of parameters.

    Parameters:
        num_params (int): Number of parameters to sweep over (0, 1, or 2).
        param_names (list): List of parameter names to sweep over. Must match the number of parameters.
        param_ranges (list): List of tuples, where each tuple defines the range (start, stop, num) for a parameter.
    
    Returns:
        list: List of dictionaries, where each dictionary contains a set of parameters for a simulation.
    """
    
    params_list = []

    if num_params == 0:
        return [base_params]
    
    elif num_params == 1:
        for swept_param in np.linspace(param_ranges[0][0], param_ranges[0][1], param_ranges[0][2]):
            params = base_params.copy()
            params[param_names[0]] = swept_param
            params_list.append(params)
    
    elif num_params == 2:
        for swept_param1 in np.linspace(param_ranges[0][0], param_ranges[0][1], param_ranges[0][2]):
            for swept_param2 in np.linspace(param_ranges[1][0], param_ranges[1][1], param_ranges[1][2]):
                params = base_params.copy()
                params[param_names[0]] = swept_param1
                params[param_names[1]] = swept_param2
                params_list.append(params)

    return params_list

def calculate_expectation_values(params, tlist, observables, unitary_evolution=True):
    """
    Calculate the expectation values of observables for a given set of parameters

    Parameters:
        params (dict): Dictionary of parameters for the simulation.
        tlist (array): Time array for simulation.
        observables (list): List of observables to track.    
    
    Returns:
        tuple: Tuple of expectation values for electron and nuclear spin components.
    """

    # Unpack parameters from dictionary
    nu_RF_Amplitude = params['nu_RF_Amplitude']
    A = params['A']
    nu_e = params['nu_e']
    nu_n = params['nu_n']
    nu_RF = params['nu_RF']
    N = params['N']
    psi0 = params['psi0']
    t_max = params['t_max']

    n_traj = params['n_traj']
    gamma_dephase_e = params['gamma_dephase_e']
    gamma_relax_e = params['gamma_relax_e']
    gamma_dephase_n = params['gamma_dephase_n']
    gamma_relax_n = params['gamma_relax_n']
    # Print swept parameter values
    swept_params = params['swept_params']

    print(tlist)
    print(f'{time.strftime("%H:%M:%S")}: Calculating expectation values for', end='')
    for param in swept_params:
        print(f'       {param} = {params[param]}', end='')
    print()


    # Call the time dependent hamiltonian with the given parameters
    H = define_hamiltonian(nu_RF_Amplitude, A, nu_e, nu_n, N, nu_RF, tlist, t_max)

    collapse_operators=define_collapse_operators(N, gamma_relax_e, gamma_dephase_e, gamma_relax_n, gamma_dephase_n)

    if unitary_evolution:
        output = sesolve(H, psi0, tlist, observables, options={'progress_bar': False, 'method': 'dop853'})
    else:
        output = mcsolve(H, psi0, tlist, collapse_operators, observables,  n_traj, options={'progress_bar': False, 'method': 'dop853'})

    return params, output.expect[0], output.expect[1], output.expect[2], output.expect[3], output.expect[4], output.expect[5]

