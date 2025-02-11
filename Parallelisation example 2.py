import matplotlib.pyplot as plt
from qutip import *
import numpy as np
import random
import time

###############################################################################################################################################################################################################################

# Define parameters
N = 15  # Number of nuclear spins in the bath
gamma_e = 1.0  # Electron gyromagnetic ratio (units chosen arbitrarily)
gamma_n = 0.01  # Nuclear gyromagnetic ratio (typically << gamma_e)
B = 1  # External magnetic field in z-direction

# Range of hyperfine coupling strengths to explore
A_values = np.linspace(1, 2, 8)  # Example range
print(f"Simulating for A values: {A_values}")

t_max = 20  # Simulation time
n_steps = 1000  # Number of time steps

###############################################################################################################################################################################################################################

# define time array for simulation
tlist = np.linspace(0, t_max, n_steps)

# Define spin operators for central electron spin
Sx = tensor(sigmax(), qeye([2]*(N)))
Sy = tensor(sigmay(), qeye([2]*(N)))
Sz = tensor(sigmaz(), qeye([2]*(N)))
Sm = tensor(sigmam(), qeye([2]*(N)))  # Lowering operator
Sp = tensor(sigmap(), qeye([2]*(N)))  # Raising operator



def bath_op(spin_index, op):
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

    # Total nuclear spin operator
    total_nuclear_z = sum(bath_op(i, sigmaz()) for i in range(N))

    # Define observables to track
    observables = [
        Sz,  # Electron spin z-component
        total_nuclear_z  # Total nuclear spin z-component
    ]

    return observables

def define_initial_state(N):
    """
    Define the initial state of the system.
    
    Parameters: 
        N (int): Number of nuclear spins in the bath.
    
    Returns: Qobj: Initial state of the system.
    """

    electron_initial = basis(2, 1)  # Spin up state
    nuclear_initial = [basis(2, random.randint(0,1)) for _ in range(N)]  # Random nuclear spins up/down
    return tensor([electron_initial] + nuclear_initial)

def define_hamiltonian(A, B, gamma_e, gamma_n, N, tlist):
    """
    Define the Hamiltonian for the system.
    
    Parameters:
        A (float): Hyperfine coupling strength.
        B (float): External magnetic field.
        gamma_e (float): Electron gyromagnetic ratio.
        gamma_n (float): Nuclear gyromagnetic ratio.
        N (int): Number of nuclear spins in the bath.
        tlist (array): Time array for simulation. (Currently unused, but can be used for time-dependent Hamiltonians)
    """
    
    H_zeeman_e = -gamma_e * B * Sz

    # Zeeman interactions for nuclear spins
    H_zeeman_n = 0
    for i in range(N): 
        H_zeeman_n += -gamma_n * B * bath_op(i, sigmaz())
    
    # Hyperfine interactions between electron and nuclear spins
    H_hyperfine = 0
    for i in range(N):
        H_hyperfine += A * (
            Sx * bath_op(i, sigmax()) +
            Sy * bath_op(i, sigmay()) +
            Sz * bath_op(i, sigmaz())
        )
    
    return H_zeeman_e + H_zeeman_n + H_hyperfine

def calculate_expectation_values(A, B, gamma_e, gamma_n, N, tlist, psi0, observables):
    """
    Calculate the expectation values of observables for a given set of parameters

    Parameters:
        A (float): Hyperfine coupling strength.
        B (float): External magnetic field.
        gamma_e (float): Electron gyromagnetic ratio.
        gamma_n (float): Nuclear gyromagnetic ratio.
        N (int): Number of nuclear spins in the bath.
        tlist (array): Time array for simulation.
        psi0 (Qobj): Initial state of the system.
        observables (list): List of observables to track.    
    
    Returns:
        tuple: Tuple of expectation values for electron and nuclear spin components.
    """

    H=define_hamiltonian(A,B,gamma_e,gamma_n,N, tlist)

    output = sesolve(H, psi0, tlist, observables, options={'progress_bar': False})

    # Extract expectation values
    electron_z = output.expect[0]
    nuclear_z = output.expect[1]
    return electron_z, nuclear_z



# Record start time
t0 = time.time()


psi0=define_initial_state(N)
observables=define_observables(N)

# Use loky_pmap to parallelize the simulations
results = loky_pmap(calculate_expectation_values, A_values, task_args=(B, gamma_e, gamma_n, N, tlist, psi0, observables))


# Record end time
print(f"Time taken: {time.time() - t0:.2f} seconds")
 
# Plotting the results
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

for i, A in enumerate(A_values):
    electron_z, nuclear_z = results[i]

    axes[0].plot(tlist, electron_z, label=f'A={A}')
    axes[1].plot(tlist, nuclear_z, label=f'A={A}')
    axes[2].plot(tlist, electron_z + nuclear_z, label=f'A={A}')

axes[0].set_xlabel('Time')
axes[0].set_ylabel('<Sz>')
axes[0].set_title('Electron Spin Z-Component')
axes[0].legend()

axes[1].set_xlabel('Time')
axes[1].set_ylabel('<Total Nuclear Sz>')
axes[1].set_title('Total Nuclear Spin Z-Component')
axes[1].legend()

axes[2].set_xlabel('Time')
axes[2].set_ylabel('<Sz>')
axes[2].set_title('Electron + Total Nuclear Spin Z-Component')
axes[2].legend()

plt.tight_layout()
plt.show()