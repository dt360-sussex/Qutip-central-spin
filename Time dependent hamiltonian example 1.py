import matplotlib.pyplot as plt
from qutip import *
import numpy as np
import random
import time

from scipy.constants import e, h, pi
from qutip import Bloch
Bohr_magneton = 9.2740100783e-24  # Bohr magneton in J/T
Nuclear_magneton = 5.0507837461e-27  # Nuclear magneton in J/T

###############################################################################################################################################################################################################################
# Define parameters
# Range of hyperfine coupling strengths to explore
RF_Amplitudes = np.linspace(1e6,1e8,20)  # Units are all messed up right now



N = 10  # Number of nuclear spins in the bath
A=200e3
nu_e = 15e6  # Electron larmor frequency
nu_n = 2e6  # Nuclear Larmor frequency
nuRF = 2e6   # RF carrier frequency



t_max = 20e-6  # Simulation time
n_steps = 10000  # Number of time steps

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

def RF_pulse_envelope(tlist, t_mean=10e-6, t_width=4e-6):
    # return 1e5*np.exp(-((tlist-t_mean)/t_width)**2)
    return (1-np.cos(2*pi*tlist/(20e-6)))


def define_observables(N):
    """
    Define observables to track during the simulation.
    
    Parameters:
        N (int): Number of nuclear spins in the bath.
    
    Returns:
        list: List of observables to track.
    """

    # Define observables to track
    observables = [
        Sx,
        Sy,
        Sz,  # Electron spin z-component
        sum(bath_op(i, sigmax()) for i in range(N)),
        sum(bath_op(i, sigmay()) for i in range(N)),
        sum(bath_op(i, sigmaz()) for i in range(N))  
    ]

    return observables

def define_initial_state(N, initial_electron_spin=0, random_seed=42, random_nuclear_spins=False):
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
        nuclear_initial = [basis(2, 0) for _ in range(N)]  # All nuclear spins up

    return tensor([electron_initial] + nuclear_initial)

def define_hamiltonian(RF_Amplitude,A, nu_e, nu_n, N,nuRf, tlist):
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

    # Zeeman interactions for electron spin
    H_zeeman_e = -nu_e * Sz

    # Zeeman interactions for nuclear spins
    H_zeeman_n = 0
    for i in range(N): 
        H_zeeman_n += -nu_n * bath_op(i, sigmaz())
    
    # Hyperfine interactions between electron and nuclear spins
    H_hyperfine = 0
    for i in range(N):
        H_hyperfine += A*(
            Sx * bath_op(i, sigmax()) +
            Sy * bath_op(i, sigmay()) +
            Sz * bath_op(i, sigmaz())
        )

    # RF interactions
    H_RF_x = 0
    for i in range(N):
        H_RF_x += (bath_op(i, sigmax()))

    H_RF_y = 0
    for i in range(N):
        H_RF_y += (bath_op(i, sigmay()))

    # Define the Hamiltonian as a time dependent Quantum Object, QobjEvo. 
    # QobjEvo is defined as a list of quantum objects (which must all be the same dimensions), tuples must be of the form (Qobj, list of coefficients)
    H = QobjEvo(
        [H_zeeman_e + H_zeeman_n + H_hyperfine, 
         [H_RF_x, RF_Amplitude*RF_pulse_envelope(tlist)*np.cos(2*pi*nuRf*tlist)], 
         [H_RF_y, -RF_Amplitude*RF_pulse_envelope(tlist)*np.sin(2*pi*nuRf*tlist)]],
        tlist=tlist
    )
    return  H
 
def calculate_expectation_values(RF_Amplitude, A, nu_e, nu_n, N, tlist, psi0, observables):
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

    # Call the time dependent hamiltonian with the given parameters
    H = define_hamiltonian(RF_Amplitude, A, nu_e, nu_n, N, nuRF, tlist)

    output = sesolve(H, psi0, tlist, observables, options={'progress_bar': False, 'nsteps': 10000, 'rtol': 1e-8, 'atol': 1e-8})

    # Extract expectation values
    return output.expect[0], output.expect[1], output.expect[2], output.expect[3], output.expect[4], output.expect[5]

# Record start time
t0 = time.time()

psi0=define_initial_state(N)
observables=define_observables(N)

# Use loky_pmap to parallelize the simulations
results = loky_pmap(calculate_expectation_values, RF_Amplitudes, task_args=(A,nu_e, nu_n, N, tlist, psi0, observables), progress_bar=True)

# results=[calculate_expectation_values(i, A, nu_e, nu_n, N, tlist, psi0, observables) for i in RF_Amplitudes]

# Record end time
print(f"Time taken: {time.time() - t0:.2f} seconds")

# # Create Bloch sphere visualization
# b = Bloch()

# # For each RF amplitude

# _, _, _, nuclear_x, nuclear_y, nuclear_z = results[1]
# nuclear_x = nuclear_x/N
# nuclear_y = nuclear_y/N
# nuclear_z = nuclear_z/N

# # Add points to Bloch sphere
# b.add_points([nuclear_x, nuclear_y, nuclear_z], meth='l')
    
# # Customize appearance
# b.point_color = ['r', 'g', 'b', 'y']  # Different colors for different amplitudes
# b.point_marker = ['o']
# b.point_size = [20]

# # Render the Bloch sphere
# b.render()
# plt.show()

# ###############################################################################################################################################################################################################################
# # Plotting the results
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

for i, A in enumerate(RF_Amplitudes):
    electron_x, electron_y, electron_z, nuclear_x, nuclear_y, nuclear_z = results[i]

    axes[1].plot(tlist, nuclear_x/N, color='red', linestyle='--')
    axes[1].plot(tlist, nuclear_y/N, color='green', linestyle=':')
    axes[1].plot(tlist, nuclear_z/N, color='blue')

    axes[2].plot(tlist, electron_x, color='red', linestyle='--')
    axes[2].plot(tlist, electron_y, color='green', linestyle=':')
    axes[2].plot(tlist, electron_z, color='blue')

axes[0].set_xlabel('Time')
axes[0].set_ylabel('RF intensity')
axes[0].set_title('RF')
axes[0].plot(tlist, RF_pulse_envelope(tlist,1)*np.cos(2*pi*nuRF*tlist), label='RF_x')

axes[1].set_xlabel('Time')
axes[1].set_ylabel('<Total Nuclear Sz>')
axes[1].set_title('Total Nuclear Spin Z-Component')
axes[1].legend()

axes[2].set_xlabel('Time')
axes[2].set_ylabel('<Sz>')
axes[2].set_title('Electron Spin Z-Component')
axes[2].legend()


plt.tight_layout()
plt.show()

