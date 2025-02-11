import matplotlib.pyplot as plt
from qutip import *
import numpy as np
import random
import time

print("Git test")
# Define parameters
N = 15  # Number of nuclear spins in the bath
gamma_e = 1.0  # Electron gyromagnetic ratio (units chosen arbitrarily)
gamma_n = 0.01  # Nuclear gyromagnetic ratio (typically << gamma_e)
B = 1  # External magnetic field in z-direction
# A = 5  # Hyperfine coupling strength, being iterated over currently.

t_max = 20  # Simulation time
n_steps = 1000  # Number of time steps

# Create time array for simulation
tlist = np.linspace(0, t_max, n_steps)

# Define spin operators for central electron spin
# Central spin operators (first in tensor product)
Sx = tensor(sigmax(), qeye([2]*(N)))
Sy = tensor(sigmay(), qeye([2]*(N)))
Sz = tensor(sigmaz(), qeye([2]*(N)))
Sm = tensor(sigmam(), qeye([2]*(N)))  # Lowering operator
Sp = tensor(sigmap(), qeye([2]*(N)))  # Raising operator


# Function to create bath spin operators
def bath_op(spin_index, op):
    return tensor([qeye(2)] + [op if i == spin_index else qeye(2) for i in range(N)])


# Define observables to track
# Total nuclear spin operator
total_nuclear_z = sum(bath_op(i, sigmaz()) for i in range(N))

observables = [
    Sz,  # Electron spin z-component
    total_nuclear_z  # Total nuclear spin z-component
]

# Set initial state: electron spin up, nuclear spins down
electron_initial = basis(2, 1)  # Spin up state
nuclear_initial = [basis(2, random.randint(0,1)) for _ in range(N)]  # Random nuclear spins up/down
psi0 = tensor([electron_initial] + nuclear_initial)


def calculate_expectation_values(A, B=B, gamma_e=gamma_e, gamma_n=gamma_n, N=N, tlist=tlist, psi0=psi0):
    # Create Hamiltonian components
    # Zeeman interaction for electron
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

    # Total Hamiltonian
    H = H_zeeman_e + H_zeeman_n + H_hyperfine 

    # Propagate the system in time
    output = sesolve(H, psi0, tlist, observables, options={'progress_bar': False})

    # Extract expectation values
    electron_z = output.expect[0]
    nuclear_z = output.expect[1]
    return electron_z, nuclear_z

# Range of hyperfine coupling strengths to explore
A_values = np.linspace(1, 2, 8)  # Example range

print(f"Simulating for A values: {A_values}")
print(f"Number of tasks: {len(A_values)}")
t0 = time.time()

# Use loky_pmap to parallelize the simulations
results = loky_pmap(calculate_expectation_values, A_values, task_args=(B, gamma_e, gamma_n, N, tlist, psi0))  
# Equivalent to result = [task(value, *task_args, **task_kwargs) for value in values]


print(f"Time taken: {time.time() - t0:.2f} seconds")
print(f"Time per task: {(time.time() - t0) / len(A_values):.2f} seconds")


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