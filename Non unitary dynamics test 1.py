import matplotlib.pyplot as plt
from qutip import *
import numpy as np
import random
from scipy.constants import pi

###############################################################################################################################################################################################################################
# Define parameters
# Using units where hbar = 1, all parameters below are in Hz

N = 6  # Number of nuclear spins in the bath
A= 200e3
nu_e = 15e6   # Electron Larmor frequency
nu_n = 2e6   # Nuclear Larmor frequency

nu_RF = 1.9e6   # RF carrier frequency
nu_RF_Amplitude=2.5e4

t_max = 20e-6  # Simulation time
n_steps = 5000  # Number of time steps

###############################################################################################################################################################################################################################
# define time array for simulation
tlist = np.linspace(0, t_max, n_steps)

# Define spin operators for central electron spin
Sx = tensor(0.5*sigmax(), qeye([2]*(N)))
Sy = tensor(0.5*sigmay(), qeye([2]*(N)))
Sz = tensor(0.5*sigmaz(), qeye([2]*(N)))

Ix = 0.5*sigmax()
Iy = 0.5*sigmay()
Iz = 0.5*sigmaz()

def RF_pulse_envelope(tlist, t_mean=10e-6, t_width=4e-6):
    # return 1e5*np.exp(-((tlist-t_mean)/t_width)**2)
    return (1-np.cos(2*pi*tlist/(t_max)))

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

    # Define observables to track
    observables = [
        Sx,
        Sy,
        Sz,  # Electron spin z-component
        sum(bath_op(i, Ix) for i in range(N)),
        sum(bath_op(i, Iy) for i in range(N)),
        sum(bath_op(i, Iz) for i in range(N))  
    ]

    return observables

def define_initial_state(N, initial_electron_spin=0, initial_nuclear_spin=0, random_nuclear_spins=False, random_seed=42):
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

def define_hamiltonian(nu_RF_Amplitude,A, nu_e, nu_n, N,nu_RF, tlist):
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
    H_zeeman_e =2*pi * nu_e * Sz

    # Zeeman interactions for nuclear spins
    H_zeeman_n = 0
    for i in range(N): 
        H_zeeman_n -= 2 * pi *nu_n * bath_op(i, Iz)
    
    # Hyperfine interactions between electron and nuclear spins
    H_hyperfine = 0
    for i in range(N):
        H_hyperfine += (2*pi)*A*(
            Sx * bath_op(i, Ix) +
            Sy * bath_op(i, Iy) +
            Sz * bath_op(i, Iz)
        )

    # H_dipole = 0
    # # Dipole-dipole interactions between nuclear spins
    # for i in range(N):
    #     for j in range(i+1, N):
    #         H_dipole += A/100 * (
    #             bath_op(i, sigmax()) * bath_op(j, sigmax()) +
    #             bath_op(i, sigmay()) * bath_op(j, sigmay()) +
    #             bath_op(i, sigmaz()) * bath_op(j, sigmaz())
    #         )

    # RF interactions
    H_RF_x = 0
    for i in range(N):
        H_RF_x += (bath_op(i, Ix))

    H_RF_y = 0
    for i in range(N):
        H_RF_y += (bath_op(i, Iy))

    # Define the Hamiltonian as a time dependent Quantum Object, QobjEvo. 
    # QobjEvo is defined as a list of quantum objects (which must all be the same dimensions), tuples must be of the form (Qobj, list of coefficients)
    H = QobjEvo(
        [H_zeeman_e + H_zeeman_n + H_hyperfine, 
         [H_RF_x, 2*pi*nu_RF_Amplitude*RF_pulse_envelope(tlist)*np.cos(2*pi*nu_RF*tlist)], 
         [H_RF_y, -2*pi*nu_RF_Amplitude*RF_pulse_envelope(tlist)*np.sin(2*pi*nu_RF*tlist)]],
        tlist=tlist
    )

    return  H
 
def calculate_expectation_values(params, tlist, observables):
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

    # Unpack parameters from dictionary
    nu_RF_Amplitude = params['nu_RF_Amplitude']
    A = params['A']
    nu_e = params['nu_e']
    nu_n = params['nu_n']
    nu_RF = params['nu_RF']
    N = params['N']
    psi0 = params['psi0']

    # Call the time dependent hamiltonian with the given parameters
    H = define_hamiltonian(nu_RF_Amplitude, A, nu_e, nu_n, N, nu_RF, tlist)

    output = sesolve(H, psi0, tlist, observables, options={'progress_bar': False, 'nsteps': 10000, 'rtol': 1e-8, 'atol': 1e-8})

    # Extract expectation values
    return params, output.expect[0], output.expect[1], output.expect[2], output.expect[3], output.expect[4], output.expect[5]

def plot_results(results, tlist):
    """
    Plot the simulation results.
    
    Parameters:
        results: List of tuples containing simulation results
        tlist: Time array used in simulation
        nu_RF: RF carrier frequency
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    ax3 = axes[2].twinx()  # Create a twin Axes sharing the x-axis
    ax3.ticklabel_format(style='plain', useOffset=False)

    for result in results:
        params, electron_x, electron_y, electron_z, nuclear_x, nuclear_y, nuclear_z = result
        N = params['N']  # Get N from params

        axes[1].plot(tlist, nuclear_x/N, color='red', linestyle='--')
        axes[1].plot(tlist, nuclear_y/N, color='green', linestyle=':')
        axes[1].plot(tlist, nuclear_z/N, color='blue')

        axes[2].plot(tlist, electron_x, color='red', linestyle='--')
        axes[2].plot(tlist, electron_y, color='green', linestyle=':')
        
        ax3.plot(tlist, electron_z, color='blue')
        ax3.set_ylabel('<Sz> (Right Axis)', color='blue')  # Set label for the right axis
        ax3.tick_params(axis='y', labelcolor='blue')

    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('RF intensity')
    axes[0].set_title('RF')
    axes[0].plot(tlist, nu_RF_Amplitude*RF_pulse_envelope(tlist)*np.cos(2*pi*nu_RF*tlist), label='RF_x')

    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('<Total Nuclear Sz>')
    axes[1].set_title('Total Nuclear Spin Z-Component')
    axes[1].legend()

    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('<Sx>, <Sy>')
    axes[2].set_title('Electron Spin Z-Component')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def plot_results_2d(results, x_axis_name, y_axis_name):
    # Extract unique RF frequencies and amplitudes from params
    x_axis = sorted(list(set(result[0][x_axis_name] for result in results)))
    y_axis = sorted(list(set(result[0][y_axis_name] for result in results)))
    
    # Create empty matrix for z-component values
    z_matrix = np.zeros((len(x_axis), len(y_axis)))
    
    # Fill matrix with final z-component values
    for result in results:
        params, electron_x, electron_y, electron_z, nuclear_x, nuclear_y, nuclear_z = result
        x_axis_idx = x_axis.index(params[x_axis_name])
        y_axis_idx = y_axis.index(params[y_axis_name])
        z_matrix[x_axis_idx, y_axis_idx] = nuclear_z[-1] / params['N']
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.imshow(z_matrix, aspect='auto', origin='lower', 
               extent=[min(x_axis), max(x_axis), min(y_axis), max(y_axis)])
    plt.colorbar(label='Nuclear spin z-component')
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title('Nuclear Spin Z-Component at Final Time')
    plt.show()

def plot_results_1d(results, x_axis_name):
    # Extract RF amplitudes and final nuclear z values
    x_axis = []
    final_nuclear_z = []
    
    for result in results:
        params, _, _, _, _, _, nuclear_z = result
        x_axis.append(params[x_axis_name])
        final_nuclear_z.append(nuclear_z[-1] / params['N'])  # Normalize by N
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, final_nuclear_z, 'b.')
    plt.xlabel(x_axis_name)
    plt.ylabel('Final Nuclear Spin Z-Component')
    plt.title('RF Amplitude vs Final Nuclear Spin State')
    plt.grid(True)
    plt.show()

def plot_results_bloch_sphere(results, result_idx, nuclear_idx):
    """
    Plot the trajectory of a specific nuclear spin on the Bloch sphere.
    
    Parameters:
        results: List of simulation results
        result_idx: Index of the simulation result to plot
        nuclear_idx: Index of the nuclear spin to plot (0 to N-1)
    """
    # Create Bloch sphere
    b = Bloch()
    
    # Get the specific result
    params, ex, ey, ez, nx, ny, nz = results[result_idx]
    N = params['N']
    
    # Extract individual nuclear spin components
    # Divide by N since the results are summed over all spins
    x = 2*nx/N
    y = 2*ny/N
    z = 2*nz/N
    
    # Add points to Bloch sphere
    b.add_points([x, y, z])
    
    # Optional: add a line connecting the points
    b.add_points([x[0], y[0], z[0]], 'm') # Mark start point
    b.add_points([x[-1], y[-1], z[-1]], 'm') # Mark end point
    
    b.render()
    plt.show()

psi0=define_initial_state(N)
observables=define_observables(N)

#Define the parameters for each simulation, as a list of dictionaries. 
params_list=[]
for nu_RF in np.linspace(1.5e6, 2.5e6, 20):
    for nu_n in np.linspace(1.5e6, 2.5e6, 20):
        params_list.append({'nu_RF_Amplitude': nu_RF_Amplitude, 'A': A, 'nu_e': nu_e, 'nu_n': nu_n, 'nu_RF': nu_RF, 'N': N,  'psi0': psi0})

# Use loky_pmap to parallelize the simulations
results = loky_pmap(calculate_expectation_values, params_list,task_args=(tlist,observables), progress_bar=True)

plot_results(results, tlist)

# plot_results_1d(results, 'nu_n')
plot_results_2d(results, 'nu_RF', 'nu_n')
# plot_spin_bloch_sphere(results, 0, 0)

#To do
#Dipole dipole interactions
#Explore more interesting initial states
#Use mcsolve to simulate dephasing and relaxation
#Iterate over many cycles of mcsolve where the electron spin is reset, in order to simulate optical pumping of the spins.
