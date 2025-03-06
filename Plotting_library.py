import matplotlib.pyplot as plt
from qutip import *
import numpy as np
import random
import time
from scipy.constants import pi
import dask
from dask.distributed import Client, progress
from Simulation_library import *

def plot_results(results, tlist,base_params, num_params, param_names):
    """
    Plot the simulation results.
    
    Parameters:
        results: List of tuples containing simulation results
        tlist: Time array used in simulation
        nu_RF: RF carrier frequency
    """
    nu_RF_Amplitude=base_params['nu_RF_Amplitude']
    nu_RF=base_params['nu_RF']
    t_max=base_params['t_max']

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
    axes[0].plot(tlist, nu_RF_Amplitude*RF_pulse_envelope(tlist,t_max)*np.cos(2*pi*nu_RF*tlist), label='RF_x')

    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('<Total Nuclear Sz>')
    axes[1].set_title('Total Nuclear Spin Z-Component')
 
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('<Sx>, <Sy>')
    axes[2].set_title('Electron Spin Z-Component')
 

    plt.tight_layout()
    plt.show()

    if num_params == 1:
        plot_results_1d(results, param_names[0])
    elif num_params == 2:
        plot_results_2d(results, param_names[0], param_names[1])

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
