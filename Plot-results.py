import pickle
import os
from datetime import datetime
from Plotting_library import *

picklefile='results_0603_1449.pkl'

# Open and load the pickle file
with open(f'Simulation Results/{picklefile}', 'rb') as f:
    results, tlist,base_params, num_params, param_names = pickle.load(f)

plot_results(results, tlist,base_params, num_params, param_names)

plot_results_1d(results, param_names[0])