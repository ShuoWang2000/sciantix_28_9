import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

def plot_data(file_path1, file_path2):
    data1 = pd.read_csv(file_path1, delimiter='\t')

    time1 = data1['time']
    diffusivity1 = data1['helium diffusivity pre exponential']
    activation_energy1 = data1['helium diffusivity activation energy']

    data2 = pd.read_csv(file_path2, delimiter='\t')

    time2 = data2['time']
    diffusivity2 = data2['helium diffusivity pre exponential']
    activation_energy2 = data2['helium diffusivity activation energy']

    n_points = len(time1.to_numpy())

    plt.figure(figsize=(10, 6))
    plt.plot(time1.to_numpy(), diffusivity1.to_numpy()[-1]*np.ones(n_points), 'b', label='Helium diffusivity pre-exponential - global offline optimization')
    plt.plot(time1.to_numpy(), diffusivity1.to_numpy(), '-xb', label='Helium diffusivity pre-exponential - stepwise offline optimization')
    plt.plot(time2.to_numpy(), diffusivity2.to_numpy(), '-go', label='Helium diffusivity pre-exponential - online optimization')
    plt.xlabel('Time (h)')
    plt.ylabel('Scaling factor')
    plt.legend(loc='best')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(time1.to_numpy(), activation_energy1.to_numpy()[-1]*np.ones(n_points), 'b', label='Helium diffusivity activation energy - global offline optimization')
    plt.plot(time1.to_numpy(), activation_energy1.to_numpy(), '-xb',label='Helium diffusivity activation energy - stepwise offline optimization')
    plt.plot(time2.to_numpy(), activation_energy2.to_numpy(), '-go', label='Helium diffusivity activation energy - online optimization')
    plt.xlabel('Time (h)')
    plt.ylabel('Scaling factor')
    plt.legend(loc='best')
    plt.show()

file_path1 = 'optimization_offline.txt'
file_path2 = 'optimization_online.txt'

plot_data(file_path1, file_path2)
