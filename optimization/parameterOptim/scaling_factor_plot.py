import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 20    # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 20   # Font size for x-ticks
plt.rcParams['ytick.labelsize'] = 20  # Font size for y-ticks

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
    plt.plot(time1.to_numpy(), diffusivity1.to_numpy()[-1]*np.ones(n_points), label='Helium diffusivity pre-exponential - global offline optimization')
    plt.scatter(time1.to_numpy(), diffusivity1.to_numpy(), label='Helium diffusivity pre-exponential - stepwise offline optimization')
    plt.scatter(time2.to_numpy(), diffusivity2.to_numpy(), label='Helium diffusivity pre-exponential - online optimization')
    
    # Set labels and title
    plt.xlabel('Time (h)')
    plt.ylabel('Scaling factor')
    
    # Show a legend
    plt.legend(loc='best')

    # Show the plot
    plt.show()
    
    # Create a figure and plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(time1.to_numpy(), activation_energy1.to_numpy()[-1]*np.ones(n_points), label='Helium diffusivity activation energy - global offline optimization')
    plt.plot(time1.to_numpy(), activation_energy1.to_numpy(), label='Helium diffusivity activation energy - stepwise offline optimization')
    plt.scatter(time2.to_numpy(), activation_energy2.to_numpy(), label='Helium diffusivity activation energy - online optimization')
    
    # Set labels and title
    plt.xlabel('Time (h)')
    plt.ylabel('Scaling factor')
    
    # Show a legend
    plt.legend(loc='best')

    # Show the plot
    plt.show()

# Example usage:
file_path1 = 'optimization_offline.txt'  # Replace with the actual file path
file_path2 = 'optimization_online.txt'  # Replace with the actual file path

plot_data(file_path1, file_path2)
