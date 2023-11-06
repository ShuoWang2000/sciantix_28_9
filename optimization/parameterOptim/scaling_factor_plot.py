import pandas as pd
import matplotlib.pyplot as plt

def plot_data(file_path1, file_path2):
    data1 = pd.read_csv(file_path1, delimiter='\t')

    time1 = data1['time']
    diffusivity1 = data1['helium diffusivity pre exponential']
    activation_energy1 = data1['helium diffusivity activation energy']

    data2 = pd.read_csv(file_path2, delimiter='\t')

    time2 = data2['time']
    diffusivity2 = data2['helium diffusivity pre exponential']
    activation_energy2 = data2['helium diffusivity activation energy']

    plt.figure(figsize=(10, 6))
    plt.plot(time1.to_numpy(), diffusivity1.to_numpy(), label='Helium diffusivity pre-exponential - offline optimization')
    plt.plot(time2.to_numpy(), diffusivity2.to_numpy(), label='Helium diffusivity pre-exponential - online optimization')
    
    # Set labels and title
    plt.xlabel('Time (h)')
    plt.ylabel('Scaling factor')
    
    # Show a legend
    plt.legend(loc='best')

    # Show the plot
    plt.show()
    
    # Create a figure and plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(time1.to_numpy(), activation_energy1.to_numpy(), label='Helium diffusivity activation energy - offline optimization')
    plt.plot(time2.to_numpy(), activation_energy2.to_numpy(), label='Helium diffusivity activation energy - online optimization')
    
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