import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_selected_variables_value_from_output_last_line(variable_selected, source_file):
    with open(source_file, 'r') as file:
        header = file.readline().strip().split('\t')
        
        for last_non_empty_line in reversed(file.readlines()):
            if last_non_empty_line.strip():
                break

    variable_positions = [header.index(var) for var in variable_selected if var in header]
    
    # Optional: Warn about missing variables
    missing_vars = [var for var in variable_selected if var not in header]
    if missing_vars:
        print(f"Warning: The following variables were not found in the file and will be ignored: {missing_vars}")

    # Use only the last non-empty line to create the data array
    data = np.genfromtxt([last_non_empty_line], dtype='str', delimiter='\t')
    variable_selected_value = data[variable_positions].astype(float)
    
    return variable_selected_value

code_container = os.getcwd()
optim_container = os.path.join(code_container, 'Optimization')

os.chdir(optim_container)
variables = np.array(["Time (h)","Temperature (K)","He fractional release (/)", "He release rate (at/m3 s)"])

time_point = np.linspace(0, 3.62725, 101)
fr = np.zeros_like(time_point)


for i in range(1,len(time_point)):
    for folder in os.listdir(optim_container):
        folder_path = os.path.join(optim_container, folder)
        if f'to_{np.round(time_point[i],3)}' in folder:
            os.chdir(folder_path)
            fr[i] = get_selected_variables_value_from_output_last_line(variables, 'output.txt')[2]

plt.plot(time_point, fr)
plt.show()

