import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.optimize import Bounds
import re
import pandas as pd

def getSelectedVariablesValueFromOutput(variable_selected, source_file):
    numberOfSelectedVariable = len(variable_selected)
    with open(source_file, 'r') as file:
        lines = [line for line in file if line.strip()]
    
    data = np.genfromtxt(lines, dtype='str', delimiter='\t')
    l = len(data[:, 0]) - 1
    variablePosition = np.zeros((numberOfSelectedVariable), dtype='int')
    variable_selected_value = np.zeros((l, numberOfSelectedVariable), dtype='float')
    
    for i in range(numberOfSelectedVariable):
        j = np.where(data == variable_selected[i])
        j = j[1]
        variablePosition[i] = j[0]
        variable_selected_value[:, i] = data[1:, variablePosition[i]].astype(float)
    
    return variable_selected_value

def readSFfromInputScalingFactors(end_time, keyword):
    """
    this function read the sf from the optimization file from time 0
    keyword: the scaling factors name, for example: "Helium diffuscivity pre exponential"
    """
    find_value =False
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    for folder_name in os.listdir(parent_directory):
        if folder_name == f"Optimization_0__{end_time}_":
            os.chdir(f"Optimization_0__{end_time}_")
            with open("input_scaling_factors.txt", 'r') as file:
                line_number = 0
                previous_line = None
                for line in file:
                    line_number += 1
                    if keyword in line:
                        break
                    previous_line = line.strip() 
            if previous_line is not None:
                value = float(previous_line)
                find_value = True
            break
        else:
            pass
    
    if find_value == True:
        pass
    else:
        value = "No data"
    return value

os.chdir("test_Talip2014_1600K")
current_directory = os.getcwd()

#########
# OFFLINE
#########

keyword = "Optimization_0_"
folder1_collection = []
pattern = r'Optimization_\d+_(\d+\.\d+)_'
find = False
for folder_name in os.listdir(current_directory):
    folder_path = os.path.join(current_directory, folder_name)
    if os.path.isdir(folder_path) and keyword in folder_name:
        find = True
        folder1_collection.append(folder_name)

time_end1 = []
for file_name in folder1_collection:
    match = re.search(pattern, file_name)
    if match:
        time = np.round(float(match.group(1)),3)
        time_end1.append(time)
time_end1 = np.array(time_end1)

time_end_previous = np.max(time_end1)
time_end_first = np.min(time_end1)
folder_path_global_time_end_previous = os.path.join(current_directory, f"Optimization_0_{time_end_previous}_")
folder_path_global_time_end_first = os.path.join(current_directory, f"Optimization_0_{time_end_first}_")

variable_selected = np.array(["Time (h)","Temperature (K)","He fractional release (/)", "He release rate (at/m3 s)"])

os.chdir(folder_path_global_time_end_previous)
data_global = getSelectedVariablesValueFromOutput(np.array(["Time (h)","Temperature (K)","He fractional release (/)", "He release rate (at/m3 s)"]),"output.txt")

os.chdir(folder_path_global_time_end_first)
data0 = getSelectedVariablesValueFromOutput(np.array(["Time (h)","Temperature (K)","He fractional release (/)", "He release rate (at/m3 s)"]),"output.txt")
length0 = len(data0)

########
# ONLINE
# ######

os.chdir(current_directory)

keyword2 = "Optimization_"
pattern2 = r'Optimization_(\d+\.\d+)+__(\d+\.\d+)_'

folder2_collection = []
find2 = False
for folder_name in os.listdir(current_directory):
    folder_path = os.path.join(current_directory, folder_name)
    if os.path.isdir(folder_path) and keyword2 in folder_name:
        find = True
        folder2_collection.append(folder_name)
# print(folder2_collection)
time_end2 = []
time_start2 = []
for file_name in folder2_collection:
    match2 = re.search(pattern2, file_name)
    if match2:
        time_start = np.round(float(match2.group(1)),3)
        time_end = np.round(float(match2.group(2)),3)
        time_start2.append(time_start)
        time_end2.append(time_end)

time_start2 = np.sort(np.array(time_start2))
time_end2 = np.sort(np.array(time_end2))

data = np.empty((0,4))
data = np.vstack((data, data0))
# folder_path_online = np.empty((0,1))

length = np.zeros(len(time_start2))
time = []

for i in range(len(time_start2)):
    os.chdir(f"Optimization_{time_start2[i]}__{time_end2[i]}_")
    data_output = getSelectedVariablesValueFromOutput(variable_selected, "output.txt")
    length[i] = len(data_output)
    # data_output[:,0] = data_output[:,0] + time_start2[i]
    data_output[:,0] = data_output[:,0] + time_start2[i]
    # print(data_output[0,0])
    data = np.vstack((data, data_output))
    os.chdir(current_directory)

length_all = np.zeros(len(time_start2)+1)

length_all[0] = length0
length_all[1:] = length
for i in range(1,len(length_all)):
    length_all[i] = length_all[i] + length_all[i-1]

cloumnsFR  = np.genfromtxt("Talip2014_release_data.txt",dtype = 'float',delimiter='\t')
cloumnsRR = np.genfromtxt("Talip2014_rrate_data.txt",dtype = 'float',delimiter='\t')
variable_selected = np.array(["Time (h)","Temperature (K)","He fractional release (/)", "He release rate (at/m3 s)"])
coloumnsOutput_nominal = getSelectedVariablesValueFromOutput(variable_selected,"output.txt")

time_exp  = cloumnsFR[:,0]
FR_exp = cloumnsFR[:,1]
temperature_exp = cloumnsRR[:,0]
RR_exp = cloumnsRR[:,1]

time_sciantix = coloumnsOutput_nominal[:,0]
temperature_sciantix = coloumnsOutput_nominal[:,1]
FR_nominal = coloumnsOutput_nominal[:,2]
RR_nominal = coloumnsOutput_nominal[:,3]
temperature_new = data[:,1]
FR_new = data[:,2]
RR_new = data[:,3]

fig, ax = plt.subplots(1,2)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.34,
                    hspace=0.4)

ax[0].scatter(time_exp, FR_exp, marker = '.', c = '#B3B3B3', label='Data from Talip et al. (2014)')
ax[0].plot(time_sciantix, FR_nominal, 'g', label='SCIANTIX 2.0 - Nominal')
ax[0].plot(data_global[:,0], data_global[:,2], label = "Offline optimization" )
ax[0].scatter(time_new[0:length0], FR_new[0:length0], label = 'Online optimization')

for i in range(0,len(time_start2)):
    ax[0].scatter(time_new[int(length_all[i]):int(length_all[i+1])], FR_new[int(length_all[i]):int(length_all[i+1])], label = None)

# ax[0].scatter(time_new[length0:int(length[0])], FR_new[length0:int(length[0])], marker = 'x',label = f'optimized_0.567__1.134')

# ax[0].scatter(time_new, FR_interpolated, marker = 'x',color = 'blue',label = 'interpolated')
axT = ax[0].twinx()
axT.set_ylabel('Temperature (K)')
axT.plot(time_sciantix, temperature_sciantix, 'r', linewidth=1, label="Temperature")

ax[0].set_xlabel('Time (h)')
ax[0].set_ylabel('Helium fractional release (/)')
h1, l1 = ax[0].get_legend_handles_labels()
h2, l2 = axT.get_legend_handles_labels()
# ax[0].legend(h1+h2, l1+l2)
ax[0].legend(loc = 'upper left')

""" Plot: Helium release rate """
ax[1].scatter(temperature_exp, RR_exp, marker = '.', c = '#B3B3B3', label='Data from Talip et al. (2014)')
ax[1].plot(temperature_sciantix, RR_nominal, 'g', label='SCIANTIX 2.0 - Nominal')
ax[1].plot(data_global[:,1], data_global[:,3], label = "Offline optimization")
ax[1].scatter(temperature_new[0:length0], RR_new[0:length0],label = 'Online optimization')
for i in range(0,len(time_start2)):
    ax[1].scatter(temperature_new[int(length_all[i]):int(length_all[i+1])], RR_new[int(length_all[i]):int(length_all[i+1])], label = None)

# ax.set_title(file + ' - Release rate')
ax[1].set_xlabel('Temperature (K)')
ax[1].set_ylabel('Helium release rate (at m${}^{-3}$ s${}^{-1}$)')
ax[1].legend()

sf_D0 = readSFfromInputScalingFactors(time_end1[-1], "helium diffusivity pre exponential")
sf_De = readSFfromInputScalingFactors(time_end1[-1], "helium diffusivity activation energy")
plt.title(f"sf_D0:{sf_D0}; sf_De: {sf_De} ")

# plt.savefig(file + '.png')
plt.show()
