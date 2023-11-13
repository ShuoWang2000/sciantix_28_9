import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas

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

variable_selected = np.array(["time", "helium diffusivity pre exponential", "helium diffusivity activation energy", "error"])
online_data = getSelectedVariablesValueFromOutput(variable_selected, "optimization_onlie.txt")

time = online_data[:,0]
helium_diffusivity_pre_exponential = online_data[:,1]
helium_diffusivity_activation_energy = online_data[:,2]



