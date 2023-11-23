import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.optimize import Bounds
import pandas
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt import SequentialDomainReductionTransformer

class inputOutput():
	def __init__(self):
		pass

	def writeInputHistory(self,history_original, time_start, time_end):
		time_history_original = history_original[:,0]
		temperature_history_original = history_original[:,1]
		temperature_start = interpolate_1D(time_history_original, temperature_history_original, time_start)
		temperature_end = interpolate_1D(time_history_original, temperature_history_original, time_end)
		# print(temperature_end)
		index_start = np.where(time_history_original <= time_start)[0][-1]
		index_end = np.where(time_history_original >= time_end)[0][0]
		if index_start != index_end:

			new_history = history_original[index_start:index_end+1,:].copy()
			new_history[0,0:2] = [time_start, temperature_start]
			new_history[-1,0:2] = [time_end, temperature_end]
			new_history[:,0] = new_history[:,0] - time_start
		else:
			new_history = history_original[index_start:index_end+2,:].copy()
			new_history[0,0:2] = [time_start, temperature_start]
			new_history[-1,0:2] = [time_end, temperature_end]
			new_history[:,0] = new_history[:,0] - time_start

		new_history = [[str(item) for item in row] for row in new_history]
		
		with open("input_history.txt", 'a') as file:
			total_rows = len(new_history)
			for index, row in enumerate(new_history):
				line = "\t".join(row)  # Join columns with a tab separator
				if index == total_rows - 1:
					file.write(line)
				else:
					file.write(line + "\n")

	def writeInputInitialConditions(self, ic_new, ic_grainRadius, ic_intraGrainBubbleRadius ):
		with open("input_initial_conditions.txt", 'r') as file:
			lines = file.readlines()

		keyword1 = "#	initial He (at/m3) produced, intragranular, intragranular in solution, intragranular in bubbles, grain boundary, released"
		keyword2 = "#	initial grain radius (m)"
		keyword3 = "#	initial intragranular bubble concentration (at/m3), radius (m)"

		for i, line in enumerate(lines):
			if keyword1 in line:
				ic_1 = '\t'.join(map(str, ic_new))
				lines[i - 1] = ic_1 + '\n'  

				break
		for j, line in enumerate(lines):
			if keyword2 in line:
				ic_2 = ic_grainRadius
				lines[j - 1] = str(ic_2) + '\n'  

				break
		for k, line in enumerate(lines):
			if keyword3 in line:
				ic_3 = '  '.join(map(str, ic_intraGrainBubbleRadius))
				lines[k - 1] = ic_3 + '\n'  

				break

		with open("input_initial_conditions.txt", 'w') as file:
			file.writelines(lines)

	def writeInputScalingFactors(self,scaling_factors,sf_selected,kwargs):
		# assign new sf value and write the file, and run sciantix.x
		# for i in range(len(sf_selected)):
			# scaling_factors[sf_selected[i]] = sf_selected_value[i]
		for i in range(len(sf_selected)):
			scaling_factors[sf_selected[i]] = kwargs[sf_selected[i]]

		with open("input_scaling_factors.txt",'w') as file:
			for key, value in scaling_factors.items():
				if(f'{key}' == 'helium diffusivity pre exponential'):
					file.write(f'{np.exp(value)}\n')
					file.write(f'# scaling factor - {key}\n')
				elif(f'{key}' == 'henry constant pre exponential'):
					file.write(f'{np.exp(value)}\n')
					file.write(f'# scaling factor - {key}\n')

				else:
					file.write(f'{value}\n')
					file.write(f'# scaling factor - {key}\n')		
		
		# self.sf_selected_value = sf_selected_value

		# sciantix simulation during optimization
		os.system("./sciantix.x")
	
	def readOutput(self):
		self.variable_selected = np.array(["Time (h)","Temperature (K)","He fractional release (/)", "He release rate (at/m3 s)"])
		self.variable_selected_value = getSelectedVariablesValueFromOutput(self.variable_selected,"output.txt")
		self.time_sciantix = self.variable_selected_value[:,0]
		self.temperature_sciantix = self.variable_selected_value[:,1]

		self.FR_sciantix = self.variable_selected_value[:,2]
		self.RR_sciantix = self.variable_selected_value[:,3]

class optimization():
	def __init__(self) -> None:
		pass
	def setCase(self,caseName):
		self.caseName = caseName

		os.chdir(self.caseName)
		cloumnsFR  = np.genfromtxt("Talip2014_release_data.txt",dtype = 'float',delimiter='\t')
		cloumnsRR = np.genfromtxt("Talip2014_rrate_data.txt",dtype = 'float',delimiter='\t')
		history = np.genfromtxt("input_history.txt",dtype = 'float', delimiter='\t')

		self.time_exp  = cloumnsFR[:,0]
		self.FR_exp = cloumnsFR[:,1]
		self.temperature_exp = cloumnsRR[:,0]
		self.RR_exp = moving_average(cloumnsRR[:,1],5)
		
		FR_smoothed = moving_average(self.FR_exp,100)
		for i in range(len(FR_smoothed)):
			if FR_smoothed[i] < 0:
				FR_smoothed[i] = 0
		
		self.FR_smoothed = FR_smoothed

		self.history_original = history
		self.time_history_original = self.history_original[:,0]
		self.temperature_history_original = self.history_original[:,1]

	def setStartEndTime(self, time_start, time_end):
		self.time_start = time_start
		self.time_end = time_end

		if self.time_start !=0:
			folder_name = f"Optimization_{self.time_start}__{self.time_end}_"
		else:
			folder_name = f"Optimization_0_{self.time_end}_"
		
		if not os.path.exists(folder_name):
			os.makedirs(folder_name)
		else:
			shutil.rmtree(folder_name)
			os.makedirs(folder_name)
	
		os.chdir(folder_name)
		shutil.copy("../input_scaling_factors.txt", os.getcwd())
		shutil.copy("../../../../bin/sciantix.x", os.getcwd())
		shutil.copy("../input_initial_conditions.txt", os.getcwd())
		shutil.copy("../input_settings.txt", os.getcwd())

	def setInitialConditions(self):
		current_directory = os.getcwd()
		parent_directory = os.path.dirname(current_directory)

		ic_origin = readICfromInputInitialConditions("#	initial He (at/m3) produced")
		ic_grainRadius_origin = readICfromInputInitialConditions("#	initial grain radius (m)")[0]

		with open("input_initial_conditions.txt", 'r') as file:
			intraGBR = "# initial intragranular bubble concentration (at/m3), radius (m)"
			line_number = 0
			previous_line = None
			for line in file:
				line_number += 1
				if intraGBR in line:
					break 
				previous_line = line.strip() 
		if previous_line is not None:
			values = [float(val) for val in previous_line.split('  ')]
			ic_intraGrainBubbleRadius_origin = np.array(values)

		if self.time_start != 0:
			keyword1 = f"__{self.time_start}_"
			keyword2 = f"Optimization_0_{self.time_start}_"
			find1 = False
			find2 = False
			for folder_name in os.listdir(parent_directory):
				folder_path = os.path.join(parent_directory, folder_name)
				if os.path.isdir(folder_path) and keyword1 in folder_name:
					folder_path1 = folder_path
					find1 = True
				elif os.path.isdir(folder_path) and keyword2 in folder_name:
					folder_path2 = folder_path
					find2 = True
			
			if find1 == True:
				os.chdir(folder_path1)
				# print(folder_path1)
				self.folder_path_previous_optimization = folder_path1

			else:
				os.chdir(folder_path2)
				self.folder_path_previous_optimization = folder_path2
			
			ic_name = np.array(["He produced (at/m3)","He in grain (at/m3)", "He in intragranular solution (at/m3)", "He in intragranular bubbles (at/m3)", "He at grain boundary (at/m3)", "He released (at/m3)"])
			ic_value_previous = getSelectedVariablesValueFromOutput(ic_name,"output.txt")
			ic_new = ic_value_previous[-1,:]
			ic_grainRadius = getSelectedVariablesValueFromOutput(np.array(["Grain radius (m)"]),"output.txt")[-1][0]
			ic_intraGrainBubbleRadius = getSelectedVariablesValueFromOutput(np.array(["Intragranular bubble concentration (bub/m3)","Intragranular bubble radius (m)"]),"output.txt")[-1,:]
			self.output_previous = getSelectedVariablesValueFromOutput(np.array(["Time (h)","Temperature (K)","He fractional release (/)", "He release rate (at/m3 s)"]), "output.txt")
			RR_ic = self.output_previous[-1,3]
			FR_ic = self.output_previous[-1,2]
			
			self.interpolated_previous = np.genfromtxt("interpolated_data.txt",dtype = 'float',delimiter='\t')
			RR_interpolated_ic = self.interpolated_previous[-1,1]
			if RR_interpolated_ic < 0:
				RR_interpolated_ic = 0
			FR_interpolated_ic = self.interpolated_previous[-1,0]

			self.scaling_factors = {}
			with open("input_scaling_factors.txt", 'r') as file:
				lines = file.readlines()
			sf_name = []
			i = 0
			while i < len(lines):
				value = float(lines[i].strip())
				sf_name.append(lines[i + 1].strip()[len("# scaling factor - "):])
				self.scaling_factors[sf_name[-1]] = value
				i += 2
			
		else: # self.time_start == 0:
			os.chdir(current_directory)
			ic_new = ic_origin
			ic_grainRadius = ic_grainRadius_origin
			ic_intraGrainBubbleRadius = ic_intraGrainBubbleRadius_origin
			RR_ic = 0
			FR_ic = 0
			RR_interpolated_ic = 0
			FR_interpolated_ic = 0
			
			self.scaling_factors = {}
			with open("input_scaling_factors.txt", 'r') as file:
				lines = file.readlines()
			sf_name = []
			i = 0
			while i < len(lines):
				value = float(lines[i].strip())
				sf_name.append(lines[i + 1].strip()[len("# scaling factor - "):])
				self.scaling_factors[sf_name[-1]] = value
				i += 2

			self.scaling_factors['helium diffusivity pre exponential'] = 0.0
			self.scaling_factors['henry constant pre exponential'] = 0.0

		self.ic_new = ic_new
		self.ic_grainRadius = ic_grainRadius
		self.ic_intraGrainBubbleRadius = ic_intraGrainBubbleRadius
		self.RR_ic = RR_ic
		self.FR_ic = FR_ic
		self.RR_interpolated_ic = RR_interpolated_ic
		self.FR_interpolated_ic = FR_interpolated_ic
		
		os.chdir(current_directory)
		self.current_directory = current_directory
	
	def setScalingFactors(self,*args):
		
		self.sf_selected = []
		for arg in args:
			self.sf_selected.append(arg)

		self.sf_selected_initial_value = np.ones([len(self.sf_selected)])
		for i in range(len(self.sf_selected)):
			self.sf_selected_initial_value[i] = self.scaling_factors[self.sf_selected[i]]
		
		self.sf_selected_bounds = np.zeros([2,len(self.sf_selected_initial_value)])
		for i in range(len(self.sf_selected)):
			if self.sf_selected[i] == "helium diffusivity pre exponential":
				self.sf_selected_bounds[0,i] = np.log(0.05)
				self.sf_selected_bounds[1,i] = np.log(19.9)
			elif self.sf_selected[i] == "helium diffusivity activation energy":
				self.sf_selected_bounds[0,i] = 0.835
				self.sf_selected_bounds[1,i] = 1.2
			elif self.sf_selected[i] == "henry constant pre exponential":
				self.sf_selected_bounds[0,i] = np.log(0.0627)
				self.sf_selected_bounds[1,i] = np.log(16.09)
			elif self.sf_selected[i] == "henry constant activation energy":
				self.sf_selected_bounds[0,i] = 0.431
				self.sf_selected_bounds[1,i] = 1.55
			else:
				self.sf_selected_bounds[0,i] = 0
				self.sf_selected_bounds[0,i] = float('inf')
		self.bounds = Bounds(self.sf_selected_bounds[0,:],self.sf_selected_bounds[1,:])

		self.BO_bounds = {}
		for i in range(len(self.sf_selected)):
			self.BO_bounds[self.sf_selected[i]] = (self.sf_selected_bounds[0,i], self.sf_selected_bounds[1,i])

	def optimization(self,inputOutput, new_bounds):
		
		inputOutput.writeInputHistory(self.history_original, self.time_start, self.time_end)
		inputOutput.writeInputInitialConditions(self.ic_new, self.ic_grainRadius, self.ic_intraGrainBubbleRadius)

		def costFunction(**kwargs):
			# sf_selected_value = np.fromiter(kwargs.values(), dtype=float)
			
			inputOutput.writeInputScalingFactors(self.scaling_factors,self.sf_selected,kwargs)
			inputOutput.readOutput()
			FR_interpolated = np.zeros_like(inputOutput.FR_sciantix)
			RR_interpolated = np.zeros_like(inputOutput.RR_sciantix)
			RR_fr = np.zeros_like(inputOutput.RR_sciantix)
			time_sciantix = inputOutput.time_sciantix + self.time_start
			temperature_sciantix = inputOutput.temperature_sciantix

			FR_sciantix = inputOutput.FR_sciantix
			RR_sciantix = inputOutput.RR_sciantix
			dFR_dt_sciantix = np.zeros_like(inputOutput.RR_sciantix)
			if self.time_start == 0.0:
				dFR_dt_sciantix[0] = 0
			else:
				dFR_dt_sciantix[0] = (self.output_previous[-1,2]-self.output_previous[-2,2])/(self.output_previous[-1,0]-self.output_previous[-2,0])
			for i in range(1, len(FR_sciantix)):
				if time_sciantix[i] == time_sciantix[i-1]:
					dFR_dt_sciantix[i] = dFR_dt_sciantix[i-1]
				else:
					dFR_dt_sciantix[i] = (FR_sciantix[i] - FR_sciantix[i-1])/(time_sciantix[i]-time_sciantix[i-1])
						
			dFR_dt = np.zeros_like(inputOutput.FR_sciantix)
			dFR_dtdt = np.zeros_like(inputOutput.FR_sciantix)
			Helium_total = self.ic_new[0]

			FR_interpolated[0] = self.FR_interpolated_ic
			
			RR_interpolated[0] = self.RR_interpolated_ic
			plateau_index = []

			if time_sciantix[0] > max(self.time_exp):
				index_max_time_exp = 0

			else:
				index_max_time_exp = findClosestIndex_1D(time_sciantix, max(self.time_exp))
				
				if len(time_sciantix) > index_max_time_exp+1:
					if time_sciantix[index_max_time_exp] == time_sciantix[index_max_time_exp+1]:
						index_max_time_exp = index_max_time_exp + 1
				if time_sciantix[index_max_time_exp] > max(self.time_exp):
						index_max_time_exp = index_max_time_exp - 1
				if index_max_time_exp == 0:
					pass
				else:
					for i in range(1,index_max_time_exp + 1):

						if time_sciantix[i] == time_sciantix[i-1]:
							# in sciantix history there might be doubled time points
							FR_interpolated[i] = FR_interpolated[i-1]
							RR_fr[i] = RR_fr[i-1]
							RR_interpolated[i] = RR_interpolated[i-1]
							
						else:
							FR_interpolated[i] = interpolate_1D(self.time_exp, self.FR_smoothed, time_sciantix[i])
							if FR_interpolated[i] <0 :
								FR_interpolated[i] =0
							if FR_interpolated[i] < FR_interpolated[i-1]:
								FR_interpolated[i] = FR_interpolated[i-1]
							
							RR_fr[i] =(FR_interpolated[i]-FR_interpolated[i-1])/(time_sciantix[i]-time_sciantix[i-1])/3600*Helium_total
							RR_interpolated[i] = interpolate_2D(self.temperature_exp, self.RR_exp, temperature_sciantix[i], RR_fr[i])
							if RR_interpolated[i] < 0:
								RR_interpolated[i] = 0
					
					if self.time_start == 0:

						dFR_dt[0:index_max_time_exp+1] = np.gradient(FR_interpolated[0:index_max_time_exp+1], time_sciantix[0:index_max_time_exp+1])
						for i in range(index_max_time_exp+1):
							if dFR_dt[i] < 0:
								dFR_dt[i] = 0
						dFR_dtdt[0:index_max_time_exp+1] = np.gradient(dFR_dt[0:index_max_time_exp+1],time_sciantix[0:index_max_time_exp+1])

					else:
						first_derivative = np.gradient(self.interpolated_previous[:,0], self.output_previous[:,0])
						second_derivative = np.gradient(first_derivative, self.output_previous[:,0])

						for j in range(len(first_derivative)):
							if np.isnan(first_derivative[::-1])[j] == False:
								dFR_dt[0] = first_derivative[::-1][j]
								break

						for k in range(len(second_derivative)):
							if np.isnan(second_derivative[::-1])[k] == False:
								dFR_dtdt[0] = first_derivative[::-1][k]
								break 

						dFR_dt[1:index_max_time_exp+1] = (np.gradient(FR_interpolated[1:index_max_time_exp+1], time_sciantix[1:index_max_time_exp+1]))
						for i in range(index_max_time_exp+1):
							if dFR_dt[i] < 0:
								dFR_dt[i] = 0
						dFR_dtdt[1:index_max_time_exp+1] = np.gradient(dFR_dt[1:index_max_time_exp+1],time_sciantix[1:index_max_time_exp+1])
					
			for i in range(index_max_time_exp+1,len(time_sciantix)):
				
				if index_max_time_exp == 0:
					first_derivative = np.gradient(self.interpolated_previous[:,0], self.output_previous[:,0])
					second_derivative = np.gradient(first_derivative, self.output_previous[:,0])

					for j in range(len(first_derivative)):
						if np.isnan(first_derivative[::-1])[j] == False:
							dFR_dt[0] = first_derivative[::-1][j]
							break
					for k in range(len(second_derivative)):
						if np.isnan(second_derivative[::-1])[k] == False:
							dFR_dtdt[0] = first_derivative[::-1][k]
							break
				
				elif index_max_time_exp == 1:
					first_derivative = np.gradient(self.interpolated_previous[:,0], self.output_previous[:,0])
					second_derivative = np.gradient(first_derivative, self.output_previous[:,0])
					if time_sciantix[index_max_time_exp] != time_sciantix[index_max_time_exp-1]:
						dFR_dt[index_max_time_exp] = (FR_interpolated[index_max_time_exp]-FR_interpolated[index_max_time_exp-1])/(time_sciantix[index_max_time_exp]-time_sciantix[index_max_time_exp-1])
						for j in range(len(first_derivative)):
							if np.isnan(first_derivative[::-1])[j] == False:
								dFR_dt[0] = first_derivative[::-1][j]
								break
						dFR_dtdt[index_max_time_exp] = (dFR_dt[1] - dFR_dt[0])/(time_sciantix[1]-time_sciantix[0])
					else:
						for j in range(len(first_derivative)):
							if np.isnan(first_derivative[::-1])[j] == False:
								dFR_dt[0] = first_derivative[::-1][j]
								break
						for k in range(len(second_derivative)):
							if np.isnan(second_derivative[::-1])[k] == False:
								dFR_dtdt[0] = first_derivative[::-1][k]
								break

				else:
					if time_sciantix[index_max_time_exp] != time_sciantix[index_max_time_exp-1] and time_sciantix[index_max_time_exp-1] != time_sciantix[index_max_time_exp-2]:
						
						dFR_dt[index_max_time_exp] = (FR_interpolated[index_max_time_exp]-FR_interpolated[index_max_time_exp-1])/(time_sciantix[index_max_time_exp]-time_sciantix[index_max_time_exp-1])
						dFR_dt[index_max_time_exp-1] = (FR_interpolated[index_max_time_exp-1] - FR_interpolated[index_max_time_exp-2])/(time_sciantix[index_max_time_exp-1]-time_sciantix[index_max_time_exp-2])
						dFR_dtdt[index_max_time_exp] = (dFR_dt[index_max_time_exp]-dFR_dt[index_max_time_exp-1])/(time_sciantix[index_max_time_exp]-time_sciantix[index_max_time_exp-1])
					
					else:
						if index_max_time_exp > 2 and time_sciantix[index_max_time_exp] == time_sciantix[index_max_time_exp-1] and time_sciantix[index_max_time_exp-1] != time_sciantix[index_max_time_exp-2]:
							
							dFR_dt[index_max_time_exp] = (FR_interpolated[index_max_time_exp-1]-FR_interpolated[index_max_time_exp-2])/(time_sciantix[index_max_time_exp-1]-time_sciantix[index_max_time_exp-2])
							dFR_dt[index_max_time_exp-1] = (FR_interpolated[index_max_time_exp-2] - FR_interpolated[index_max_time_exp-3])/(time_sciantix[index_max_time_exp-2]-time_sciantix[index_max_time_exp-3])
							dFR_dtdt[index_max_time_exp] = (dFR_dt[index_max_time_exp]-dFR_dt[index_max_time_exp-1])/(time_sciantix[index_max_time_exp-1]-time_sciantix[index_max_time_exp-2])
						
						elif index_max_time_exp> 2 and time_sciantix[index_max_time_exp] != time_sciantix[index_max_time_exp-1] and time_sciantix[index_max_time_exp-1] == time_sciantix[index_max_time_exp-2]:
							dFR_dt[index_max_time_exp] = (FR_interpolated[index_max_time_exp]-FR_interpolated[index_max_time_exp-1])/(time_sciantix[index_max_time_exp]-time_sciantix[index_max_time_exp-1])
							dFR_dt[index_max_time_exp-1] = (FR_interpolated[index_max_time_exp-2] - FR_interpolated[index_max_time_exp-3])/(time_sciantix[index_max_time_exp-2]-time_sciantix[index_max_time_exp-3])
							dFR_dtdt[index_max_time_exp] = (dFR_dt[index_max_time_exp]-dFR_dt[index_max_time_exp-1])/(time_sciantix[index_max_time_exp]-time_sciantix[index_max_time_exp-1])
						else:
							first_derivative = np.gradient(self.interpolated_previous[:,0], self.output_previous[:,0])
							second_derivative = np.gradient(first_derivative, self.output_previous[:,0])

							for j in range(len(first_derivative)):
								if np.isnan(first_derivative[::-1])[j] == False:
									dFR_dt[0] = first_derivative[::-1][j]
									break
							for k in range(len(second_derivative)):
								if np.isnan(second_derivative[::-1])[k] == False:
									dFR_dtdt[0] = first_derivative[::-1][k]
									break
				
				state,temperature_state_start, temperature_state_end = plateauIdentify(self.time_history_original,self.temperature_history_original,time_sciantix[i])
				if time_sciantix[i] == time_sciantix[i-1]:
					FR_interpolated[i] = FR_interpolated[i-1]
					RR_interpolated[i] = RR_interpolated[i-1]
					dFR_dt[i] = dFR_dt[i-1]
					dFR_dtdt[i] = dFR_dtdt[i-1]
				else:
					if state == "increase" and FR_interpolated[i-1]<1:
						index_state_end = findClosestIndex_1D(self.temperature_exp, temperature_state_end)
						RR_interpolated[i] = interpolate_1D(self.temperature_exp[:index_state_end+1], self.RR_exp[:index_state_end+1], temperature_sciantix[i])
						if RR_interpolated[i] <0:
							RR_interpolated[i] = 0
						FR_interpolated[i] = FR_interpolated[i-1] + RR_interpolated[i] * (time_sciantix[i]-time_sciantix[i-1]) * 3600/Helium_total
						if FR_interpolated[i] > 1:
							FR_interpolated[i] =1
							RR_interpolated[i] = 0

					elif state == "plateau" and FR_interpolated[i-1] < 1:    

						plateau_index.append(i)
						FR_interpolated[i] = FR_interpolated[i-1] + dFR_dt[i-1] * (time_sciantix[i]-time_sciantix[i-1]) + 0.5 * dFR_dtdt[i-1] * (time_sciantix[i]-time_sciantix[i-1])**2
						RR_interpolated[i] = (FR_interpolated[i] - FR_interpolated[i-1])*Helium_total/(time_sciantix[i]-time_sciantix[i-1])/3600

						if FR_interpolated[i] >= 1:
							FR_interpolated[i] = 1
							RR_interpolated[i] = 0

					elif state == "decrease" and FR_interpolated[i-1] < 1:
						index_state_start = len(self.temperature_exp) - findClosestIndex_1D(self.temperature_exp[::-1], temperature_state_start) -1
						RR_interpolated[i] = interpolate_1D(self.temperature_exp[index_state_start:],self.RR_exp[index_state_start:], temperature_sciantix[i])
						if RR_interpolated[i] < 0:
							RR_interpolated[i] = 0
						FR_interpolated[i] = FR_interpolated[i-1] + RR_interpolated[i] * (time_sciantix[i]-time_sciantix[i-1]) * 3600/Helium_total
						if FR_interpolated[i] >= 1:
							FR_interpolated[i] = 1
							RR_interpolated[i] = 0

					else:
						FR_interpolated[i] = 1
						RR_interpolated[i] = 0

					dFR_dt[i] = (FR_interpolated[i] - FR_interpolated[i-1])/(time_sciantix[i]-time_sciantix[i-1])
					dFR_dtdt[i] = (dFR_dt[i] - dFR_dt[i-1])/(time_sciantix[i]-time_sciantix[i-1])
			
			dFR_dt = moving_average(dFR_dt,20)
			FR_interpolated = moving_average(FR_interpolated, 10)
			
			self.FR = FR_interpolated
			self.RR = RR_interpolated
			self.time = time_sciantix
			# print(self.time)
			self.temperature = temperature_sciantix
			# writing 

			data = np.column_stack((self.FR, self.RR))
			with open("interpolated_data.txt",'w') as file:
				for i, row in enumerate(data):
					file.write(f"{row[0]}\t{row[1]}")
					if i < len(data) -1 :
						file.write("\n")

			# error function
			error_related = np.zeros_like(FR_interpolated)
			# error =  - max(abs(FR_sciantix - FR_interpolated))
			for i in range(len(FR_interpolated)):
				if FR_interpolated[i] != 0:
					error_related[i] = abs(FR_sciantix[i] - FR_interpolated[i])/FR_interpolated[i]
				else:
					if FR_sciantix[i] == 0:
						error_related[i] = 1
					else:
						error_related[i] = 0
			error = - sum(error_related)
			# error function
			# # error_slop = -max(abs(dFR_dt - dFR_dt_sciantix))
			# print(dFR_dt)
			# print(dFR_dt_sciantix)
			# error_derivative = np.zeros_like(dFR_dt_sciantix)
			# dFR_dt_last = 0
			# for j in range(len(dFR_dt_sciantix)):
			# 	if np.isnan(dFR_dt_sciantix)[j] == False and np.isnan(dFR_dt)[j] ==  False:
			# 		error_derivative[j] = dFR_dt[j] - dFR_dt_sciantix[j]
			# 	else:
			# 		error_derivative[j] = 0
			# 		break
			# for i in range(len(dFR_dt)):
			# 	if error_derivative[::-1][i] > 10^-9:
			# 		error_derivative_last = error_derivative[i]
			# 		dFR_dt_last = dFR_dt[i]
			# 		break
			
			#########################################################################
			# if RR_sciantix[-1] < 0:
			# 	RR_sciantix[-1] = RR_sciantix[-2]
			# if self.time_start != 0:
			# 	if self.output_previous[-1,3] < 0:
			# 		self.output_previous[-1,3] = self.output_previous[-2,3]
			# 	RR_sciantix[0] = self.output_previous[-1,3]
			

			# FR_interpolated_low = FR_interpolated * (0.98)
			# FR_interpolated_up = FR_interpolated * (1.02)
			# dFR_dt_mean =np.average(dFR_dt[~np.isnan(dFR_dt)])
			# error_related = np.zeros_like(FR_sciantix)
			# if self.time_start == 0:
			# 	for i in range(len(FR_sciantix)):
			# 		if FR_sciantix[i] > FR_interpolated_up[i] or FR_sciantix[i] < FR_interpolated_low[i]:
			# 			error_related[i] = 10 * abs((FR_sciantix[i] - FR_interpolated[i]))
			# 		elif FR_sciantix[i] < FR_interpolated_up[i] and FR_sciantix[i] > FR_interpolated_low[i]:
			# 			if max(dFR_dt_sciantix) - min(dFR_dt_sciantix) > 0.05*dFR_dt_mean:
			# 				error_related[i] = 5 * abs(FR_interpolated[i] - FR_sciantix[i])
			# 			else:
			# 				error_related[i] = abs(FR_interpolated[i] - FR_sciantix[i])
			# 	error = -sum(error_related)
			
			
			# else:
			# 	if FR_sciantix[-1] > FR_interpolated[-1] and dFR_dt_sciantix[-1] > dFR_dt[~np.isnan(dFR_dt)][-1]:
			# 		error = -15 * abs((FR_sciantix[-1] - FR_interpolated[-1]))
				
			# 	elif FR_sciantix[-1] > FR_interpolated[-1] and dFR_dt_sciantix[-1] < dFR_dt[~np.isnan(dFR_dt)][-1]:
			# 		error = -10 * abs((FR_sciantix[-1] - FR_interpolated[-1]))

			# 	elif FR_sciantix[-1] <= FR_interpolated[-1]:
			# 		if FR_sciantix[-1] > FR_interpolated_low[-1]:
			# 			if (dFR_dt_sciantix[1] - self.output_previous[-1,3]*3600/Helium_total)/(self.output_previous[-1,3]*3600/Helium_total )> 0.05:
			# 				error = -12 * abs(FR_interpolated[-1] - FR_sciantix[-1])
			# 			else:
			# 				error= -abs(FR_interpolated[-1] - FR_sciantix[-1])
			# 		elif FR_sciantix[-1] < FR_interpolated_low[-1]:
			# 			error = -10 * abs((FR_sciantix[-1] - FR_interpolated[-1]))


			
			return error
		


		



		
		#####
		# domain reduction
		#####
		error_limit = - 0.1 
		current_error = -0.15 
		bound_size = np.zeros(len(self.sf_selected))

		for i in range(len(self.sf_selected)):
			bound_size[i] = new_bounds[self.sf_selected[i]][1] - new_bounds[self.sf_selected[i]][0]

		current_iteration = 0
		while np.round((current_error),2) < error_limit and current_iteration < 5:
			# pbounds = self.BO_bounds
			minimum_window = 0.01
			bounds_transformer = SequentialDomainReductionTransformer(minimum_window=minimum_window)
			self.bounds = {}
			if min(bound_size) > minimum_window:
				optimizer = BayesianOptimization(
					f = costFunction, 
					# pbounds  = pbounds,
					pbounds = new_bounds,
					verbose = 2,
					bounds_transformer = bounds_transformer,
					allow_duplicate_points=True
					)
				acq_function = UtilityFunction(kind = 'ucb')
				optimizer.maximize(
					init_points=20,
					n_iter=30,
					acquisition_function = acq_function
				)
				bounds = bounds_transformer.bounds[-1]
				for i in range(len(self.sf_selected)):
					self.bounds[self.sf_selected[i]] = bounds[i,:]
				
			else:
				optimizer = BayesianOptimization(
					f = costFunction, 
					# pbounds  = pbounds,
					pbounds = new_bounds,
					verbose = 2,
					allow_duplicate_points=True
					)
				acq_function = UtilityFunction(kind = 'ucb')
				optimizer.maximize(
					init_points=20,
					n_iter=30,
					acquisition_function = acq_function
				)
				# bounds = np.zeros((len(self.sf_selected, 2)))
				# for i in range(len(self.sf_selected)):
				# 	bounds[i,:] = new_bounds[self.sf_selected[i]]
				self.bounds =  new_bounds
			current_iteration = current_iteration + 1
			interpolated_data = np.genfromtxt("interpolated_data.txt",dtype = 'float',delimiter='\t')
			FR_interpolated = interpolated_data[:,0]
			current_error = optimizer.max['target']
			if self.time_start == 0:
				error_limit = current_error - 1
			else:
				# error_limit = - 0.1 * np.average(FR_interpolated)
				error_limit = - 0.05 * len(FR_interpolated)
			print(error_limit, current_error)
		


		for i, res in enumerate(optimizer.res):
			print("Iteration {}: \n\t{}".format(i, res))

		self.optimization_results = np.zeros(len(self.sf_selected)+1)
		for i in range(len(self.sf_selected)):
			self.scaling_factors[self.sf_selected[i]] = optimizer.max['params'][self.sf_selected[i]]
			if self.sf_selected[i] == "helium diffusivity pre exponential":
				self.optimization_results[i] = np.exp(optimizer.max['params'][self.sf_selected[i]])
			elif self.sf_selected[i] == "henry constant pre exponential":
				self.optimization_results[i] = np.exp(optimizer.max['params'][self.sf_selected[i]])
			else:
				self.optimization_results[i] = optimizer.max['params'][self.sf_selected[i]]
			print(f"{self.sf_selected[i]}:{self.optimization_results[i]}")
		print(f"Final error:{optimizer.max['target']}")
		self.optimization_results[-1] = optimizer.max['target']


		
		with open("input_scaling_factors.txt",'w') as file:
			for key, value in self.scaling_factors.items():
				if(f'{key}' == 'helium diffusivity pre exponential'):
					file.write(f'{np.exp(value)}\n')
					file.write(f'# scaling factor - {key}\n')
				elif(f'{key}' == 'henry constant pre exponential'):
					file.write(f'{np.exp(value)}\n')
					file.write(f'# scaling factor - {key}\n')
				else:
					file.write(f'{value}\n')
					file.write(f'# scaling factor - {key}\n')	
		
		# sciantix simulation with optimized sf
		os.system("./sciantix.x")

		self.final_data = getSelectedVariablesValueFromOutput(np.array(["Time (h)","Temperature (K)","He fractional release (/)", "He release rate (at/m3 s)"]),"output.txt")
		self.final_data[:,0] = self.final_data[:,0]+self.time_start
		final_interpolated = np.zeros_like(self.final_data)
		final_interpolated[:,0:2] = self.final_data[:,0:2]
		final_interpolated[:,2] = self.FR
		final_interpolated[:,3] = self.RR
		self.final_interpolated = final_interpolated
		
		os.chdir('../..')


# helpful functions

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

def readICfromInputInitialConditions(keyword):
	with open("input_initial_conditions.txt", 'r') as file:
		line_number = 0
		previous_line = None
		for line in file:
			line_number += 1
			if keyword in line:
				break 
			previous_line = line.strip() 
	if previous_line is not None:
		values = [float(val) for val in previous_line.split('\t')]
		ic_origin = np.array(values)
	return ic_origin

def moving_average(data, window_size):
	half_window = window_size // 2
	smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='same')
	smoothed_data[:half_window] = smoothed_data[half_window]
	smoothed_data[-half_window:] = smoothed_data[-half_window]
	return smoothed_data

def findClosestIndex_1D(source_data, targetValue):
	"""
	find the index of the value in source_data that closest to targetValue
	"""
	differences = [abs(x - targetValue) for x in source_data]
	index = int(np.argmin(differences))
	return index

def findClosedIndex_2D(source_data1, source_data2, targetElement1, targetElement2):
	differences1 = [abs(x - targetElement1) for x in source_data1]
	differences2 = [abs(y - targetElement2) for y in source_data2]

	differences = np.array(differences1)/np.max(differences1) + np.array(differences2)/np.max(differences2)
	index = np.argmin(differences)
	return index

def interpolate_1D(source_data_x, source_data_y, inserted_x):
	"""
	inserted_x has to be within the range of source_data_x
	
	"""
	up_bound = max(source_data_x)
	low_bound = min(source_data_x)
	if inserted_x > up_bound or inserted_x < low_bound:
		raise ValueError("interpolated_1D: inserted_x is out of the source_data_x range")
		
	difference = source_data_x - inserted_x
	index_min_difference = np.argmin(abs(difference))
	if difference[index_min_difference] == 0:
		interpolated_value = source_data_y[index_min_difference]
	
	else:
		if inserted_x > source_data_x[index_min_difference]:
			index_low = index_min_difference
			index_up_collection = np.where(source_data_x > inserted_x)[0]
			index_up = index_up_collection[np.argmin(abs(index_up_collection - index_min_difference))]

		else:# inserted_x < source_data_x[index_min_difference]:

			index_up = index_min_difference
			index_low_collection = np.where(source_data_x < inserted_x)[0]
			index_low = index_low_collection[np.argmin(abs(index_low_collection - index_min_difference))]
			
		low_x = source_data_x[index_low]
		low_y = source_data_y[index_low]
		up_x = source_data_x[index_up]
		up_y = source_data_y[index_up]
		slop = (up_y - low_y)/(up_x-low_x)
		interpolated_value = low_y + slop*(inserted_x - low_x)
	return interpolated_value

def interpolate_2D(source_data_x, source_data_y, inserted_x, inserted_y):

	differences1 = np.array([abs((x - inserted_x)/inserted_x) for x in source_data_x])
	index_x = np.where(differences1<0.02)[0]
	source_data_y_x = source_data_y[index_x]
	if len(index_x) != 0:
		if inserted_y == 0:
			value_interpolated = np.average(source_data_y_x)
		else:    
			differences2 = np.array([abs((y - inserted_y)/inserted_y) for y in source_data_y_x])
			index_y = np.where(differences2<0.02)[0]
			if len(index_y) == 0:
				index_y = np.argmin(differences2)
			index = index_x[index_y]
			value_interpolated = np.average(source_data_y[index])
	else:
		value_interpolated = source_data_y[np.argmin(differences1)]


	return value_interpolated
	
def plateauIdentify(time_history, temperature_history, time):

	index_low = np.where(time_history <= time)[0][-1]

	if time < time_history[-1]:
		index_up = np.where(time_history > time)[0][0]
	else:
		index_up = len(time_history) - 1

	difference = temperature_history[index_up] - temperature_history[index_low]

	if difference == 0:
		state = "plateau"
	elif difference < 0:
		state = "decrease"
	else:
		state = "increase"
	temperature_state_start = temperature_history[index_low]
	temperature_state_end = temperature_history[index_up]

	return state, temperature_state_start, temperature_state_end

def do_plot(Talip1320):
	os.chdir(Talip1320.current_directory)
	os.chdir("..")
	cloumnsFR  = np.genfromtxt("Talip2014_release_data.txt",dtype = 'float',delimiter='\t')
	cloumnsRR = np.genfromtxt("Talip2014_rrate_data.txt",dtype = 'float',delimiter='\t')
	variable_selected = np.array(["Time (h)","Temperature (K)","He fractional release (/)", "He release rate (at/m3 s)"])
	coloumnsOutput_nominal = getSelectedVariablesValueFromOutput(variable_selected,"output.txt")

	if Talip1320.time_start == 0:
		os.chdir(f"Optimization_{Talip1320.time_start}_{Talip1320.time_end}")
	else:
		os.chdir(f"Optimization_{Talip1320.time_start}__{Talip1320.time_end}")
	coloumnOutput_new = getSelectedVariablesValueFromOutput(variable_selected,"output.txt")

	time_exp  = cloumnsFR[:,0]
	FR_exp = cloumnsFR[:,1]
	temperature_exp = cloumnsRR[:,0]
	RR_exp = cloumnsRR[:,1]

	time_sciantix = coloumnsOutput_nominal[:,0]
	temperature_sciantix = coloumnsOutput_nominal[:,1]
	FR_nominal = coloumnsOutput_nominal[:,2]
	RR_nominal = coloumnsOutput_nominal[:,3]

	FR_interpolated = Talip1320.FR
	RR_interpolated = Talip1320.RR

	time_new = coloumnOutput_new[:,0]+Talip1320.time_start
	temperature_new = coloumnOutput_new[:,1]
	FR_new = coloumnOutput_new[:,2]
	RR_new = coloumnOutput_new[:,3]

	fig, ax = plt.subplots(1,2)
	plt.subplots_adjust(left=0.1,
						bottom=0.1,
						right=0.9,
						top=0.9,
						wspace=0.34,
						hspace=0.4)

	ax[0].scatter(time_exp, FR_exp, marker = '.', c = '#B3B3B3', label='Data from Talip et al. (2014)')
	ax[0].scatter(time_sciantix, FR_nominal,marker = 'x' ,color = '#98E18D', label='SCIANTIX 2.0 Nominal')
	ax[0].scatter(time_new, FR_new, marker = 'x',color = 'red',label = f'optimized_{Talip1320.time_start}_{Talip1320.time_end}')
	ax[0].scatter(time_new, FR_interpolated, marker = 'x',color = 'blue',label = 'interpolated')
	axT = ax[0].twinx()
	axT.set_ylabel('Temperature (K)')
	axT.plot(time_sciantix, temperature_sciantix, 'r', linewidth=1, label="Temperature")

	ax[0].set_xlabel('Time (h)')
	ax[0].set_ylabel('Helium fractional release (/)')
	h1, l1 = ax[0].get_legend_handles_labels()
	h2, l2 = axT.get_legend_handles_labels()
	ax[0].legend(h1+h2, l1+l2)
	ax[0].legend(loc = 'upper left')

	""" Plot: Helium release rate """
	ax[1].scatter(temperature_exp, RR_exp, marker = '.', c = '#B3B3B3', label='Data from Talip et al. (2014)')
	ax[1].scatter(temperature_sciantix, RR_nominal, marker = 'x',color = '#98E18D', label='SCIANTIX 2.0 Nominal')
	ax[1].scatter(temperature_new, RR_new, marker = 'x', color = 'red',label = f'optimized_{Talip1320.time_start}_{Talip1320.time_end}')
	ax[1].scatter(temperature_new, RR_interpolated, marker = 'x', color = 'blue',label = 'interpolated')

	ax[1].set_xlabel('Temperature (K)')
	ax[1].set_ylabel('Helium release rate (at m${}^{-3}$ s${}^{-1}$)')
	ax[1].legend()

	# plt.savefig(file + '.png')
	plt.show()
	
	os.chdir("../..")

# ONLINE optimization
#####################

# start = 0
# end = 0.744
# num_steps = 10
# ref_points = np.linspace(start, end, num_steps).reshape(-1, 1).round(2)


ref_points = np.array([[0], [0.2], [0.3], [0.4], [0.45], [0.5], [0.55], [0.6], [0.65],[0.7],[0.75],[0.8]])
ref_case = "test_Talip2014_1600K"
time_points = ref_points
number_of_interval = len(time_points) - 1

sf_number = 4
sf_optimized = np.ones((number_of_interval+1,sf_number))
error_optimized = np.zeros((number_of_interval+1,1))
results_data = np.empty((number_of_interval+2,sf_number+2),dtype = object)
final_data = np.empty((0,4))
final_data_interpolated = np.empty((0,4))
bounds_information = np.zeros((number_of_interval+2,2*sf_number), dtype = object)

sf_selected = np.array([
	"helium diffusivity pre exponential",
	"helium diffusivity activation energy",
	"henry constant pre exponential",
	"henry constant activation energy"])
initial_bounds = {}

for i in range(len(sf_selected)):
	if sf_selected[i] == "helium diffusivity pre exponential":
		initial_bounds[sf_selected[i]] = (np.log(0.05), np.log(19.9))

	elif sf_selected[i] == "helium diffusivity activation energy":
		# self.sf_selected_bounds[0,i] = 0.835
		# self.sf_selected_bounds[1,i] = 1.2
		initial_bounds[sf_selected[i]] = (0.835, 1.2)
	elif sf_selected[i] == "henry constant pre exponential":
		# self.sf_selected_bounds[0,i] = np.log(0.0627)
		# self.sf_selected_bounds[1,i] = np.log(16.09)
		initial_bounds[sf_selected[i]] = (np.log(0.0627),np.log(16.09))
	elif sf_selected[i] == "henry constant activation energy":
		# self.sf_selected_bounds[0,i] = 0.431
		# self.sf_selected_bounds[1,i] = 1.55
		initial_bounds[sf_selected[i]] = (0.431,1.55)
	else:
		initial_bounds[sf_selected[i]] = (0.0,float('inf'))
# print(new_bounds)
new_bounds = initial_bounds

for m in range(len(sf_selected)):
	bounds_information[0, 2 * m] = sf_selected[m]
	bounds_information[1,2*m] = initial_bounds[sf_selected[m]][0]
	bounds_information[1,2*m+1] = initial_bounds[sf_selected[m]][1]


results_data[0,0] = "time"
results_data[0,1:(sf_number+1)] = sf_selected
results_data[0,(sf_number+1)] = "error"
results_data[1,:] = [0, 1.0, 1.0, 1.0, 1.0, 0]

Talip1320 = optimization()
Talip1320.setCase(ref_case)
Talip1320.setStartEndTime(time_points[0][0], time_points[1][0])
Talip1320.setInitialConditions()
Talip1320.setScalingFactors(
	sf_selected[0],
	sf_selected[1],
	sf_selected[2],
	sf_selected[3]
)
setInputOutput = inputOutput()

Talip1320.optimization(setInputOutput, new_bounds)
results_data[2,1:] = Talip1320.optimization_results
results_data[2,0] = time_points[1][0]

for i in range(2,number_of_interval+1):
	Talip1320 = optimization()
	Talip1320.setCase(ref_case)
	
	Talip1320.setStartEndTime(time_points[i-1][0],time_points[i][0])
	# print(time_points[i-1][0])
	Talip1320.setInitialConditions()
	Talip1320.setScalingFactors(
		sf_selected[0],
		sf_selected[1],
		sf_selected[2],
		sf_selected[3]
	)
	setInputOutput = inputOutput()
	Talip1320.optimization(setInputOutput, new_bounds)
	
	results_data[i+1,1:] = Talip1320.optimization_results
	results_data[i+1,0] = time_points[i][0]
	print(abs(results_data[i+1,(sf_number+1)]),abs(results_data[i, (sf_number+1)]))
	while abs(results_data[i+1,(sf_number+1)]) > abs(results_data[i, (sf_number+1)]):

		for k in range(len(Talip1320.sf_selected)):

			bound_low = max( new_bounds[Talip1320.sf_selected[k]][0] - 0.1* abs(new_bounds[Talip1320.sf_selected[k]][0]),  initial_bounds[Talip1320.sf_selected[k]][0])
			bound_up = min( new_bounds[Talip1320.sf_selected[k]][1] + 0.1* abs(new_bounds[Talip1320.sf_selected[k]][1]),  initial_bounds[Talip1320.sf_selected[k]][1])
			print(f"bound_low:{bound_low}, bound_up:{bound_up}")
			new_bounds[Talip1320.sf_selected[k]] = (bound_low, bound_up)
		
		Talip1320 = optimization()
		Talip1320.setCase(ref_case)
		
		Talip1320.setStartEndTime(time_points[i-1][0],time_points[i][0])
		# print(time_points[i-1][0])
		Talip1320.setInitialConditions()
		Talip1320.setScalingFactors(
			sf_selected[0],
			sf_selected[1],
			sf_selected[2],
			sf_selected[3]
		)
		setInputOutput = inputOutput()
		Talip1320.optimization(setInputOutput, new_bounds)
		results_data[i+1,1:] = Talip1320.optimization_results
		print(f"previous time interval error:{abs(results_data[i, (sf_number+1)])}, current: {abs(results_data[i+1,(sf_number+1)])}")
	
	new_bounds = Talip1320.bounds
	with open(f"optimization_online.txt", 'w') as file:
		for row in results_data:
			line = "\t".join(map(str, row))
			file.write(line + "\n")
	for j in range(len(Talip1320.sf_selected)):
		index_bound = np.where(bounds_information == Talip1320.sf_selected[j])[1][0]
		bounds_information[i+1,index_bound] = new_bounds[Talip1320.sf_selected[j]][0]
		bounds_information[i+1,index_bound + 1] = new_bounds[Talip1320.sf_selected[j]][1]

	with open('bounds_information.txt','w') as file:
		for row in bounds_information:
			line = "\t".join(map(str,row))
			file.write(line + "\n")

	final_data = np.vstack((final_data, Talip1320.final_data))
	final_data_interpolated = np.vstack((final_data_interpolated, Talip1320.final_interpolated))

######################
# OFFLINE optimization
######################

# ref_points = np.array([[0],[3.867]])
# time_points = ref_points
# number_of_interval = len(time_points) - 1

# sf_optimized = np.ones((number_of_interval+1,sf_number))
# error_optimized = np.zeros((number_of_interval+1,1))
# results_data = np.empty((number_of_interval+2,sf_number+2),dtype = object)

# for i in range(1,number_of_interval+1):

# 	Talip1320 = optimization()
# 	Talip1320.setCase(ref_case)

# 	Talip1320.setStartEndTime(0,time_points[i][0])
# 	Talip1320.setInitialConditions()
# 	Talip1320.setScalingFactors(
# 		# "resolution rate",
# 		# "trapping rate",
# 		"helium diffusivity pre exponential",
# 		"helium diffusivity activation energy",
# 		"henry constant pre exponential",
# 		"henry constant activation energy"
# 	)
# 	setInputOutput = inputOutput()
# 	Talip1320.optimization(setInputOutput)
# 	results_data[i+1,1:] = Talip1320.optimization_results
# 	results_data[i+1,0] = time_points[i][0]

# results_data[0,0] = "time"
# results_data[0,1:(sf_number+1)] = Talip1320.sf_selected
# results_data[0,(sf_number+1)] = "error"
# results_data[1,:] = [0, 1.0, 1.0, 1.0, 1.0, 0]

# with open(f"optimization_offline.txt", 'w') as file:
# 	for row in results_data:
# 		line = "\t".join(map(str, row))
# 		file.write(line + "\n")
