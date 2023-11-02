import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.optimize import Bounds
import gc


class inputOutput():
    def __init__(self):
        pass

    def writeInputHistory(self,history_original, time_start, time_end):
        time_history_original = history_original[:,0]
        temperature_history_original = history_original[:,1]
        temperature_start = interpolate_1D(time_history_original, temperature_history_original, time_start)
        temperature_end = interpolate_1D(time_history_original, temperature_history_original, time_end)

        if time_start == 0:
            index_start = 0
        else:
            index_start = np.where(time_history_original < time_start)[0][-1]
        if time_end >= time_history_original[-1]:
            index_end = len(time_history_original)-1
        else:
            index_end = np.where(time_history_original > time_end)[0][0]
        
        new_history = history_original[index_start:index_end+1,:].copy()
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


    def writeInputInitialConditions(self, ic_new):
        with open("input_initial_conditions.txt", 'r') as file:
            lines = file.readlines()

        keyword = "#	initial He (at/m3) produced, intragranular, intragranular in solution, intragranular in bubbles, grain boundary, released"

        for i, line in enumerate(lines):
            if keyword in line:
                ic_str = '\t'.join(map(str, ic_new))
                lines[i - 1] = ic_str + '\n'  

                break
        with open("input_initial_conditions.txt", 'w') as file:
            file.writelines(lines)

    def writeInputScalingFactors(self,scaling_factors,sf_selected,sf_selected_value):
        # assign new sf value and write the file, and run sciantix.x
        for i in range(len(sf_selected)):
            scaling_factors[sf_selected[i]] = sf_selected_value[i]

        with open("input_scaling_factors.txt",'w') as file:
            for key, value in scaling_factors.items():
                file.write(f'{value}\n')
                file.write(f'# scaling factor - {key}\n')
        
        self.sf_selected_value = sf_selected_value

        os.system("./sciantix.x")
    
    def readOutput(self):
        # here, in Talip cases, we only interesed in FR and RR since we only have these two variables experimental data
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
        """
        caseName: string, for example: "test_Talip_1320K"
        """
        self.caseName = caseName
        #########
        #get exp data
        #########
        os.chdir(self.caseName)
        cloumnsFR  = np.genfromtxt("Talip2014_release_data.txt",dtype = 'float',delimiter='\t')
        cloumnsRR = np.genfromtxt("Talip2014_rrate_data.txt",dtype = 'float',delimiter='\t')
        history = np.genfromtxt("input_history.txt",dtype = 'float', delimiter='\t')

        self.time_exp  = cloumnsFR[:,0]
        self.FR_exp = cloumnsFR[:,1]
        self.temperature_exp = cloumnsRR[:,0]
        self.RR_exp = cloumnsRR[:,1]
        
        self.FR_smoothed = moving_average(self.FR_exp,100)
        
        self.history_original = history
        self.time_history_original = self.history_original[:,0]
        self.temperature_history_original = self.history_original[:,1]

    def setStartEndTime(self, time_start, time_end):
        self.time_start = time_start
        self.time_end = time_end
        # now build a new folder in selected benchmark folder
        if self.time_start !=0:
            folder_name = f"Optimization_{time_start}__{time_end}"
        else:
            folder_name = f"Optimization_{time_start}_{time_end}"
       
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
        with open("input_initial_conditions.txt", 'r') as file:
            keyword = "#	initial He (at/m3) produced"
            line_number = 0
            previous_line = None
            for line in file:
                line_number += 1
                if keyword in line:
                    break 
                previous_line = line.strip() 
        if previous_line is not None:
            values = [float(val) for val in previous_line.split('\t')]
            self.ic_origin = np.array(values)

        file.close()

        if self.time_start != 0:
            keyword1 = f"__{self.time_start}"
            # keyword2 = f"Optimization_0_{self.time_start}"
            keyword2 = f"0_{self.time_start}"
            parent_directory = os.path.dirname(current_directory)
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
                # variable_selected = np.array(["He produced (at/m3)","He in grain (at/m3)", "He in intragranular solution (at/m3)", "He in intragranular bubbles (at/m3)", "He at grain boundary (at/m3)","He released (at/m3)"])
                # variable_value = getSelectedVariablesValueFromOutput(variable_selected,"output.txt")
                # self.ic_new = variable_value[-1,:]
                # RR_ic = getSelectedVariablesValueFromOutput(np.array(["He release rate (at/m3 s)"]),"output.txt")[-1]
                # self.RR_ic = RR_ic[0]
                self.folder_path_previous_optimization = folder_path1

            else:
                os.chdir(folder_path2)
                # variable_selected = np.array(["He produced (at/m3)","He in grain (at/m3)", "He in intragranular solution (at/m3)", "He in intragranular bubbles (at/m3)", "He at grain boundary (at/m3)","He released (at/m3)"])
                # variable_value = getSelectedVariablesValueFromOutput(variable_selected,"output.txt")
                # self.ic_new = variable_value[-1,:]
                # RR_ic = getSelectedVariablesValueFromOutput(np.array(["He release rate (at/m3 s)"]),"output.txt")[-1]
                # self.RR_ic = RR_ic[0]
                self.folder_path_previous_optimization = folder_path2
            variable_selected = np.array(["He produced (at/m3)","He in grain (at/m3)", "He in intragranular solution (at/m3)", "He in intragranular bubbles (at/m3)", "He at grain boundary (at/m3)","He released (at/m3)"])
            variable_value = getSelectedVariablesValueFromOutput(variable_selected,"output.txt")
            self.ic_new = variable_value[-1,:]
            RR_ic = getSelectedVariablesValueFromOutput(np.array(["He release rate (at/m3 s)"]),"output.txt")[-1]
            self.RR_ic = RR_ic[0]

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
            


        else:
            self.ic_new = self.ic_origin
            self.RR_ic = 0
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
                self.sf_selected_bounds[0,i] = 0.05
                self.sf_selected_bounds[1,i] = 19.9
            elif self.sf_selected[i] == "helium diffusivity activation energy":
                self.sf_selected_bounds[0,i] = 0.835
                self.sf_selected_bounds[1,i] = 1.2
            elif self.sf_selected[i] == "henry constant pre exponential":
                self.sf_selected_bounds[0,i] = 0.0627
                self.sf_selected_bounds[1,i] = 16.09
            elif self.sf_selected[i] == "henry constant activation energy":
                self.sf_selected_bounds[0,i] = 0.431
                self.sf_selected_bounds[1,i] = 1.55
            else:
                self.sf_selected_bounds[0,i] = 0
                self.sf_selected_bounds[0,i] = float('inf')
        self.bounds = Bounds(self.sf_selected_bounds[0,:],self.sf_selected_bounds[1,:])

    def optimization(self,inputOutput):
        inputOutput.writeInputHistory(self.history_original, self.time_start, self.time_end)
        inputOutput.writeInputInitialConditions(self.ic_new)
        def costFunction(sf_selected_value):
            inputOutput.writeInputScalingFactors(self.scaling_factors,self.sf_selected,sf_selected_value)
            inputOutput.readOutput()
            #####
            #match data
            ####
            FR = np.zeros_like(inputOutput.FR_sciantix)
            RR = np.zeros_like(inputOutput.RR_sciantix)
            RR_fr = np.zeros_like(inputOutput.RR_sciantix)
            time_sciantix = inputOutput.time_sciantix + self.time_start
            temperature_sciantix = inputOutput.temperature_sciantix
            FR_sciantix = inputOutput.FR_sciantix
            RR_sciantix = inputOutput.RR_sciantix

            Helium_total = self.ic_new[0]
            #############
            #interpolate: inserted time_sciantix -----> FR and RR 
            #############

            #region1 : time_sciantix < max(time_exp)
            FR[0] = self.ic_new[-1]/self.ic_new[0]
            RR[0] = self.RR_ic

            if time_sciantix[0] > max(self.time_exp):
                index_max_time_exp = 0
            else:
                index_max_time_exp = findClosestIndex_1D(time_sciantix, max(self.time_exp))
                
                if time_sciantix[index_max_time_exp] > max(self.time_exp):
                        index_max_time_exp = index_max_time_exp - 1

                for i in range(1,index_max_time_exp + 1):
                    if time_sciantix[i] == time_sciantix[i-1]:
                        FR[i] = FR[i-1]
                        RR_fr[i] = RR_fr[i-1]
                        RR[i] = RR[i-1]
                    else:
                        FR[i] = interpolate_1D(self.time_exp, self.FR_smoothed, time_sciantix[i])
                        RR_fr[i] =(FR[i]-FR[i-1])/(time_sciantix[i]-time_sciantix[i-1])/3600*Helium_total
                        RR[i] = interpolate_2D(self.temperature_exp, self.RR_exp, temperature_sciantix[i], RR_fr[i])

            # region2 : 
            for i in range(index_max_time_exp+1,len(time_sciantix)):
                state,temperature_state_start, temperature_state_end = plateauIdentify(self.time_history_original,self.temperature_history_original,time_sciantix[i])
                # print(self.time_history_original)
                # print(i,state)
                if state == "increase" and FR[i-1]<1:
                    index_state_end = findClosestIndex_1D(self.temperature_exp, temperature_state_end)
                    RR[i] = interpolate_1D(self.temperature_exp[:index_state_end], self.RR_exp[:index_state_end], temperature_sciantix[i])
                    FR[i] = FR[i-1] + RR[i] * (time_sciantix[i]-time_sciantix[i-1]) * 3600/Helium_total
                    if FR[i] > 1:
                        FR[i] =1
                        RR[i] = 0
                    # print(i,FR[i])
                elif state == "plateau" and FR[i-1] < 1:
                    slop = (FR[i-1] - FR[i-3])/(time_sciantix[i-1]-time_sciantix[i-3])
                    time_saturation = time_sciantix[i-1] + (1-FR[i-1])/slop
                    if time_sciantix[i] < time_saturation:
                        FR[i] = FR[i-1] + (time_sciantix[i]-time_sciantix[i-1])*slop
                        RR[i] = FR[i-1]
                    else:
                        FR[i] = 1
                        RR[i] = 0
                    # print(i,FR[i])
                elif state == "decrease" and FR[i-1] < 1:
                    index_state_start = len(self.temperature_exp) - findClosestIndex_1D(self.temperature_exp[::-1], temperature_state_start) -1
                    RR[i] = interpolate_1D(self.temperature_exp[index_state_start:],self.RR_exp[index_state_start:], temperature_sciantix[i])
                    
                    FR[i] = FR[i-1] + RR[i] * (time_sciantix[i]-time_sciantix[i-1]) * 3600/Helium_total
                    if FR[i] > 1:
                        FR[i] = 1
                        RR[i] = 0
                else: #FR[i-1] = 1
                    FR[i] = 1
                    RR[i] = 0
            
            self.FR = FR
            self.RR = RR


            current_sf_selected_value = sf_selected_value
            print(f"current value: {current_sf_selected_value}")
            error_related = np.zeros_like(FR_sciantix)

            for i in range(len(FR_sciantix)):
                if FR[i] == 0 and FR_sciantix[i] == 0:
                    error_related[i] = 0
                elif FR[i] == 0 and FR_sciantix[i] !=0:
                    error_related[i] = FR_sciantix[i]
                else:
                    # error_related[i] = abs((FR[i]-FR_sciantix[i])/FR[i])+abs(RR[i]-RR_sciantix[i]/max(RR))
                    error_related[i] = abs((FR[i]-FR_sciantix[i])/FR[i])
            error = np.sum(error_related)
            print(f"current error: {error}")
            return error
        
        results = optimize.minimize(costFunction,self.sf_selected_initial_value, method = 'SLSQP',bounds=self.bounds)

        self.optimization_results = np.zeros(len(self.sf_selected)+1)
        for i in range(len(self.sf_selected)):
            self.scaling_factors[self.sf_selected[i]] = results.x[i]
            self.optimization_results[i] = results.x[i]
            print(f"{self.sf_selected[i]}:{results.x[i]}")
        print(f"Final error:{results.fun}")
        self.optimization_results[-1] = results.fun
        

        with open("input_scaling_factors.txt",'w') as file:
            for key, value in self.scaling_factors.items():
                file.write(f'{value}\n')
                file.write(f'# scaling factor - {key}\n')
        os.system("./sciantix.x")




        variable_selected = np.array(["Time (h)","Temperature (K)","He fractional release (/)", "He release rate (at/m3 s)"])
        self.final_data = getSelectedVariablesValueFromOutput(variable_selected,"output.txt")
        self.final_data[:,0] = self.final_data[:,0]+self.time_start
        final_interpolated = np.zeros_like(self.final_data)
        final_interpolated[:,0:2] = self.final_data[:,0:2]
        final_interpolated[:,2] = self.FR
        final_interpolated[:,3] = self.RR
        self.final_interpolated = final_interpolated

        os.chdir('..')
        os.chdir('..')
        return results


##########
#functions
##########
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
    differences = np.array([(x - inserted_x) for x in source_data_x])

    index = np.argmin(abs(differences))
    # print(index,inserted_x,source_data_x[index])
    if differences[index] == 0 or index == len(source_data_x)-1 or index == 0:
        value_interpolated = source_data_y[index]
    
    else:
        for i in range(min(index, len(source_data_x)-index-1)):
            if np.sign(differences[index - i]) != np.sign(differences[index+i]):
                index_low = index-i
                index_up = index+i
                up_y = source_data_y[index_up]
                low_y = source_data_y[index_low]
                up_x = source_data_x[index_up]
                low_x = source_data_x[index_low]
                if up_x == low_x:
                    value_interpolated = source_data_y[index]
                else:
                    slop = (up_y - low_y)/(up_x-low_x)
                    value_interpolated = low_y + slop * (inserted_x - low_x)
                break;
            else:
                value_interpolated = source_data_y[index]
                break;

    return value_interpolated

def interpolate_2D(source_data_x, source_data_y, inserted_x, inserted_y):

    differences1 = np.array([abs((x - inserted_x)/inserted_x) for x in source_data_x])
    index_x = np.where(differences1<0.02)[0]
    source_data_y_x = source_data_y[index_x]
    if inserted_y == 0:
        value_interpolated = np.average(source_data_y_x)
    else:    
        differences2 = np.array([abs((y - inserted_y)/inserted_y) for y in source_data_y_x])
        index_y = np.where(differences2<0.02)[0]
        if len(index_y) == 0:
            index_y = np.argmin(differences2)
        index = index_x[index_y]
        value_interpolated = np.average(source_data_y[index])

    return value_interpolated
    
def plateauIdentify(time_history, temperature_history, time):

    index_low = np.where(time_history <= time)[0][-1]

    if time < time_history[-1]:
        index_up = np.where(time_history > time)[0][0]
    else:
        index_up = len(time_history) - 1

    difference = temperature_history[index_up] - temperature_history[index_low]
    # print(time,difference,temperature_history[index_up], temperature_history[index_low], time_history[index_low], time_history[index_up] )
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
    history = np.genfromtxt("input_history.txt")
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
    # ax[0].legend(h1+h2, l1+l2)
    ax[0].legend(loc = 'upper left')

    """ Plot: Helium release rate """
    ax[1].scatter(temperature_exp, RR_exp, marker = '.', c = '#B3B3B3', label='Data from Talip et al. (2014)')
    ax[1].scatter(temperature_sciantix, RR_nominal, marker = 'x',color = '#98E18D', label='SCIANTIX 2.0 Nominal')
    ax[1].scatter(temperature_new, RR_new, marker = 'x', color = 'red',label = f'optimized_{Talip1320.time_start}_{Talip1320.time_end}')
    ax[1].scatter(temperature_new, RR_interpolated, marker = 'x', color = 'blue',label = 'interpolated')


    # ax.set_title(file + ' - Release rate')
    ax[1].set_xlabel('Temperature (K)')
    ax[1].set_ylabel('Helium release rate (at m${}^{-3}$ s${}^{-1}$)')
    ax[1].legend()

    # plt.savefig(file + '.png')
    plt.show()
    
    os.chdir("..")
    os.chdir("..")





####
#optim in a pecifiy interval
#####

Talip1320 = optimization()
Talip1320.setCase("test_Talip2014_1320K")
Talip1320.setStartEndTime(0,0.3)
Talip1320.setInitialConditions()
Talip1320.setScalingFactors("helium diffusivity pre exponential", "helium diffusivity activation energy","henry constant pre exponential","henry constant activation energy")
setInputOutput = inputOutput()
Talip1320.optimization(setInputOutput)
do_plot(Talip1320)

Talip1320 = optimization()
Talip1320.setCase("test_Talip2014_1320K")
Talip1320.setStartEndTime(0.3, 0.6)
Talip1320.setInitialConditions()
Talip1320.setScalingFactors("helium diffusivity pre exponential", "helium diffusivity activation energy","henry constant pre exponential","henry constant activation energy")
setInputOutput = inputOutput()
Talip1320.optimization(setInputOutput)
do_plot(Talip1320)

Talip1320 = optimization()
Talip1320.setCase("test_Talip2014_1320K")
Talip1320.setStartEndTime(0.6, 0.9)
Talip1320.setInitialConditions()
Talip1320.setScalingFactors("helium diffusivity pre exponential", "helium diffusivity activation energy","henry constant pre exponential","henry constant activation energy")
setInputOutput = inputOutput()
Talip1320.optimization(setInputOutput)
do_plot(Talip1320)


# ######
# #online optimization-------testing....
# ######
# t0 = 0
# number_of_interval = 20
# time_points = np.zeros((number_of_interval+1,1))
# sf_optimized = np.ones((number_of_interval+1,4))
# error_optimized = np.zeros((number_of_interval+1,1))
# results_data = np.empty((number_of_interval+2,6),dtype = object)
# final_data = []
# final_data_interpolated = []
# for i in range(1,number_of_interval+1):

#     Talip1320 = optimization()
#     Talip1320.setCase("test_Talip2014_1320K")

#     tf = Talip1320.time_history_original[-1]
#     time_points[i] = time_points[i-1]+(tf-t0)/number_of_interval
#     time_points[i] = np.round(time_points[i],3)
#     Talip1320.setStartEndTime(time_points[i-1][0],time_points[i][0])
#     Talip1320.setInitialConditions()
#     Talip1320.setScalingFactors("helium diffusivity pre exponential", "helium diffusivity activation energy","henry constant pre exponential","henry constant activation energy")
#     setInputOutput = inputOutput()
#     Talip1320.optimization(setInputOutput)
#     results_data[i+1,1:] = Talip1320.optimization_results
#     results_data[i+1,0] = time_points[i][0]

#     # do_plot(Talip1320)
#     for row in Talip1320.final_data:
#         final_data = np.concatenate((final_data, [row]), axis=0)



#     for row in Talip1320.final_interpolated:
#         final_data_interpolated = np.concatenate((final_data_interpolated, [row]), axis=0)
#     final_data = np.array(final_data,dtype = float)
#     final_data_interpolated = np.array(final_data_interpolated, dtype = float)


# results_data[0,0] = "time"
# results_data[0,1:5] = Talip1320.sf_selected
# results_data[0,5] = "error"
# results_data[1,:] = [0,1.0,1.0,1.0,1.0,0]


# with open(f"optimization_{Talip1320.caseName}_{number_of_interval}.txt", 'w') as file:
#     for row in results_data:
#         line = "\t".join(map(str, row))
#         file.write(line + "\n")

# os.chdir("test_Talip2014_1320K")
# cloumnsFR  = np.genfromtxt("Talip2014_release_data.txt",dtype = 'float',delimiter='\t')
# cloumnsRR = np.genfromtxt("Talip2014_rrate_data.txt",dtype = 'float',delimiter='\t')
# variable_selected = np.array(["Time (h)","Temperature (K)","He fractional release (/)", "He release rate (at/m3 s)"])
# coloumnsOutput_nominal = getSelectedVariablesValueFromOutput(variable_selected,"output.txt")

# time_exp  = cloumnsFR[:,0]
# FR_exp = cloumnsFR[:,1]
# temperature_exp = cloumnsRR[:,0]
# RR_exp = cloumnsRR[:,1]

# time_sciantix = coloumnsOutput_nominal[:,0]
# temperature_sciantix = coloumnsOutput_nominal[:,1]
# FR_nominal = coloumnsOutput_nominal[:,2]
# RR_nominal = coloumnsOutput_nominal[:,3]

# FR_interpolated = final_data_interpolated[:,2]
# RR_interpolated = final_data_interpolated[:,3]


# time_new = final_data[:,0]
# temperature_new = final_data[:,1]
# FR_new = final_data[:,2]
# RR_new = final_data[:,3]

# fig, ax = plt.subplots(1,2)
# plt.subplots_adjust(left=0.1,
#                     bottom=0.1,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.34,
#                     hspace=0.4)

# ax[0].scatter(time_exp, FR_exp, marker = '.', c = '#B3B3B3', label='Data from Talip et al. (2014)')
# ax[0].scatter(time_sciantix, FR_nominal,marker = 'x' ,color = '#98E18D', label='SCIANTIX 2.0 Nominal')
# ax[0].scatter(time_new, FR_new, marker = 'x',color = 'red',label = f'optimized_{Talip1320.time_start}_{Talip1320.time_end}')
# ax[0].scatter(time_new, FR_interpolated, marker = 'x',color = 'blue',label = 'interpolated')
# axT = ax[0].twinx()
# axT.set_ylabel('Temperature (K)')
# axT.plot(time_sciantix, temperature_sciantix, 'r', linewidth=1, label="Temperature")

# ax[0].set_xlabel('Time (h)')
# ax[0].set_ylabel('Helium fractional release (/)')
# h1, l1 = ax[0].get_legend_handles_labels()
# h2, l2 = axT.get_legend_handles_labels()
# # ax[0].legend(h1+h2, l1+l2)
# ax[0].legend(loc = 'upper left')

# """ Plot: Helium release rate """
# ax[1].scatter(temperature_exp, RR_exp, marker = '.', c = '#B3B3B3', label='Data from Talip et al. (2014)')
# ax[1].scatter(temperature_sciantix, RR_nominal, marker = 'x',color = '#98E18D', label='SCIANTIX 2.0 Nominal')
# ax[1].scatter(temperature_new, RR_new, marker = 'x', color = 'red',label = f'optimized_{Talip1320.time_start}_{Talip1320.time_end}')
# ax[1].scatter(temperature_new, RR_interpolated, marker = 'x', color = 'blue',label = 'interpolated')


# # ax.set_title(file + ' - Release rate')
# ax[1].set_xlabel('Temperature (K)')
# ax[1].set_ylabel('Helium release rate (at m${}^{-3}$ s${}^{-1}$)')
# ax[1].legend()

# # plt.savefig(file + '.png')
# plt.show()

# os.chdir("..")
# os.chdir("..")