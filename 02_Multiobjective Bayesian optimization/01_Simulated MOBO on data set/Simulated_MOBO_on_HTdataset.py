"""imports"""
import os
import torch
import numpy as np
import pandas as pd
import time
from datetime import datetime

from sklearn import preprocessing
from botorch.models.multitask import MultiTaskGP, FixedNoiseMultiTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch import fit_gpytorch_model
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

#%%
print(f"Script started at {datetime.now()}")
start = datetime.now()

#%%
"""parameter definition"""
start = time.time()
dtype = torch.double
init_random_points = 5
ref_point = [550.0, 10]
percentile_top = 75
percentile_bottom = 25
runs = 1

"""set current directory to location of .py file"""
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

"""load data and skip first row"""
HT_data_import = pd.read_table("HTsummary_Co-Mn-Sb-Sn-Ti-gridsearch.txt")

"""define relevant property column names"""
elements = ["Co", "Mn", "Sb", "Sn", "Ti"]
properties = ["Overpotential", "Overpotential change"]
std = ["Overpotential_std", "Overpotential change_std"]

"""normalize compositions"""
HT_data_import[elements] = HT_data_import[elements].div(100, axis=0)

"""negate properties. MOBO need maximization as objective"""
HT_data_import[properties] = -HT_data_import[properties]
ref_point = (pd.Series(ref_point)*-1).tolist()

"""pick columns that are relevant and treat NaN"""
HT_data = HT_data_import[elements + properties + std]
HT_data[std] = HT_data[std].fillna(0.001)

#%%
"""compute total hypervolume based on all data points (need to plot normalized hypervolume vs. iteration)"""
hv = Hypervolume(ref_point = torch.tensor(ref_point))
y = HT_data[properties].values.reshape(-1,2)
y = torch.tensor(y)
pareto_mask = is_non_dominated(y)
pareto_y = y[pareto_mask]

"""compute hypervolume"""
total_hypervolume = hv.compute(pareto_y)

#%%
"""multiobj Bayesopt: qNEHVI + MultiTaskGP"""

hypervolume_MultiTaskGP = pd.DataFrame()

for j in range(runs):
    print(f"#############################################################\n run MultiTaskGP: {j} \n #############################################################")

    """random initialization"""
    HT_data_init = HT_data.sample(n = init_random_points, random_state = j)
    
    """generate dataframe with remaining candidates"""
    HT_data_remain = pd.concat([HT_data, HT_data_init])
    HT_data_remain = HT_data_remain.drop_duplicates(keep = False)
    HT_data_remain = HT_data_remain.reset_index(drop = True)
    HT_data_remain_comp = HT_data_remain[elements]
    
    """generate dataframe with iteration counter"""
    HT_data_init_iteration = HT_data_init.copy()
    HT_data_init_iteration["iteration"] = 0

    """define data for MultiTask"""
    HT_data_init_MultiTask = HT_data_init.copy()
    HT_data_remain_MultiTask = HT_data_remain.copy()
    HT_data_remain_comp_MultiTask = HT_data_remain_comp.copy()
    HT_data_init_iteration_MultiTask = HT_data_init_iteration.copy()
    
    hypervolume = []
    for i in range(0, len(HT_data)-init_random_points):
        
        print(f"cycle: {i}")
        
        """calculate current hypervolume and its ratio to total hypervolume"""
        y = HT_data_init_MultiTask[properties].values.reshape(-1,2)
        y = torch.tensor(y)
        pareto_mask = is_non_dominated(y)
        pareto_y = y[pareto_mask]
        volume = hv.compute(pareto_y)
        norm_volume = volume/total_hypervolume
        
        if i == 0: #for first cycles append initial volume as many times as you have initial points
            for v in range(0,len(y)):  
                hypervolume.append(norm_volume )
        else:
            hypervolume.append(norm_volume)
        
        """define x"""
        x_init = torch.tensor(np.reshape(np.array(HT_data_init_MultiTask[elements]), (-1,len(elements))), dtype = dtype)
        x_init_multitask = torch.cat([torch.cat([x_init, torch.zeros(x_init.size()[0],1)], -1), torch.cat([x_init, torch.ones(x_init.size()[0],1)], -1)]) #build tensor so that it suites MultiTaskGP
    
        """define and standardize y"""
        y_init = HT_data_init_MultiTask[properties].values.reshape(-1,2)
        scaler = preprocessing.StandardScaler().fit(y_init)
        y_init = scaler.transform(y_init)
        y_init = torch.tensor(y_init, dtype = dtype).view(-1,2)
        y_init = torch.cat((y_init[:, 0], y_init[:, 1])).unsqueeze(-1)
    
        """std of y values"""
        y_init_std = HT_data_init_MultiTask[std].values.reshape(-1,2)
        y_init_std = y_init_std / scaler.scale_
        y_init_std = torch.tensor(y_init_std, dtype= dtype).view(-1,2)
        
        """"build model and fitting"""
        model = MultiTaskGP(x_init_multitask, y_init, task_feature = -1)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        
        """define aquisition function"""
        ref_point_MultiTask = scaler.transform(np.array(ref_point).reshape(-1,2)) # transform ref point using the same scaler as for y
        ref_point_MultiTask = ref_point_MultiTask.flatten().tolist() # make ref_point back into a list so that its compatible with aquisition function
        X_baseline = x_init
        qNEHVI = qNoisyExpectedHypervolumeImprovement(model, ref_point = ref_point_MultiTask, X_baseline = X_baseline)
        
        """get aquisition values for the remaining points not sampled"""
        HT_data_remain_comp_MultiTask = torch.tensor(np.reshape(np.array(HT_data_remain_comp_MultiTask), (-1,1, len(elements))), dtype = dtype)
    
        aquisition = qNEHVI(HT_data_remain_comp_MultiTask)
        new_candidate_index = int(aquisition.argmax().detach().numpy())
        
        """get new candidate from HT_data_remain and merge it with HT_init"""
        new_candidate = HT_data_remain_MultiTask.iloc[new_candidate_index]
        HT_data_init_MultiTask = pd.concat([HT_data_init_MultiTask, new_candidate.to_frame().T], ignore_index = True)
        
        new_candidate["iteration"] = i + 1
        HT_data_init_iteration_MultiTask = pd.concat([HT_data_init_iteration_MultiTask, new_candidate.to_frame().T], ignore_index = True)
        
        """delete new dandiate from HT_data_remain"""
        HT_data_remain_MultiTask = HT_data_remain_MultiTask.drop([new_candidate_index])
        HT_data_remain_MultiTask = HT_data_remain_MultiTask.reset_index(drop=True)
        HT_data_remain_comp_MultiTask = HT_data_remain_MultiTask[elements]
    
    """calculate last hypervolume for last iteration and its ratio to total hypervolume"""
    y = HT_data_init_MultiTask[properties].values.reshape(-1,2)
    y = torch.tensor(y)
    pareto_mask = is_non_dominated(y)
    pareto_y = y[pareto_mask]
    volume = hv.compute(pareto_y)
    norm_volume = volume/total_hypervolume
    hypervolume.append(norm_volume)
    
    """convert negated values back"""
    HT_data_init_iteration_MultiTask[properties] = HT_data_init_iteration_MultiTask[properties]*-1
    
    """insert normalized hypervolume column"""
    hypervolume_MultiTaskGP = pd.concat((hypervolume_MultiTaskGP, pd.Series(hypervolume, name=str(j))),axis = 1)
    
    """convert comp back to 100%"""
    HT_data_init_iteration_MultiTask[elements] = HT_data_init_iteration_MultiTask[elements]*100


#%%
"""multiobj Bayesopt: qNEHVI + FixedNoiseMultiTaskGP"""

hypervolume_FixedNoiseMultiTaskGP = pd.DataFrame()

for j in range(runs):
    print(f"#############################################################\n run FixedNoiseMultiTaskGP: {j} \n #############################################################")
    
    """random initialization"""
    HT_data_init = HT_data.sample(n = init_random_points, random_state = j)
    
    """generate dataframe with remaining candidates"""
    HT_data_remain = pd.concat([HT_data, HT_data_init])
    HT_data_remain = HT_data_remain.drop_duplicates(keep = False)
    HT_data_remain = HT_data_remain.reset_index(drop = True)
    HT_data_remain_comp = HT_data_remain[elements]
    
    """generate dataframe with iteration counter"""
    HT_data_init_iteration = HT_data_init.copy()
    HT_data_init_iteration["iteration"] = 0

    """define data for FixedNoiseMultiTask"""
    HT_data_init_FixedNoiseMultiTask = HT_data_init.copy()
    HT_data_remain_FixedNoiseMultiTask = HT_data_remain.copy()
    HT_data_remain_comp_FixedNoiseMultiTask = HT_data_remain_comp.copy()
    HT_data_init_iteration_FixedNoiseMultiTask = HT_data_init_iteration.copy()
    
    hypervolume = []
    for i in range(0, len(HT_data)-init_random_points):
        
        print(f"cycle: {i}")
        
        """calculate current hypervolume and its ratio to total hypervolume"""
        y = HT_data_init_FixedNoiseMultiTask[properties].values.reshape(-1,2)
        y = torch.tensor(y)
        pareto_mask = is_non_dominated(y)
        pareto_y = y[pareto_mask]
        volume = hv.compute(pareto_y)
        norm_volume = volume/total_hypervolume
        
        if i == 0: #for first cycles append initial volume as many times as you have initial points
            for v in range(0,len(y)):  
                hypervolume.append(norm_volume )
        else:
            hypervolume.append(norm_volume)
        
        """define x"""
        x_init = torch.tensor(np.reshape(np.array(HT_data_init_FixedNoiseMultiTask[elements]), (-1,len(elements))), dtype = dtype)
        x_init_multitask = torch.cat([torch.cat([x_init, torch.zeros(x_init.size()[0],1)], -1), torch.cat([x_init, torch.ones(x_init.size()[0],1)], -1)]) #build tensor so that it suites MultiTaskGP
    
        """define and standardize y"""
        y_init = HT_data_init_FixedNoiseMultiTask[properties].values.reshape(-1,2)
        scaler = preprocessing.StandardScaler().fit(y_init)
        y_init = scaler.transform(y_init)
        y_init = torch.tensor(y_init, dtype = dtype).view(-1,2)
        y_init = torch.cat((y_init[:, 0], y_init[:, 1])).unsqueeze(-1)
    
        """std of y values"""
        y_init_std = HT_data_init_FixedNoiseMultiTask[std].values.reshape(-1,2)
        y_init_std = y_init_std / scaler.scale_
        y_init_std = torch.tensor(y_init_std, dtype= dtype).view(-1,2)
        y_init_std = torch.cat((y_init_std[:, 0], y_init_std[:, 1])).unsqueeze(-1)
        
        """"build model and fitting"""
        model = FixedNoiseMultiTaskGP(x_init_multitask, y_init, train_Yvar = y_init_std, task_feature = -1)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        
        """define aquisition function"""
        ref_point_FixedNoiseMultiTask = scaler.transform(np.array(ref_point).reshape(-1,2)) # transform ref point using the same scaler as for y
        ref_point_FixedNoiseMultiTask = ref_point_FixedNoiseMultiTask.flatten().tolist() # make ref_point back into a list so that its compatible with aquisition function
        X_baseline = x_init
        qNEHVI = qNoisyExpectedHypervolumeImprovement(model, ref_point = ref_point_FixedNoiseMultiTask, X_baseline = X_baseline)
        
        """get aquisition values for the remaining points not sampled"""
        HT_data_remain_comp_FixedNoiseMultiTask = torch.tensor(np.reshape(np.array(HT_data_remain_comp_FixedNoiseMultiTask), (-1,1, len(elements))), dtype = dtype)
    
        aquisition = qNEHVI(HT_data_remain_comp_FixedNoiseMultiTask)
        new_candidate_index = int(aquisition.argmax().detach().numpy())
        
        """get new candidate from HT_data_remain and merge it with HT_init"""
        new_candidate = HT_data_remain_FixedNoiseMultiTask.iloc[new_candidate_index]
        HT_data_init_FixedNoiseMultiTask = pd.concat([HT_data_init_FixedNoiseMultiTask, new_candidate.to_frame().T], ignore_index = True)
        
        new_candidate["iteration"] = i + 1
        HT_data_init_iteration_FixedNoiseMultiTask = pd.concat([HT_data_init_iteration_FixedNoiseMultiTask, new_candidate.to_frame().T], ignore_index = True)
        
        """delete new dandiate from HT_data_remain"""
        HT_data_remain_FixedNoiseMultiTask = HT_data_remain_FixedNoiseMultiTask.drop([new_candidate_index])
        HT_data_remain_FixedNoiseMultiTask = HT_data_remain_FixedNoiseMultiTask.reset_index(drop=True)
        HT_data_remain_comp_FixedNoiseMultiTask = HT_data_remain_FixedNoiseMultiTask[elements]
    
    """calculate last hypervolume for last iteration and its ratio to total hypervolume"""
    y = HT_data_init_FixedNoiseMultiTask[properties].values.reshape(-1,2)
    y = torch.tensor(y)
    pareto_mask = is_non_dominated(y)
    pareto_y = y[pareto_mask]
    volume = hv.compute(pareto_y)
    norm_volume = volume/total_hypervolume
    hypervolume.append(norm_volume)
    
    """convert negated values back"""
    HT_data_init_iteration_FixedNoiseMultiTask[properties] = HT_data_init_iteration_FixedNoiseMultiTask[properties]*-1
    
    """insert normalized hypervolume column"""
    hypervolume_FixedNoiseMultiTaskGP = pd.concat((hypervolume_FixedNoiseMultiTaskGP, pd.Series(hypervolume, name=str(j))),axis = 1)
    
    """convert comp back to 100%"""
    HT_data_init_iteration_FixedNoiseMultiTask[elements] = HT_data_init_iteration_FixedNoiseMultiTask[elements]*100

    
#%%
"""Random sampling as benchmark"""

hypervolume_random = pd.DataFrame()

for j in range(runs):
    print(f"#############################################################\n run Random: {j} \n #############################################################")
       
    """random initialization"""
    HT_data_init = HT_data.sample(n = init_random_points, random_state = j)
    
    """generate dataframe with remaining candidates"""
    HT_data_remain = pd.concat([HT_data, HT_data_init])
    HT_data_remain = HT_data_remain.drop_duplicates(keep = False)
    HT_data_remain = HT_data_remain.reset_index(drop = True)
    HT_data_remain_comp = HT_data_remain[elements]
    
    """generate dataframe with iteration counter"""
    HT_data_init_iteration = HT_data_init.copy()
    HT_data_init_iteration["iteration"] = 0

    """define data for Random"""
    HT_data_init_Random = HT_data_init.copy()
    HT_data_remain_Random = HT_data_remain.copy()
    HT_data_init_iteration_Random = HT_data_init_iteration.copy()
    
    hypervolume = []
    for i in range(0, len(HT_data)-init_random_points):
            
        print(f"cycle: {i}")
        
        """calculate current hypervolume and its ratio to total hypervolume"""
        y = HT_data_init_Random[properties].values.reshape(-1,2)
        y = torch.tensor(y)
        pareto_mask = is_non_dominated(y)
        pareto_y = y[pareto_mask]
        volume = hv.compute(pareto_y)
        norm_volume = volume/total_hypervolume
        
        if i == 0: #for first cycles append initial volume as many times as you have initial points
            for v in range(0,len(y)):  
                hypervolume.append(norm_volume )
        else:
            hypervolume.append(norm_volume)
        
        new_candidate = HT_data_remain_Random.sample(n = 1, random_state = j)
        new_candidate_index = new_candidate.index[0]
        HT_data_init_Random = pd.concat([HT_data_init_Random, new_candidate])
        new_candidate["iteration"] = i + 1
        HT_data_init_iteration_Random = pd.concat([HT_data_init_iteration_Random, new_candidate], ignore_index = True)
            
        """delete new candiate from HT_data_remain"""
        HT_data_remain_Random = HT_data_remain_Random.drop([new_candidate_index])
        HT_data_remain_Random = HT_data_remain_Random.reset_index(drop=True)
    
    """calculate last hypervolume for last iteration and its ratio to total hypervolume"""
    y = HT_data_init_Random[properties].values.reshape(-1,2)
    y = torch.tensor(y)
    pareto_mask = is_non_dominated(y)
    pareto_y = y[pareto_mask]
    volume = hv.compute(pareto_y)
    norm_volume = volume/total_hypervolume
    hypervolume.append(norm_volume)
    
    """insert normalized hypervolume column"""
    hypervolume_random = pd.concat((hypervolume_random, pd.Series(hypervolume, name=str(j))),axis = 1)
        
    """convert negated values back"""
    HT_data_init_iteration_Random[properties] = HT_data_init_iteration_Random[properties]*-1
    
#%%
"""get top and bottom percentile for all hypervolume improvements with MultiTaskGP. Also get mean between top and bottom percentile"""
hypervolume_summary_MultiTaskGP = []

for index, row in hypervolume_MultiTaskGP.iterrows():
    row_sort = row.sort_values(ascending=True, ignore_index=True)
    
    bottom = np.percentile(row, percentile_bottom)
    top = np.percentile(row, percentile_top)
    
    index_bottom = round(len(row_sort)*(percentile_bottom/100))
    index_top = round(len(row_sort)*(percentile_top/100))
    avg = row_sort[index_bottom:index_top].mean()
    
    hypervolume_summary_MultiTaskGP.append(list([avg, avg-bottom, top-avg]))

HT_data_init_iteration_MultiTask[["hypervolume_avg", "hypervolume_percentile_"+str(percentile_bottom), "hypervolume_percentile_"+str(percentile_top)]] = hypervolume_summary_MultiTaskGP
        

"""get top and bottom percentile for all hypervolume improvements with FixedNoiseMultiTaskGP. Also get mean between top and bottom percentile"""
hypervolume_summary_FixedNoiseMultiTaskGP = []

for index, row in hypervolume_FixedNoiseMultiTaskGP.iterrows():
    row_sort = row.sort_values(ascending=True, ignore_index=True)
    
    bottom = np.percentile(row, percentile_bottom)
    top = np.percentile(row, percentile_top)
    
    index_bottom = round(len(row_sort)*(percentile_bottom/100))
    index_top = round(len(row_sort)*(percentile_top/100))
    avg = row_sort[index_bottom:index_top].mean()
    hypervolume_summary_FixedNoiseMultiTaskGP.append(list([avg, avg-bottom, top-avg]))
    
HT_data_init_iteration_FixedNoiseMultiTask[["hypervolume_avg", "hypervolume_percentile_"+str(percentile_bottom), "hypervolume_percentile_"+str(percentile_top)]] = hypervolume_summary_FixedNoiseMultiTaskGP
    

"""get top and bottom percentile for all hypervolume improvements with Random. Also get mean between top and bottom percentile"""
hypervolume_summary_random = []

for index, row in hypervolume_random.iterrows():
    row_sort = row.sort_values(ascending=True, ignore_index=True)
    
    bottom = np.percentile(row, percentile_bottom)
    top = np.percentile(row, percentile_top)
    
    index_bottom = round(len(row_sort)*(percentile_bottom/100))
    index_top = round(len(row_sort)*(percentile_top/100))
    avg = row_sort[index_bottom:index_top].mean()
    
    hypervolume_summary_random.append(list([avg, avg-bottom, top-avg]))
    
HT_data_init_iteration_Random[["hypervolume_avg", "hypervolume_percentile_"+str(percentile_bottom), "hypervolume_percentile_"+str(percentile_top)]] = hypervolume_summary_random

#%%
"""construct output DataFrame"""
summary_MultiTaskGP = pd.concat([HT_data_init_iteration_MultiTask["iteration"], pd.DataFrame(hypervolume_summary_MultiTaskGP)], axis=1)
summary_MultiTaskGP.columns = ["iteration", "hypervolume_avg", "hypervolume_percentile_"+str(percentile_bottom), "hypervolume_percentile_"+str(percentile_top)]  

summary_FixedNoiseMultiTaskGP = pd.concat([HT_data_init_iteration_FixedNoiseMultiTask["iteration"], pd.DataFrame(hypervolume_summary_FixedNoiseMultiTaskGP)], axis=1)
summary_FixedNoiseMultiTaskGP.columns = ["iteration", "hypervolume_avg", "hypervolume_percentile_"+str(percentile_bottom), "hypervolume_percentile_"+str(percentile_top)] 

summary_Random = pd.concat([HT_data_init_iteration_Random["iteration"], pd.DataFrame(hypervolume_summary_random)], axis=1)
summary_Random.columns = ["iteration", "hypervolume_avg", "hypervolume_percentile_"+str(percentile_bottom), "hypervolume_percentile_"+str(percentile_top)]

#%%
"""save"""
summary_MultiTaskGP.to_csv("MOBO_simulated_MultiTaskGP.txt", index = False, sep = "\t")
summary_FixedNoiseMultiTaskGP.to_csv("MOBO_simulated_FixedNoiseMultiTaskGP.txt", index = False, sep = "\t")
summary_Random.to_csv("MOBO_simulated_Random.txt", index = False, sep = "\t")

#%%
print(f"Script ended at {datetime.now()}")
end = datetime.now()
print(f"Script excecution took {end - start}")
