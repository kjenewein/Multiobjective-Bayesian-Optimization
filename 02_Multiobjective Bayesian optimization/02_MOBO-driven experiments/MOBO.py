"""Imports"""
import os
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.spatial import distance

from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP, HeteroskedasticSingleTaskGP
from botorch.models.multitask import MultiTaskGP, FixedNoiseMultiTaskGP

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf

#%%
print(f"Script started at {datetime.now()}")
start = datetime.now()

#%%
device = torch.device("cpu")
dtype = torch.double
torch.manual_seed(3)

new_candidates = 15
ref_point = [550, 10]  #for qNEHVI [activity goal, stability goal]

"""define relevant property column names"""
elements = ["Co", "Mn", "Sb", "Sn", "Ti"]

properties = ["Overpotential", "Overpotential change"]
std = ["Overpotential_std", "Overpotential change_std"]

#%%
"""set current directory to location of .py file"""
dir_path = os.path.dirname(os.path.realpath("__file__"))
os.chdir(dir_path)
path_data = dir_path + "\\HT_Data_MOBO"

"""load total composition space (10% spread)"""
comp_space_total = pd.read_table("Composition_space_5element_10%.txt")

"""load HT data and skip first row / merge all current HT_summaries together"""
files = [f for f in os.listdir(path_data)]
HT_data_import = []

if len(files) > 1: #concatenate if more than 1 HT file detected
    for enum,f in enumerate(files):
        if f.endswith(".txt"):
            temp_data = pd.read_table(path_data + "\\" + f, skiprows = [1])
            temp_data["Iteration"] = enum
            HT_data_import.append(temp_data)
    HT_data_import = pd.concat(HT_data_import) # concatenate all data frames
    HT_data_import = HT_data_import.reset_index(drop=True)
    
    """save merged HT_summary files so that you can plot progress so far"""
    HT_data_import = HT_data_import.drop(columns=["Index"])
    now = datetime.now()
    HT_data_import.to_csv(dir_path + "\\MOBO_summary_" + now.strftime("%Y%m%d_%H-%M-%S") + ".txt", index = True, sep = "\t")
else:
    HT_data_import = pd.read_table(path_data + "\\" + files[0], skiprows = [1]) # if i only have one file saving progress doesnt make sense

"""normalize compositions"""
HT_data_import[elements] = HT_data_import[elements].div(100, axis=0)

"""negate properties. MOBO maximized properties so lager values must mean better"""
HT_data_import[properties] = -HT_data_import[properties]
ref_point = (pd.Series(ref_point)*-1).tolist()

"""pick columns that are relevant"""
HT_data = HT_data_import[elements + properties + std]
HT_data[std] = HT_data[std].fillna(0) # convert NaN to 0

#%%
"""take out HT data compositions from total compositions"""
index_to_drop = []
for index, row in HT_data[elements].iterrows():
    index_to_drop.append(int(np.where((np.array(comp_space_total/100) == np.array(row)).all(axis = 1))[0]))

comp_space_reduced = comp_space_total.drop(index = index_to_drop).reset_index(drop = True)

#%%
"""Multiobj Bayesopt: qNEHVI + FixedNoiseMultiTaskGP"""

"""define x"""
x_init = torch.tensor(np.reshape(np.array(HT_data[elements]), (-1,len(elements))), dtype = dtype)
x_init_multitask = torch.cat([torch.cat([x_init, torch.zeros(x_init.size()[0],1)], -1), torch.cat([x_init, torch.ones(x_init.size()[0],1)], -1)]) #build tensor so that it suites MultiTaskGP

"""define and standardize y"""
y_init = HT_data[properties].values.reshape(-1,2)
scaler = preprocessing.StandardScaler().fit(y_init)
y_init = scaler.transform(y_init)
y_init = torch.tensor(y_init, dtype = dtype).view(-1,2)
y_init = torch.cat((y_init[:, 0], y_init[:, 1])).unsqueeze(-1)

"""std of y values"""
y_init_std = HT_data[std].values.reshape(-1,2)
y_init_std = y_init_std / scaler.scale_
y_init_std = torch.tensor(y_init_std, dtype= dtype).view(-1,2)
y_init_std = torch.cat((y_init_std[:, 0], y_init_std[:, 1])).unsqueeze(-1)

""""build model and fitting"""
model = FixedNoiseMultiTaskGP(x_init_multitask, y_init, train_Yvar = y_init_std, task_feature = -1)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)

"""define aquisition function"""
ref_point = scaler.transform(np.array(ref_point).reshape(-1,2)) # transform ref point using the same scaler as for y
ref_point = ref_point.flatten().tolist() # make ref_point back into a list so that its compatible with aquisition function
X_baseline = x_init
acq_function = qNoisyExpectedHypervolumeImprovement(model, ref_point = ref_point, X_baseline = X_baseline)

#%%
"""optimize aquisition function and get new candidates"""
bounds = torch.cat([torch.zeros(x_init.size()[1]), torch.ones(x_init.size()[1])]).view(2,-1)
equality_constraint = [(torch.tensor([0,1,2,3,4]), torch.tensor([1,1,1,1,1], dtype = torch.float32), 1)]
candidates, _ = optimize_acqf(
                acq_function = acq_function,
                bounds = bounds,
                equality_constraints = equality_constraint,
                q = new_candidates,
                sequential = True,
                num_restarts = 300,
                raw_samples = 1024,
                options = {"batch_limit" : 5, "maxiter" : 400}
                )

"""treat new candidates: convert to numpy, make to 100% scale, pick closest composition in entire reduced comp space"""
candidates = candidates.detach().numpy() * 100

#create empty DataFrame
candidates_selection = pd.DataFrame(columns = elements, index = [*range(0,new_candidates,1)], dtype = "int64")
for index,i in enumerate(candidates):
    #calculate euclidean distance of each new candidate to every possible (remaining) composition in space
    dist = []
    for j in np.array(comp_space_reduced):
        dist.append(distance.euclidean(i, j))
    dist = pd.Series(dist)
    
    #get index of closest composition in reduced comp space and put into candidates_selection DataFrame
    index_closest_composition = dist.idxmin()
    candidates_selection.iloc[index] = comp_space_reduced.iloc[index_closest_composition]
        
    # # drop the composition you took from the reduced comp space (so you cant sample this point anymore) 
    comp_space_reduced = comp_space_reduced.drop(index = index_closest_composition).reset_index(drop = True)

#%%
"""move row with most Co to first"""
Co_max_index = candidates_selection["Co"].idxmax()

# reorder index by bringing Co_max_index up
new_idx = [Co_max_index] + [i for i in range(len(candidates_selection)) if i != Co_max_index]
candidates_selection = candidates_selection.iloc[new_idx].reset_index(drop=True)

#%%
"""Save rounded candidates"""
now = datetime.now()
candidates = pd.DataFrame(candidates, columns = elements)
candidates.to_csv(dir_path + "\\selection_MOBO" + "\\MOBO_candidates_original_" + now.strftime("%Y%m%d_%H-%M-%S") + ".txt", index = False, sep = "\t")
candidates_selection.to_csv(dir_path + "\\selection_MOBO" + "\\MOBO_candidates_" + now.strftime("%Y%m%d_%H-%M-%S") + ".txt", index = False, sep = "\t")

#%%
print(f"Script ended at {datetime.now()}")
end = datetime.now()
print(f"Script excecution took {end - start}")
