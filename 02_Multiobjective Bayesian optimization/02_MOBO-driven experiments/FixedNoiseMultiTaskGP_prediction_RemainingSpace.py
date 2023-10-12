"""Imports"""
import os
from datetime import datetime
import torch
import numpy as np
import pandas as pd

from sklearn import preprocessing
from botorch.models.multitask import FixedNoiseMultiTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from sklearn.manifold import MDS     

#%%
print(f"Script started at {datetime.now()}")
start = datetime.now()

#%%
device = torch.device("cpu")
dtype = torch.double
torch.manual_seed(3)

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

"""pick columns that are relevant"""
HT_data = HT_data_import[elements + properties + std]

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

""""Predictions"""
posterior = model.posterior(torch.tensor(comp_space_reduced.div(100, axis=0).values, dtype=dtype))
y_pred, var = posterior.mean, posterior.variance
y_pred = scaler.inverse_transform(y_pred.detach().numpy())

#%%
"""catetgorize compositions into uniary, binary, ternary, quaternary, quinary"""
category_dict = {1 : "uniary", 2 : "binary", 3 : "ternary", 4 : "quaternary", 5 : "quinary"}
category = []

for index, row in comp_space_reduced.iterrows():
    counter = 0
    for element in row:
        if element != 0:
            counter += 1
    category.append(category_dict[counter])

print(pd.Series(category).value_counts(sort=False))

#%%
"""append results to total composition space"""
comp_space_result = comp_space_reduced
comp_space_result[["pred_overpotential", "pred_overpotential change"]] = y_pred
comp_space_result["category"] = category

#%%
"""Save rounded candidates"""
comp_space_result.to_csv("FixedNoiseMultiTaskGP_predictions_RemainingSpace.txt", index = False, sep = "\t")

#%%
print(f"Script ended at {datetime.now()}")
end = datetime.now()
print(f"Script excecution took  {end - start}")