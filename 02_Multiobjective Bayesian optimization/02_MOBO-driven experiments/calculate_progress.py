"""
calculate following metrics:
    - normalized hypervolume per index
"""

#%%
"""Imports"""
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

#%%
print(f"Script started at {datetime.now()}")
start = datetime.now()

#%%
"""Parameters"""
ref_point = [550.0, 10.0]
top_x_percent = 10 # for tracking MOBO progress

elements = ["Co", "Mn", "Sb", "Sn", "Ti"]
activity = "Overpotential"
stability = "Overpotential change"
properties = [activity, stability]
std = ["Overpotential_std", "Overpotential change_std"]

data_file_MOBO = "MOBO_summary_20230617_16-21-38.txt"
data_file_random = "MOBO_summary_random_20230617_16-13-29.txt"
#%%
"""set current directory to location of .py file"""
dir_path = os.path.dirname(os.path.realpath("__file__"))
os.chdir(dir_path)

"""import data"""
data_MOBO =  pd.read_table(data_file_MOBO)
data_random = pd.read_table(data_file_random)

#%%
"""pre calculations that are used for both MOBO and random"""

"""total hypervolume"""
data_concat = pd.concat([data_MOBO, data_random])

hv = Hypervolume(ref_point = torch.tensor((pd.Series(ref_point)*-1).tolist()))
y_total = -data_concat[properties].values.reshape(-1,2)
y_total = torch.tensor(y_total)
pareto_mask_total = is_non_dominated(y_total)
pareto_y_total = y_total[pareto_mask_total]
total_hypervolume = hv.compute(pareto_y_total)

#%%
"""Calculations for MOBO"""
"""hypervolume"""
#iterate through data and calculate norm. hypervolume per data point
norm_hypervolume_MOBO = []
for i in range(len(data_MOBO)):
    #iterate through data and keep adding next row
    y = -data_MOBO[0:i][properties].astype(float).values.reshape(-1,2)
    y = torch.tensor(y)
    pareto_mask = is_non_dominated(y)
    pareto_y = y[pareto_mask]
    volume = hv.compute(pareto_y)
    norm_volume = volume/total_hypervolume
    norm_hypervolume_MOBO.append(norm_volume)

#%%
"""add calculation to data"""
data_MOBO["Pareto?"] = pareto_mask_total.detach().numpy()[0:len(data_MOBO)]
data_MOBO["norm. hypervolume"] = norm_hypervolume_MOBO

#%%
"""Calculations for Random"""

"""hypervolume"""
#iterate through data and calculate norm. hypervolume per data point"""
norm_hypervolume_random = []
for i in range(len(data_random)):
    #iterate through data and keep adding next row
    y = -data_random[0:i][properties].astype(float).values.reshape(-1,2)
    y = torch.tensor(y)
    pareto_mask = is_non_dominated(y)
    pareto_y = y[pareto_mask]
    volume = hv.compute(pareto_y)
    norm_volume = volume/total_hypervolume
    norm_hypervolume_random.append(norm_volume)

"""add calculation to data"""
data_random["Pareto?"] = pareto_mask_total.detach().numpy()[len(data_MOBO):]
data_random["norm. hypervolume"] = norm_hypervolume_random

#%%
"""export data"""
data_MOBO.to_csv(dir_path + "\\Progress_MOBO\\" + data_file_MOBO.split(".")[0] + "_Progress.txt", index = False, sep = "\t")
data_random.to_csv(dir_path + "\\Progress_Random\\" + data_file_random.split(".")[0] + "_Progress.txt", index = False, sep = "\t")

"""export pareto compositions for MOBO"""
MOBO_pareto_composition = data_MOBO[data_MOBO["Pareto?"] == True][elements]
MOBO_pareto_composition.to_csv(dir_path + "\\Progress_MOBO\\" + data_file_MOBO.split(".")[0] + "_Progress_Pareto_Compositions.txt", index = False, sep = "\t")

"""export pareto compositions for MOBO with activity stability values"""
MOBO_pareto_composition = data_MOBO[data_MOBO["Pareto?"] == True][["Composition"] + elements + properties + std]
name = MOBO_pareto_composition["Composition"].squeeze().tolist()
name = [i.replace('_', '\-(') for i in name]
name = [i.replace(';', ')') for i in name]
MOBO_pareto_composition["Composition"] = name

MOBO_pareto_composition.to_csv(dir_path + "\\Progress_MOBO\\" + data_file_MOBO.split(".")[0] + "_Progress_Pareto_Compositions with values.txt", index = False, sep = "\t")

#%%
print(f"Script ended at {datetime.now()}")
end = datetime.now()
print(f"Script excecution took {end - start}")
