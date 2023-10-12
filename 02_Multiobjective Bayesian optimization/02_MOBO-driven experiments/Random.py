"""Imports"""
import os
from datetime import datetime
import pandas as pd
import numpy as np


#%%
print(f"Script started at {datetime.now()}")
start = datetime.now()
#%%

"""Parameters"""
random_candidates = 15
random_state = 1

"""define relevant property column names"""
elements = ["Co", "Mn", "Sb", "Sn", "Ti"]

#%%
"""set current directory to location of .py file"""
dir_path = os.path.dirname(os.path.realpath("__file__"))
os.chdir(dir_path)
path_data = dir_path + "\\HT_Data_random"

"""load data with entire composition space"""
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
    HT_data_import.to_csv(dir_path + "\\MOBO_summary_random_" + now.strftime("%Y%m%d_%H-%M-%S") + ".txt", index = True, sep = "\t")
else:
    HT_data_import = pd.read_table(path_data + "\\" + files[0], skiprows = [1]) # if i only have one file saving progress doesnt make sense


"""pick columns that are relevant"""
HT_data = HT_data_import[elements]


"""take out HT data compositions from total compositions"""
index_to_drop = []
for index, row in HT_data[elements].iterrows():
    index_to_drop.append(int(np.where((np.array(comp_space_total) == np.array(row)).all(axis = 1))[0]))
comp_space_reduced = comp_space_total.drop(index = index_to_drop).reset_index(drop = True)


#%%
"""Random"""
new_candidates = comp_space_reduced.sample(n = random_candidates, random_state = random_state)
new_candidates = new_candidates.reset_index(drop = True)

#%%
"""move row with most Co to first"""
Co_max_index = new_candidates["Co"].idxmax()

# reorder index by bringing Co_max_index up
new_idx = [Co_max_index] + [i for i in range(len(new_candidates)) if i != Co_max_index]
new_candidates = new_candidates.iloc[new_idx].reset_index(drop=True)

#%%
"""Save initial candidates"""
now = datetime.now()
new_candidates.to_csv(dir_path + "\\selection_random" + "\\random candidates_" + now.strftime("%Y%m%d_%H-%M-%S") + ".txt", index = False, sep = "\t")

#%%
print(f"Script ended at {datetime.now()}")
end = datetime.now()
print(f"Script excecution took {end - start}")
