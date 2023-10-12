"""Imports"""
from datetime import datetime
import pandas as pd
import os
from sklearn.manifold import MDS     
import numpy as np

#%%
print(f"Script started at {datetime.now()}")
start = datetime.now()

#%%
"""set current directory to location of .py file"""
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

"""load data and skip first row"""
data_file = "MOBO_summary_20230617_16-21-38.txt"

HT_data = pd.read_table(data_file)
comp_space = pd.read_table("Composition_space_5element_10%.txt")

#%%
"""get element composition columns and make np.array"""
elements = ["Co", "Mn", "Sb", "Sn", "Ti"]
HT_data_elements = HT_data[elements]
X = comp_space.to_numpy()

#%%
"""get index of 100% compositions for later annotating of MDS plot"""
index_100 = []
for i in elements:
    index_100.append(comp_space.index[comp_space[i] == 100][0])
    
#%%
"""set MDS object for 2D"""
mds = MDS(n_components=2, random_state=0)
X_transform_2d = mds.fit_transform(X)
np.savetxt("2D_MDS_coordinates_quinary_10percent.txt", X_transform_2d, delimiter = "\t")

#%%
"""Pick the MDS coordinates that match the HT_compositions"""

"""get index of composition space for each HT composition"""
index_comp = []
for index, row in HT_data_elements.iterrows():
    index_comp.append(int(np.where((np.array(comp_space) == np.array(row)).all(axis = 1))[0]))

"""get MDS coordinates for these indecies and add them to HT data"""
HT_data["MDS_X"] = X_transform_2d[index_comp][:, 0]
HT_data["MDS_Y"] = X_transform_2d[index_comp][:, 1]

#%%
"""save"""
HT_data.to_csv(data_file.split(".")[0] + "_MDS.txt", index = False, sep = "\t")

#%%
print(f"Script ended at {datetime.now()}")
end = datetime.now()
print(f"Script excecution took {end - start}")
