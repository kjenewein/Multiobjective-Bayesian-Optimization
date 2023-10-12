# HT_Data_MOBO
Folder containing the electrochemical data during MOBO sampling.

# HT_Data_random
Folder containing the electrochemical data during random sampling.

# Progress_MOBO
Folder containing calculated normalized hypervolume per iteration for MOBO screening. It also contains data for the Pareto front.

# Progress_Random
Folder containing calculated normalized hypervolume per iteration for random screening.

# Selection_MOBO
Folder containing candidates selected by MOBO algorithm per iteration.

# Selection_Random
Folder containing candidates selected by random sampler per iteration.

# MOBO.py
This script reads the data from the HT experiments and performs the MOBO algorithm. The output is the next 15 candidates to be screened.

# Random.py
This script is used to generate random selections from the given composition space. It is used to benchmark the MOBO against.

# calculate_progress.py
This script calculates the normalized hypervolume per iteration and outputs the Pareto front for the given observations.

# FixedNoiseMultiTaskGP_prediction_RemainingSpace.py
This script is used to fit the FixedNoiseMultiTaskGP model on the observations made during the MOBO screening to predict the activity (overpotential) and stability (overpotential change) for the remaining compositions within the search space. It also categorizes each composition into uniary, binary, ternary, quaternary, and quinary.