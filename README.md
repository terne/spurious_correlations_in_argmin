# spurious_correlations_in_argmin

1) Request UKP data from Stab et al. (2018) and save the files in the argument_mining_data/ArgMin folder. Then run the the bash script save_ukp_files.sh. The other three datasets are already prepared for you in the tabulated formats ready for MT-DNN (folder names). 

2) Clone the MT-DNN repo: ... (version or my copy?)

In the MT-DNN repo, there is are script for preprocessing the data for the model, called prepro_std.py. This script except tab separated data and a task definition in a YAML file. If you followed the steps above, the data should be ready for preprocessing. All the task definitions used in this study are found in the folder argument_mining_scripts/taskdefs.



Use unkdata_gen.sh to prepare the UKP data with open-class words replaced by unknown tokens. Replace the path to the mt-dnn/prepro_std.py script with your path (i.e. where you placed the MT-DNN repo).

