# Spurious Correlations in Cross-Topic Argument Mining

Please cite our paper if using this repository: Thorn Jakobsen, T., Barrett, M., & Søgaard, A. (2021). Spurious Correlations in Cross-Topic Argument Mining. STARSEM. https://aclanthology.org/2021.starsem-1.25/ 

&nbsp; 

### Reproduce the experiments
1) Request UKP data from Stab et al. (2018) and save the files in the argument_mining_data/ArgMin folder. Then run the the bash script save_ukp_files.sh. The other three datasets are already prepared for you in the tabulated formats ready for MT-DNN (folder names). 

2) Clone the MT-DNN repository from https://github.com/terne/mt-dnn.git and place it beside (not within) this repository.

In the MT-DNN repo, there is are script for preprocessing the data for the model, called prepro_std.py. This script expect tab separated data and a task definition in a YAML file. If you followed the steps above, the data should be ready for preprocessing.  All the task definitions used in this study are found in the folder argument_mining_scripts/taskdefs. We already provided the preprocessed auxiliary data for you, so you only need the preprocess the UKP data: 

3) Run prepro.sh [path to repo]. For example, if you placed this repository and the MT-DNN repository in the /home/user then write: prepro.sh /home/user

4) Use unkdata_gen.sh [path to repo] to prepare the UKP data with open-class words replaced by unknown tokens. 

5) Now you are ready to train and test models. To train and test the single-task models, you can use the scripts in the ST folder, and to train and test the multi-task model, you can use the scripts in the MT folder. Again, pass the path to the repositories as an argument to the shell scripts.

### Interpretability
After the above steps, you can reproduce the experiments made with LIME by running the scripts sbatch_interpret.sh and sbatch_challenge_interpret.sh (TO DO: change paths for output and default unprep_dev_data in MT-DNN/interpretability.py)


### TO DO
Scripts for preparing data for the ukp-top model.




