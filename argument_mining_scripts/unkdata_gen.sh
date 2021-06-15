#!/bin/bash
#SBATCH --job-name=UNKdata
#SBATCH --ntasks=1 --cpus-per-task=6 --mem=16GB
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=1-00:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

declare -a StringArray=("abortion" "cloning" "death_penalty" "gun_control" "marijuana_legalization" "school_uniforms" "minimum_wage" "nuclear_energy")
for i in ${StringArray[@]}; do
   echo $i
   #python3 unk_data_proc.py "../argument_mining_data/UKPArgMin/$i/" "UKP_dev.tsv" False
   #python3 unk_data_proc.py "../prj1/argument_mining_data/UKPArgMin/$i/" "UKP_test.tsv" False
   #python3 unk_data_proc.py "../argument_mining_data/UKPArgMin/$i/" "UKP_train.tsv" False
   #python3 unk_data_proc.py "../argument_mining_data/UKPArgMin/$i/" "UKP_heldout_test.tsv" False
   #python3 unk_data_proc.py "../argument_mining_data/UKPArgMin/$i/" "UKP_heldout_dev.tsv" False

   python3 unk_data_proc.py "../argument_mining_data/UKPArgMin/$i/" "UKP_dev.tsv" True
   python3 unk_data_proc.py "../argument_mining_data/UKPArgMin/$i/" "UKP_test.tsv" True
   python3 unk_data_proc.py "../argument_mining_data/UKPArgMin/$i/" "UKP_train.tsv" True
   python3 unk_data_proc.py "../argument_mining_data/UKPArgMin/$i/" "UKP_heldout_test.tsv" True
   python3 unk_data_proc.py "../argument_mining_data/UKPArgMin/$i/" "UKP_heldout_dev.tsv" True

   python3 /home/user/mt-dnn/prepro_std.py --do_lower_case --root_dir ../argument_mining_data/UKPArgMin/$i --task_def ../argument_mining_scripts/taskdefs/UNKcommonContentWordsUKP_taskdef.yml --model bert-base-uncased

done
