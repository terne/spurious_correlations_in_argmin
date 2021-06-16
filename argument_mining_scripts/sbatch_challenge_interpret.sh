#!/bin/bash
#SBATCH --job-name=interpretUNK
#SBATCH --ntasks=1 --cpus-per-task=6 --mem=32GB
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=1-00:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

path="$1"

# single-task UKP:
python3 $path/mt-dnn/interpretability.py --cuda True --task_id 0 --checkpoint '$path/spurious_correlations_in_argmin/argument_mining_output/cloning/2018/model_9.pt' --unprep_dev_data "$path/spurious_correlations_in_argmin/argument_mining_data/challenge/challenge_sentence.csv"
# with UNK model:
#python3 $path/mt-dnn/interpretability.py --cuda True --task_id 0 --checkpoint '$path/spurious_correlations_in_argmin/argument_mining_output/UNK_models/ArgumentQuality_UKP/cloning/2018/model_9.pt' --unprep_dev_data "/home/ktj250/spurious_correlations_in_argmin/argument_mining_data/challenge/UNKchallenge_sentence.tsv"

# Multi-task with another task id did not work, figure out why:
#python3 $path/mt-dnn/interpretability.py --cuda True --task_id 3 --checkpoint '$path/spurious_correlations_in_argmin/argument_mining_output/VacClaim_UKP_IBM_AQ/cloning/2018/model_9.pt' --unprep_dev_data "$path/spurious_correlations_in_argmin/argument_mining_data/challenge/challenge_sentence.csv"
