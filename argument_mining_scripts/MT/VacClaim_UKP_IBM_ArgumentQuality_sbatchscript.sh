#!/bin/bash
#SBATCH --job-name=UKPVaccIBMAQ
#SBATCH --ntasks=1 --cpus-per-task=6 --mem=6000M
#SBATCH -p gpu --gres=gpu:titanx:3
#SBATCH --time=10-00:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

echo $CUDA_VISIBLE_DEVICES
declare -a StringArray=("abortion" "cloning" "death_penalty" "gun_control" "marijuana_legalization" "school_uniforms" "minimum_wage" "nuclear_energy")
indel=0
seed=2018
path="$1"
for i in ${StringArray[@]}; do
   echo $i
   output_directory=$path/spurious_correlations_in_argmin/argument_mining_output/VacClaim_UKP_IBM_AQ/"$i"/$seed
   echo "training..."
   python3 $path/mt-dnn/train.py --multi_gpu_on --tensorboard --epochs 10 --init_checkpoint bert-base-uncased --data_dir $path/spurious_correlations_in_argmin/argument_mining_data/UKPArgMin/"$i"/bert_base_uncased_lower --train_datasets UKP,VaccinationCorpusClaims,IBMdebaterArgSearch-PremiseOnly,ArgumentQuality --test_datasets UKP,UKP_heldout,VaccinationCorpusClaims,IBMdebaterArgSearch-PremiseOnly,ArgumentQuality --task_def $path/spurious_correlations_in_argmin/argument_mining_scripts/taskdefs/all_tasks_taskdef.yml --seed $seed --batch_size 5 --batch_size_eval 5 --output_dir $output_directory
   python3 $path/spurious_correlations_in_argmin/argument_mining_scripts/mtdnn_testEval.py "$output_directory/UKP_heldout_test_scores_9.json" "$path/spurious_correlations_in_argmin/argument_mining_data/UKPArgMin/$i/UKP_heldout_test.tsv" 3 "$output_directory" $i
   (( indel++ ))
   echo "done with $i"
done
