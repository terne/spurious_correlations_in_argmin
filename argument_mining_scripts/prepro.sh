#!/bin/bash
#SBATCH --job-name=mtdnn_test
#SBATCH --ntasks=1 --cpus-per-task=6 --mem=6000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=0-02:00:00

echo $CUDA_VISIBLE_DEVICES
path="$1"
declare -a StringArray=("abortion" "cloning" "death_penalty" "gun_control" "marijuana_legalization" "school_uniforms" "minimum_wage" "nuclear_energy")
indel=0
for i in ${StringArray[@]}; do
   echo $i
   echo "preproccessing and making json files..."
   python3 $path/mt-dnn/prepro_std.py --do_lower_case --root_dir $path/spurious_correlations_in_argmin/argument_mining_data/UKPArgMin/$i --task_def $path/spurious_correlations_in_argmin/argument_mining_scripts/taskdefs/UKPthreeLabelSingle_task_def.yml --model bert-base-uncased
   echo "done with $i"
done
