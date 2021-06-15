#!/bin/bash
#SBATCH --job-name=mtdnn_test
#SBATCH --ntasks=1 --cpus-per-task=6 --mem=6000M
#SBATCH --time=0-01:00:00

echo $CUDA_VISIBLE_DEVICES
declare -a StringArray=("abortion" "cloning" "death_penalty" "gun_control" "marijuana_legalization" "school_uniforms" "minimum_wage" "nuclear_energy")
indel=0
for i in ${StringArray[@]}; do
   echo $i
   python3 save_data_in_tab_format.py $indel "$i"
   echo "done saving new tsv files"
done
