#!/bin/bash
#SBATCH --job-name=mtdnn_test
#SBATCH --ntasks=1 --cpus-per-task=6 --mem=6000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=0-20:00:00

echo $CUDA_VISIBLE_DEVICES

seed=2018
path="$1"
output_directory=$path/spurious_correlations_in_argmin/argument_mining_output/UKP_topic_MTL/$seed
echo "training..."
python3 $path/mt-dnn/train.py --init_checkpoint bert-base-uncased --data_dir $path/spurious_correlations_in_argmin/argument_mining_data/UKPArgMin/topic_seperate_tasks --train_datasets abortion,cloning,deathpenalty,guncontrol,marijuanalegalization,schooluniforms,minimumwage,nuclearenergy --test_datasets abortion,cloning,deathpenalty,guncontrol,marijuanalegalization,schooluniforms,minimumwage,nuclearenergy --task_def $path/spurious_correlations_in_argmin/argument_mining_scripts/UKPthreeLabelMulti_task_def.yml --epochs 10 --seed $seed --batch_size 5 --batch_size_eval 5 --output_dir $output_directory
