#!/bin/bash
#SBATCH --job-name=VaccST
#SBATCH --ntasks=1 --cpus-per-task=6 --mem=6000M
#SBATCH -p gpu --gres=gpu:titanx:2
#SBATCH --time=4-00:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

echo $CUDA_VISIBLE_DEVICES
path="$1"
seed=2018
output_directory=$path/spurious_correlations_in_argmin/argument_mining_output/VacClaim_ST/$seed
echo "training..."
python3 $path/mt-dnn/train.py --multi_gpu_on --tensorboard --epochs 10 --init_checkpoint bert-base-uncased --data_dir $path/spurious_correlations_in_argmin/argument_mining_data/UKPArgMin/abortion/bert_base_uncased_lower --train_datasets VaccinationCorpusClaims --test_datasets VaccinationCorpusClaims --task_def $path/spurious_correlations_in_argmin/argument_mining_scripts/taskdefs/all_tasks_taskdef.yml --seed $seed --batch_size 5 --batch_size_eval 5 --output_dir $output_directory
