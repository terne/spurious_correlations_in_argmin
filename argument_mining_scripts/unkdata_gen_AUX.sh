#!/bin/bash
#SBATCH --job-name=UNKdata
#SBATCH --ntasks=1 --cpus-per-task=6 --mem=16GB
#SBATCH --time=1-00:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err


python3 unk_data_proc.py "/home/ktj250/prj1/argument_mining_data/Argument-Extraction-Corpus-annotated-phrases-by-topic" "ArgumentQuality_dev.tsv"
python3 unk_data_proc.py "/home/ktj250/prj1/argument_mining_data/Argument-Extraction-Corpus-annotated-phrases-by-topic" "ArgumentQuality_test.tsv"
python3 unk_data_proc.py "/home/ktj250/prj1/argument_mining_data/Argument-Extraction-Corpus-annotated-phrases-by-topic" "ArgumentQuality_train.tsv"


python3 unk_data_proc.py "/home/ktj250/prj1/argument_mining_data/IBM_debater_argument_search_tabulated_testset" "IBMdebaterArgSearch-PremiseOnly_dev.tsv"
python3 unk_data_proc.py "/home/ktj250/prj1/argument_mining_data/IBM_debater_argument_search_tabulated_testset" "IBMdebaterArgSearch-PremiseOnly_test.tsv"
python3 unk_data_proc.py "/home/ktj250/prj1/argument_mining_data/IBM_debater_argument_search_tabulated_testset" "IBMdebaterArgSearch-PremiseOnly_train.tsv"


python3 unk_data_proc.py "/home/ktj250/prj1/argument_mining_data/VaccinationCorpus_claim_tab" "VaccinationCorpusClaims_dev.tsv"
python3 unk_data_proc.py "/home/ktj250/prj1/argument_mining_data/VaccinationCorpus_claim_tab" "VaccinationCorpusClaims_test.tsv"
python3 unk_data_proc.py "/home/ktj250/prj1/argument_mining_data/VaccinationCorpus_claim_tab" "VaccinationCorpusClaims_train.tsv"
