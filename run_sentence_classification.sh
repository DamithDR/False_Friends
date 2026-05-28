#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --job-name=ff_sent_cls
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/ff_sent_cls_%j.out
#SBATCH --error=logs/ff_sent_cls_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d.dolamullage@lancaster.ac.uk

mkdir -p logs

# Pass --lang es / --lang fr via sbatch, or leave empty to run both sequentially:
#   sbatch run_sentence_classification.sh --lang es
#   sbatch run_sentence_classification.sh --lang fr
#   sbatch run_sentence_classification.sh
python sentence_classification.py "$@"
