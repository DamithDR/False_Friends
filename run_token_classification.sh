#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --job-name=ff_tok_cls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ff_tok_cls_%j.out
#SBATCH --error=logs/ff_tok_cls_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d.dolamullage@lancaster.ac.uk

source venv/bin/activate

mkdir -p logs outputs/token_classification

# Defaults to FR. Override with the LANG_ARG env var, e.g.:
#   LANG_ARG=es sbatch run_token_classification.sh
#   sbatch run_token_classification.sh --epochs 15 --model_name microsoft/mdeberta-v3-base
LANG_ARG="${LANG_ARG:-fr}"

python token_classification.py train \
    --model_name xlm-roberta-base \
    --data_dir data/token_classification \
    --lang "$LANG_ARG" \
    --output_dir "outputs/token_classification/ff_xlmr_${LANG_ARG}" \
    --epochs 10 \
    --batch_size 16 \
    --lr 5e-5 \
    "$@"
