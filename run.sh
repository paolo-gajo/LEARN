#!/bin/bash
#SBATCH -J clic-it
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=./.slurm/%A/%a_output.log
#SBATCH --error=./.slurm/%A/%a_error.log
#SBATCH --mem=64g
mkdir -p .slurm
nvidia-smi
module load rust gcc arrow
. .env/bin/activate

# python ./src/train.py
# python ./src/train.py --model_name meta-llama/Llama-3.3-70B-Instruct --load_in_4bit 1
python ./src/train.py --train_steps 5 --eval_steps 5 --epochs 1