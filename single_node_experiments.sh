#!/bin/bash
#SBATCH -J clic-it
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=./.slurm/%A/%a_output.log
#SBATCH --error=./.slurm/%A/%a_error.log
#SBATCH --mem=64g

slurm_dir="./.slurm/$SLURM_ARRAY_JOB_ID"
mkdir -p $slurm_dir
echo "Creating directory: $slurm_dir"
nvidia-smi
. .env/bin/activate

# Cartesian product function
cartesian_product() {
    local result=("")
    local -n arrays=$1
    
    for array_name in "${arrays[@]}"; do
        local -n current_array=$array_name
        local new_result=()
        
        for existing in "${result[@]}"; do
            for item in "${current_array[@]}"; do
                new_result+=("${existing:+$existing,}$item")
            done
        done
        result=("${new_result[@]}")
    done
    
    printf '%s\n' "${result[@]}"
}
declare -a seed=(
    0
    1
    2
    3
    4
)
declare -a do_train=(
    # 0
    1
)
declare -a use_prompt_tags=(
    0
    1
)
declare -a n_icl_samples=(
    0
    5
    10
    15
    20
)
declare -a model_name=(
    meta-llama/Llama-3.1-8B-Instruct
    # meta-llama/Llama-3.3-70B-Instruct
    # mistralai/Ministral-8B-Instruct-2410
)
declare -a coarse=(
    0   
    # 1
)
# Generate all combinations
array_names=(
            seed
            use_prompt_tags
            n_icl_samples
            model_name
            do_train
            coarse
            )
combinations=$(cartesian_product array_names)

load_in_4bit=0
batch_size_train=4
batch_size_eval=4
epochs=3
verbose_eval=1
# Convert combinations to commands
declare -a commands=()
while IFS= read -r combo; do
    IFS=',' read -ra params <<< "$combo"

    if [[ ${params[3]} == *"70B"* ]]; then
        load_in_4bit=1
    fi
    
    cmd="python ./src/train.py --seed ${params[0]} --use_prompt_tags ${params[1]} --n_icl_samples ${params[2]} --model_name ${params[3]} --do_train ${params[4]} --coarse ${params[5]} --suffix ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID} --load_in_4bit $load_in_4bit --batch_size_train $batch_size_train --batch_size_eval $batch_size_eval --epochs $epochs --verbose_eval $verbose_eval"
    commands+=("$cmd")
done <<< "$combinations"

# for comb in ${combinations[@]}; do
#     echo ${comb}
# done

total_combinations=${#commands[@]}

for command_to_run in "${commands[@]}"; do
        echo "$command_to_run"
        $command_to_run
done