#!/bin/bash
#SBATCH -J clic-it
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=./.slurm/%A/%a_output.log
#SBATCH --error=./.slurm/%A/%a_error.log
#SBATCH --mem=64g
#SBATCH --array=0-N
slurm_dir="./.slurm/$SLURM_ARRAY_JOB_ID"
mkdir -p $slurm_dir
echo "Creating directory: $slurm_dir"
nvidia-smi
module load rust gcc arrow
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
    # meta-llama/Llama-3.1-8B-Instruct
    meta-llama/Llama-3.3-70B-Instruct
    # mistralai/Ministral-8B-Instruct-2410
)
# Generate all combinations
array_names=(
            seed
            use_prompt_tags
            n_icl_samples
            model_name
            )
combinations=$(cartesian_product array_names)


# Convert combinations to commands
declare -a commands=()
while IFS= read -r combo; do
    IFS=',' read -ra params <<< "$combo"
    
    cmd="python ./src/hf_train.py
                --seed ${params[0]}
                --use_prompt_tags ${params[1]}
                --n_icl_samples ${params[2]}
                --model_name ${params[3]}
                --suffix ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
                "
    commands+=("$cmd")
done <<< "$combinations"

total_combinations=${#commands[@]}

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    command_to_run="${commands[$SLURM_ARRAY_TASK_ID]}"
    echo "$command_to_run"
    $command_to_run
    {
        for array_name in "${array_names[@]}"; do
            # Access array by name using indirect expansion
            values="${array_name}[@]"
            echo "$array_name: ${!values}"
        done
    } > "${slurm_dir}/hyperparameters.txt"
else
    echo "This script should be run as a SLURM array job."
    echo "Use: sbatch --array=0-$((total_combinations-1)) $0"
    echo "This will distribute $total_combinations jobs across N GPUs."
fi