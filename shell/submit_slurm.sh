#!/bin/bash
nodes=("cn11-02" "cn11-07")
gpus=(1)

n_nodes=${#nodes[@]}
n_gpus=${#gpus[@]}
n_resources=$((n_nodes*n_gpus))

job_template="shell/run_train.job"
job_dir="shell/slurm"
mkdir -p "$job_dir"

# Find YAML files excluding common.yaml
config_files=($(find "$1" -name '*.yaml' ! -name 'common.yaml'))
n_files=${#config_files[@]}

declare -a job_idx=( $(for i in {0..${n_resources}}; do echo 0; done) )

create_job_file() {
    job_id=$1
    config_file=$2
    gpu_idx=$3
    output_file="$job_dir/job_${job_id}.job"
    cp "$job_template" "$output_file"
    echo -e "\necho ${config_file}\n" >> $output_file
    echo -e "export CUDA_VISIBLE_DEVICES=${gpu_idx}\n" >> $output_file
    echo -e "echo python scripts/run.py --config_file ${config_file} --fit True --test True" >> $output_file
    echo ${output_file}
}

resource_index=0
for ((i=0; i<n_files; i++)); do
    config_file="${config_files[$i]}"
    resource_index=$(( (i) % ${n_resources} ))
    node=${nodes[$((resource_index / ${n_gpus}))]}
    gpu_index=${gpus[$((resource_index % ${n_gpus}))]}

    job_file=$(create_job_file "$i" "$config_file" "$gpu_index")
    submit_cmd=(sbatch --nodelist="${nodes[$node_index]}" --parsable)

    if [[ ${job_idx[$resource_index]} -ne 0 ]]; then
        submit_cmd+=(--dependency=afterok:"${job_idx[$resource_index]}")
    fi

    submit_cmd+=("$job_dir/job_${i}.job")
    echo "${submit_cmd[@]}"
    job_idx[$resource_index]=$("${submit_cmd[@]}")
done
