#!/bin/bash
nodes=("cn11-05" "cn11-06" "cn11-02" "cn11-13")
n_nodes=${#nodes[@]}
echo ${n_nodes}
job_template="shell/run_train.job"      
job_dir="shell/slurm"                   
mkdir -p "$job_dir"

# Find YAML files excluding common.yaml
config_files=($(find "$1" -name '*.yaml' ! -name 'common.yaml'))
n_files=${#config_files[@]}

declare -a job_idx=( $(for i in {0..${n_nodes}}; do echo 0; done) )

create_job_file() {
    job_id=$1
    config_file=$2
    output_file="$job_dir/job_${job_id}.job"
    cp "$job_template" "$output_file"
    echo -e "\necho ${config_file}\npython scripts/run.py --config_file ${config_file} --fit True --test True" >> $output_file
    echo ${output_file}
}

node_index=0
for ((i=0; i<n_files; i++)); do
    config_file="${config_files[$i]}"
    node_index=$(( (i) % ${n_nodes} ))

    job_file=$(create_job_file "$i" "$config_file")
    echo ${job_file}

    echo "Submitting job $i with config $config_file to $node"
    submit_cmd=(sbatch --nodelist="${nodes[$node_index]}" --parsable)

    if [[ ${job_idx[$node_index]} -ne 0 ]]; then
        submit_cmd+=(--dependency=afterok:"${job_idx[$node_index]}")
    fi

    submit_cmd+=("$job_dir/job_${i}.job")
    echo "${submit_cmd[@]}"

    # Submit the job
    job_idx[$node_index]=$("${submit_cmd[@]}")
    
done