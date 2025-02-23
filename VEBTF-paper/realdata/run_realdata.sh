#!/bin/bash

K=10
n=4096
start_idx=0
data_names=("ETTh1" "weather")
# data_names=("illness")
for data_name in "${data_names[@]}"; do
    file_name="${data_name}_4096"
    echo "Running CV for data_name=${data_name}, file_name=${file_name}"

    python -u realdata/main.py --K $K --n $n --data_name $data_name --file_name $file_name --start_idx $start_idx
done

