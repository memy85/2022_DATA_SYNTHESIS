#!/bin/bash

source ./bin/functions

set_python_env

# python environment
activate_pyenv

# make log files
if [ -d "log" ]; then
    if [ -d "log/0_preprocess" ]; then
        echo "log and log/0_preprocess exists"
        touch log/0_preprocess/output.txt
        date >> log/0_preprocess/output.txt
    else 
        echo "log exists but not 0_preprocess"
        mkdir log/0_preprocess 
        touch log/0_preprocess/output.txt
        date >> log/0_preprocess/output.txt
    fi
else 
    echo "log does not exist."
    echo "creating log ..."
    mkdir log log/0_preprocess
    touch log/0_preprocess/output.txt
    date >> log/0_preprocess/output.txt
fi


# preprocess data

preprocess_data dg_rcnf
preprocess_data ex_diag
preprocess_data oprt_nfrm
preprocess_data pth_bpsy
preprocess_data pth_mlcr
preprocess_data pth_mnty
preprocess_data pth_srgc
preprocess_data pt_bsnf
preprocess_data trtm_casb
preprocess_data trtm_rd
preprocess_data dead_nfrm

# finished preprocess

echo "finished preprocessing"

# start concating all the output data
echo "======================= concating all the output =======================" >> log/0_preprocess/output.txt
echo "======================= concating all the output ======================="
python3 src/0_preprocess/concat_all_data.py >> log/0_preprocess/output.txt 2>&1

echo "concated all data"
