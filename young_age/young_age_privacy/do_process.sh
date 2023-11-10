#!/bin/bash


# This file executes the young age cancer project if you have the appropriate data
PATH=$(pwd)

function do_process {

    process_number=$1
    age_info=50
    random_seed=$2

    echo "the random_seed is $random_seed"
    echo "you have chosen to do until process $process_number"

    for p in $( ls src | grep -E '[0-9]' | head -$process_number )
    do 
        echo "the process is $p"
        for sp in $( ls "src/$p" | grep -E '[0-9]' )
        do
            echo "doing process.."
            echo "python3 src/$sp"

            python3 src/$p/$sp --age $age_info --random_seed $random_seed
            
            if [ $? != 0 ] 
            then 

                echo "$sp did not end well"
                exit 99	
                
            else 
                echo "$sp ended successfully"

            fi

        done
        
    done

}

# 100 times repeated experiment
# if the seed = 1 then only do the process once


# The process will produce a synthetic data for age 50
SET=$(seq 1 $2)

if [ $2 -eq 1 ]
then
    do_process $1 50 1
    echo "ended for only seed = 0"

else 

    for seed in $SET

    do
        do_process $1 50 $seed
        echo "ended process for seed $seed"

    done
fi

# now do the testing
python3 src/3_privacy_test/privacy_test.py --age 50 --random_seed 0 --feature_split 0.5
echo "tested privacy test!"

# copies the the results to figures folder of young_age
cp $PATH/figures/privacy_test1.csv $PATH/../figures/privacy_test1.csv









