#!/bin/bash


# This file executes the young age cancer project if you have the appropriate data

function do_process {

    process_number=$1
    age_info=$2
    random_seed=$3

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


# SET=$(seq 1 $3)
#
# if [ $3 -eq 1 ]
# then
#     do_process $1 $2 0
#     echo "ended for only seed = 0"
#
# else 
#
#     for seed in $SET
#
#     do
#         do_process $1 $2 $seed
#         echo "ended process for seed $seed"
#
#     done
# fi

