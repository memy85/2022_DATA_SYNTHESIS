#!/bin/bash
#


for age in 50 45 40 35 30 ;
do
    python3 src/4_results/5_training_strategy.py --age $age

    if [ "$?" != 0 ]
    then 
        echo "the process did note end well in age $age"
        echo ""
        exit 1
    else 
        echo "ended for age : $age"
    fi
done

echo "finished for all ages"
