#!/bin/bash

source do_process.sh

# for age in 50 45 40 35 30 ;
# do
#     python3 src/4_results/5_training_strategy.py --age $age --random_seed $1
#
#     if [ "$?" != 0 ]
#     then 
#       echo "the process did note end well in age $age"
#         echo ""
#         exit 1
#     else 
#         echo "ended for age : $age"
#     fi
# done
#
# echo "finished for all ages"

for age in 50 45 40 35 30 
do 
    do_process 2 $age 0 
    echo "finished process for $age"
done

cd young_age_privacy
./do_process.sh 2 0
echo "finisehd the privacy test!"
