#!/bin/bash



# This file executes the young age cancer project if you have the appropriate data


process_number=$1
age_info=$2

echo "you have chosen to do until process $process_number"

for p in $( ls src | grep -E '[0-9]' | head -$process_number )
do 
	echo "the process is $p"
	for sp in $( ls "src/$p" | grep -E '[0-9]' )
	do
		echo "doing process.."
		echo "python3 src/$sp"

		python3 src/$p/$sp --age $age_info
		
		if [ $? != 0 ] 
		then 

			echo "$sp did not end well"
			exit 99	
			
		else 
			echo "$sp ended successfully"

		fi

	done
	
done
