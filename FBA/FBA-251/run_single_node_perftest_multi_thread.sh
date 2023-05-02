#!/bin/bash

THREADS=$1
MLX_INDEX=$2
HOSTNAME=$3
LOG_FOLDER=./log/
sudo mkdir $LOG_FOLDER

if [ "$#" != "3" ]; then
    echo
    echo "Check your arguments"
    echo "USAGE: $0 [threads] [1 or 2] [hostname]"
    echo "Example: If you want to run 256 threads on a dual-port Mellanox NIC,"
    echo "e.g. $0 256 2 smc01.amd.com"
    echo
    exit 0
fi

USER_THREADS=$(( $THREADS /16 ))
MIN_THREADS=16
MAX_THREADS=2000
SKT_START=10000

# Check that threads passed to script for testing does not exceed the max range of 2000.
# 2000 is the max number we allocated for this script.
if [[ $THREADS -gt $MAX_THREADS ]];
then
	echo "ERROR: Argument $THREADS exceeds maximum supported threads for this test script. Range is $MIN_THREADS to $MAX_THREADS."
	exit 0
elif [[ $THREADS -lt $MIN_THREADS ]];
then
	echo "ERROR: Argument $THREADS must be at least $MIN_THREADS. Range is $MIN_THREADS to $MAX_THREADS."
	exit 0
fi


echo
echo "Generating $THREADS ib_write_bw server/client threads on Mellanox NICs"

######## CLIENT THREAD ##########
for ROCM in 0 1 2 3 4 5 6 7
do
	rocm_index=$(( $ROCM * $MAX_THREADS ))
	mlx=$(( $ROCM * $MLX_INDEX ))
	echo -n "mlx5_$mlx "
	CURRENT_THREAD=1
	while [[ $CURRENT_THREAD -le $USER_THREADS ]]
	do
		socket=$(( $SKT_START + $rocm_index + $CURRENT_THREAD ))
		#echo $socket mlx5_$mlx rocm$ROCM
		if [[ $ROCM < 4 ]]; then
			#taskset -c 0-63,128-191 /home/AMD/perftest/ib_write_bw -S 0 -p $socket --report_gbits -d mlx5_$mlx -m 4096 -n 10000 -x 3 --use_rocm=$ROCM 2>&1 > /home/AMD/test_01_02_with_rocm_0715_nic_set_100/server_ctr_ubbsmc02_mlx5_$mlx-client_port$socket.txt &
			taskset -c 0-63,128-191 /home/AMD/perftest/ib_write_bw -S 0 -p $socket --report_gbits -d mlx5_$mlx -m 4096 -n 10000 -x 3 --use_rocm=$ROCM 2>&1 | tee $LOG_FOLDER/server-$HOSTNAME.$socket.mlx5_$mlx.rocm-$ROCM.log &
		else
			#taskset -c 64-127,192-254 /home/AMD/perftest/ib_write_bw -S 0 -p $socket --report_gbits -d mlx5_$mlx -m 4096 -n 10000 -x 3 --use_rocm=$ROCM 2>&1 > /home/AMD/test_01_02_with_rocm_0715_nic_set_100/server_ctr_ubbsmc02_mlx5_$mlx-client_port$socket.txt &
			taskset -c 64-127,192-254 /home/AMD/perftest/ib_write_bw -S 0 -p $socket --report_gbits -d mlx5_$mlx -m 4096 -n 10000 -x 3 --use_rocm=$ROCM 2>&1 | tee $LOG_FOLDER/server-$HOSTNAME.$socket.mlx5_$mlx.rocm-$ROCM.log &
		fi
		let CURRENT_THREAD=CURRENT_THREAD+1
	done
done

sleep 10

######## SERVER THREAD ##########
CURRENT_THREAD=1
while [[ $CURRENT_THREAD -le $USER_THREADS ]]
do
	for ROCM in 0 1 2 3 4 5 6 7
	do
		rocm_index=$(( $ROCM * $MAX_THREADS ))
		mlx=$(( $ROCM * $MLX_INDEX ))
		socket=$(( $SKT_START + $rocm_index + $CURRENT_THREAD ))
		#echo $socket mlx5_$mlx rocm$ROCM
		if [[ $ROCM < 4 ]]; then
			#taskset -c 0-63,128-191 /home/AMD/perftest/ib_write_bw -S 0 -p $socket --report_gbits -d mlx5_$mlx -m 4096 -n 10000  -x 3 -F $HOSTNAME -D 100 --use_rocm=$ROCM &> /home/AMD/test_01_02_with_rocm_0715_nic_set_100/client_ctr_ubbsmc02_mlx5_$mlx-server_port$socket.txt &
			taskset -c 0-63,128-191 /home/AMD/perftest/ib_write_bw -S 0 -p $socket --report_gbits -d mlx5_$mlx -m 4096 -n 10000  -x 3 -F $HOSTNAME -D 100 --use_rocm=$ROCM 2&>1  | tee $LOG_FOLDER/client-$HOSTNAME.$socket.mlx5_$mlx.rocm-$ROCM.log
		else
			#taskset -c 64-127,192-254 /home/AMD/perftest/ib_write_bw -S 0 -p $socket --report_gbits -d mlx5_$mlx -m 4096 -n 10000 -x 3 -F $HOSTNAME -D 100 --use_rocm=$ROCM &> /home/AMD/test_01_02_with_rocm_0715_nic_set_100/client_ctr_ubbsmc02_mlx5_$mlx-server_port$socket.txt &
			taskset -c 64-127,192-254 /home/AMD/perftest/ib_write_bw -S 0 -p $socket --report_gbits -d mlx5_$mlx -m 4096 -n 10000 -x 3 -F $HOSTNAME -D 100 --use_rocm=$ROCM 2&>1 | tee $LOG_FOLDER/client-$HOSTNAME.$socket.mlx5_$mlx.rocm-$ROCM.log
		fi
	done
	let CURRENT_THREAD=CURRENT_THREAD+1
done
echo
echo "Starting RDMA perftest..."
sleep 1
echo -n "pgrep: "
pgrep ib_write_bw | wc -l
