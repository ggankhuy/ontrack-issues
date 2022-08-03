PROCESSESS=260
MPIRUN=/usr/lib64/openmpi/bin/mpirun
$MPIRUN --allow-run-as-root -np $PROCESSESS -N $PROCESSESS --oversubscribe ./a.out
