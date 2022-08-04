PROCESSESS=8
MPIRUN=/usr/lib64/openmpi/bin/mpirun
APP_PATH=./a.out
$MPIRUN --allow-run-as-root -np $PROCESSESS -N $PROCESSESS --oversubscribe $APP_PATH



