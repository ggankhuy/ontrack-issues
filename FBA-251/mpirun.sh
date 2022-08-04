
PROCESSESS=8
MPIRUN=`sudo find /usr/ -name mpirun`
if [[ -z $MPIRUN ]] ; then
    echo "Unable to find mpirun. Will try installing openmpi-devel..."
    sudo yum install openmpi-devel -y
    MPIRUN=`sudo find /usr/ -name mpirun`
    if [[ -z $MPIRUN ]] ; then
        echo "Unable to find mpirun after installing openmpi-devel. Giving up."
        exit 1
    fi
fi


APP_PATH=./a.out
$MPIRUN --allow-run-as-root -np $PROCESSESS -N $PROCESSESS --oversubscribe $APP_PATH



