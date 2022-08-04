PROCESSESS=384
MPIRUN=`sudo find /usr/ -name mpirun`
# MinSi's, simple call to hipGetDeviceCont

APP_PATH=./a.out
CONFIG_USE_MPIRUN=0
LOG_DIR=./log
sudo mkdir -p $LOG_DIR

#FBA-251
APP_NAME=hip_get_device_count_repro
APP_PATH=./$APP_NAME.out

# hip-examples, hello world, multitide of HIP API calls.
APP_NAME=HelloWorld
APP_PATH=/home/AMD/gg/ROCm-5.2/HIP-Examples/HIP-Examples-Applications/$APP_NAME/$APP_NAME

if [[ -z $MPIRUN ]] ; then
    echo "Unable to find mpirun. Will try installing openmpi-devel..."
    sudo yum install openmpi-devel -y
    MPIRUN=`sudo find /usr/ -name mpirun`
    if [[ -z $MPIRUN ]] ; then
        echo "Unable to find mpirun after installing openmpi-devel. Giving up."
        exit 1
    fi
fi

t1=$SECONDS

if [[ $CONFIG_USE_MPIRUN -eq 1 ]] ; then
    echo "Using mpirun..."
    $MPIRUN --allow-run-as-root -np $PROCESSESS -N $PROCESSESS --oversubscribe $APP_PATH
else 
    echo "Using for loop..."
    for (( n=0; n < $PROCESSESS; n++ )) ; do
        echo "i, log file: $n, $APP_NAME.$n.log"
        $APP_PATH 2>&1 | sudo tee $LOG_DIR/$APP_NAME.$n.log &
    done
fi

t2=$SECONDS 
echo "total time for $PROCESSESS processes: "
echo $((t2-t1))



