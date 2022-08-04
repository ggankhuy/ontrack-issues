PROCESSESS=512
MPIRUN=`sudo find /usr/ -name mpirun`
# MinSi's, simple call to hipGetDeviceCont
APP_PATH=./a.out

# hip-examples, hello world, multitide of HIP API calls.
APP_PATH=/home/AMD/gg/ROCm-5.2/HIP-Examples/HIP-Examples-Applications/HelloWorld/a.out

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

$MPIRUN --allow-run-as-root -np $PROCESSESS -N $PROCESSESS --oversubscribe $APP_PATH

t2=$SECONDS 
echo "total time for $PROCESSESS processes: "
echo $((t2-t1))



