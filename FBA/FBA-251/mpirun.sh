PROCESSESS=512
MPIRUN=`sudo find /usr/ -name mpirun`
# MinSi's, simple call to hipGetDeviceCont
APP_PATH=./a.out
CONFIG_USE_MPIRUN=0
DBG_LOG=dbg.log
#FBA-251
APP_NAME=hip_get_device_count_repro
APP_PATH=./$APP_NAME.out

# hip-examples, hello world, multitide of HIP API calls.
#APP_NAME=HelloWorld
#APP_PATH=/home/nonroot/gg/ROCm-5.2/HIP-Examples/HIP-Examples-Applications/$APP_NAME/$APP_NAME

# hip-examples, vectorAdd, multitude of HIP API calls.
#APP_NAME=vectoradd_hip
#APP_PATH=/home/nonroot/gg/ROCm-5.2/HIP-Examples/vectorAdd/$APP_NAME.exe

dmesg --clear

if [[ ! -z $1 ]] ; then
    LOG_SUFFIX=$1
fi
DATE=`date +%Y%m%d-%H-%M-%S`
if [[ $LOG_SUFFIX ]] ;then
    LOG_DIR_ROOT=log/$DATE-$LOG_SUFFIX
else
    LOG_DIR_ROOT=log/$DATE
fi
echo LOG_DIR_ROOT: $LOG_DIR_ROOT
sleep 5
DATE=`date +%Y%m%d-%H-%M-%S`

for i in {1..1} ; do
    LOG_DIR=$LOG_DIR_ROOT/$i
    sudo mkdir -p $LOG_DIR
    echo "===="  | tee -a $LOG_DIR_ROOT/$DBG_LOG
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
            if [[ $n -eq 508 ]] ; then
                echo "i, log file: $n, $APP_NAME.$n.log"
                ltrace $APP_PATH 2>&1 | sudo tee $LOG_DIR/$APP_NAME.$n.log &
            else
                echo "i, log file: $n, $APP_NAME.$n.log"
                $APP_PATH 2>&1 | sudo tee $LOG_DIR/$APP_NAME.$n.log &
            fi  
            #ltrace $APP_PATH 2>&1 > $LOG_DIR/$APP_NAME.$n.log &
        done
    fi

    ret=1
    while [[ $ret -ne 0 ]] ; do
        echo "----"  | tee -a $LOG_DIR_ROOT/$DBG_LOG
        echo "dbg: i/j: $i/$j: ps -rax: " | tee -a $LOG_DIR_ROOT/$DBG_LOG
        ret=`ps -rax | grep vectoradd_hip.exe | wc -l`
        echo $ret | tee -a $LOG_DIR_ROOT/$DBG_LOG
        sleep 2
    done

    t2=$SECONDS 
    echo "total time for $PROCESSESS processes: " | tee -a $LOG_DIR_ROOT/$DBG_LOG
    echo $((t2-t1))  | tee -a $LOG_DIR_ROOT/$DBG_LOG

    dmesg >  $LOG_DIR/dmesg.log

    ret=`egrep -rn PASSED $LOG_DIR/* | grep 508`

    if [[ -z $ret ]] ; then
        echo "strace/ltrace is gathered from failing threads, done. Exiting" | tee -a $LOG_DIR_ROOT/$DBG_LOG
        echo "log dir: $LOG_DIR" | tee -a $LOG_DIR_ROOT/$DBG_LOG
        exit 0
    else
        echo "loop $i: strace/ltrace is not gathered from passing threads. Continuing..." | tee -a $LOG_DIR_ROOT/$DBG_LOG
    fi
done

echo "Unable to run strace/ltrace on failing threads."  | tee -a $LOG_DIR_ROOT/$DBG_LOG
exit 1
