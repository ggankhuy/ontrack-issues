KFD=/usr/local/bin/run_kfdtest.sh
DATE=`date +%Y%m%d-%H-%M-%S`
LOG_FOLDER=log/kfd/log-kfd-$DATE
mkdir $LOG_FOLDER -p
LOG_SUMMARY=$LOG_FOLDER/summary.log
echo -ne "" > $LOG_SUMMARY
t1=$SECONDS
for i in {2..9} ; do
    t1a=$SECONDS
    dmesg --clear
    echo "Running kfd for gpu $i..." | tee -a $LOG_SUMMARY
    #sleep 5
    #$KFD -n $i 2>&1 | sudo tee $LOG_FOLDER/kfd.gpu.$i.log
    dmesg | sudo tee $LOG_FOLDER/kfd.gpu.$i.dmesg.log
    t2a=$SECONDS
    perGpuTime=$((t2a-t1a))
    echo "gpu $i run time: $perGpuTime" | tee -a $LOG_SUMMARY
done
t2=$SECONDS
totalTime=$((t2-t1))
echo "Total run time for all gpus: $totalTime" | tee -a $LOG_SUMMARY
