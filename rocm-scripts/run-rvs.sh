
DATE_FIXED=`date +%Y%m%d-%H-%M-%S`
ROCM_ROOT=/opt/rocm-5.1.3
RVS=$ROCM_ROOT/rvs/rvs
RVS_CONFIG_ROOT=$ROCM_ROOT/rvs/conf/
LOG_FOLDER_BASE=log/rvs/
mkdir -p $LOG_FOLDER_BASE
LOG_SUMMARY=$LOG_FOLDER_BASE/summary.log
CONF_PATH=$ROCM_ROOT/share/rocm-validation-suite/conf/
CONF_PATH=$ROCM_ROOT/rvs/conf/
AMDXIO_PATH=/root/gg/tools-cp/amdxio/MI250/AMDXIO
echo "start" | sudo tee $LOG_SUMMARY
if [[ ! -f $RVS ]] ; then
    echo "Unable to find rvs in $RVS."
    ls -l $RVS
    exit 1
fi

echo "START..." >> $LOG_SUMMARY

#for i in gst_single.conf pqt_single.conf pebb_single.conf babel.conf gpup_single.conf  ; do
for i in gst_single.conf pebb_single.conf babel.conf gpup_single.conf  ; do
#for i in pqt_single.conf ; do
    for j in {1..10} ; do
        FILE=$i-$j
        DATE=`date +%Y%m%d-%H-%M-%S`

        LOG_FOLDER=$LOG_FOLDER_BASE/$DATE_FIXED/$FILE/
        mkdir $LOG_FOLDER -p

        echo $DATE | sudo tee -a $LOG_SUMMARY

        $AMDXIO_PATH -xgmilinkstatus | tee -a $LOG_FOLDER/amdxio.xgmilinkstatus.pre.log
        $AMDXIO_PATH -xgmipcserrorchecking | tee -a $LOG_FOLDER/amdxio.xgmipcserrorchecking.pre.log

        echo $RVS -c $i | sudo tee -a $LOG_SUMMARY
        $RVS  -c $CONF_PATH/$i 2>&1 | sudo tee -a $LOG_FOLDER/rvs.gpu.action-$i.log
        dmesg | sudo tee $LOG_FOLDER/rvs.gpu.$i.dmesg.log

        $AMDXIO_PATH -xgmilinkstatus | tee -a $LOG_FOLDER/amdxio.xgmilinkstatus.post.log
        $AMDXIO_PATH -xgmipcserrorchecking | tee -a $LOG_FOLDER/amdxio.xgmipcserrorchecking.post.log

        sleep 30
        dmesg --clear
    done
done
