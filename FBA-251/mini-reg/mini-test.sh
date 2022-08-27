IPMIHOST=10.6.188.58
IPMIUSER=ADMIN
IPMIPW=ADMIN

if [[ $1 -ne "" ]] ; then
    LOG_SUFFIX=$1
fi
DATE=`date +%Y%m%d-%H-%M-%S`
if [[ $LOG_SUFFIX ]] ;then
    LOG_FOLDER=log/$DATE-$LOG_SUFFIX
else
    LOG_FOLDER=log/$DATE
fi

RVS=`find /opt -name rvs`
RVS=/opt/rocm-5.2.0/rvs/rvs
RVS_CONFIG=`find /opt -name gst_single.conf`
LOG_FOLDER_RVS=$LOG_FOLDER/rvs/
mkdir -p $LOG_FOLDER_RVS

TB=`find /opt -name TransferBench`
TB_DIRNAME=`dirname $TB`
TB_BASENAME=`basename $TB`
TB_CONFIG=`pwd`/TB.8.bidi.GPU.cfg
echo "TB_DIRNAME/BASENAME/CONFIG: $TB_DIRNAME, $TB_BASENAME, $TB_CONFIG"
LOG_FOLDER_TB=$LOG_FOLDER/tb/
mkdir -p $LOG_FOLDER_TB

if [[ -z $TB ]] || [[ -z $TB_CONFIG ]]; then
    echo "Unable to find either rvs or rvs config. TB: $TB, TB_CONFIG: $TB_CONFIG"
else
    echo "Running TB..."
    dmesg --clear
    ipmitool -H $IPMIHOST -I lanplus -U $IPMIUSER -P $IPMIPW sel clear
    dmesg -wH > $LOG_FOLDER_TB/tb.dmesg.log &
    echo $TB $TB_CONFIG 512M 2>&1 | tee $LOG_FOLDER_TB/tb.log
    $TB $TB_CONFIG 512M 2>&1 | tee $LOG_FOLDER_TB/tb.log
    ipmitool -H $IPMIHOST -I lanplus -U $IPMIUSER -P $IPMIPW sel elist > $LOG_FOLDER_TB/tb.sel.log
    #kill -p $DMESG_PID
fi

if [[ -z $RVS ]] || [[ -z $RVS_CONFIG ]]; then
    echo "Unable to find either rvs or rvs config. RVS: $RVS, RVS_CONFIG: $RVS_CONFIG"
else
    echo "Running RVS..."
    dmesg --clear
    ipmitool -H $IPMIHOST -I lanplus -U $IPMIUSER -P $IPMIPW sel clear
    dmesg -wH > $LOG_FOLDER_RVS/rvs.dmesg.log &
    DMESG_PID=$!
    mkdir $LOG_FOLDER_RVS/sel -p
    $RVS -c $RVS_CONFIG | tee $LOG_FOLDER_RVS/rvs.log
    ipmitool -H $IPMIHOST -I lanplus -U $IPMIUSER -P $IPMIPW sel elist > $LOG_FOLDER_RVS/rvs.sel.log
    #kill -p $DMESG_PID
fi

LOG_FOLDER_KFDTEST=$LOG_FOLDER/kfdtest/
mkdir -p $LOG_FOLDER_KFDTEST

RUN_KFDTEST_SH=`which run_kfdtest.sh`

if [[ -z $RUN_KFDTEST_SH ]] ; then
    echo "Unable to find run_kfdtest.sh"
else
    echo "Running kfdtest..."
    dmesg --clear
    ipmitool -H $IPMIHOST -I lanplus -U $IPMIUSER -P $IPMIPW sel clear
    dmesg -wH > $LOG_FOLDER_KFDTEST/run_kfdtest.dmesg.log &
    DMESG_PID=$!
    mkdir $LOG_FOLDER_KFDTEST/sel -p
    $RUN_KFDTEST_SH | tee $LOG_FOLDER_KFDTEST/run_kfdtest.log
    ipmitool -H $IPMIHOST -I lanplus -U $IPMIUSER -P $IPMIPW sel elist > $LOG_FOLDER_KFDTEST/run_kfdtest.sel.log
    #kill -p $DMESG_PID
fi

