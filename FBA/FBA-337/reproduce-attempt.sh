set -x 
# cloned and modified from ../FBA-251/mini-reg/mini-test.sh

OPTION_ENABLE_RVS=1

if [[ ! -z $1 ]] ; then
    LOG_SUFFIX=$1
fi
DATE=`date +%Y%m%d-%H-%M-%S`
if [[ $LOG_SUFFIX ]] ;then
    LOG_FOLDER=log/$DATE-$LOG_SUFFIX
else
    LOG_FOLDER=log/$DATE
fi

echo "LOG_FOLDER: $LOG_FOLDER"
RVS=/opt/rocm/bin/rvs
RVS_CONFIG_MI300X=/opt/rocm/share/rocm-validation-suite/conf/MI300X/gst_stress.conf
RVS_CONFIG=$RVS_CONFIG_MI300X
LOG_FOLDER_RVS=$LOG_FOLDER/rvs/
mkdir -p $LOG_FOLDER_RVS

if [[ -z $RVS ]] || [[ -z $RVS_CONFIG ]]; then
    echo "Unable to find either rvs or rvs config. RVS: $RVS, RVS_CONFIG: $RVS_CONFIG"
else

    for i in {1..50}; do

        if [[ $OPTION_ENABLE_RVS == 1 ]] ; then
            mkdir -p $LOG_FOLDER_RVS/$i
            echo "Running RVS..."
            dmesg --clear
            $RVS -c $RVS_CONFIG 2>&1 | tee $LOG_FOLDER_RVS/$i/$i.rvs.run.log
            dmesg  | tee $LOG_FOLDER_RVS/$i/$i.dmesg.log &
            rocm-smi 2>&1 | tee $LOG_FOLDER_RVS/$i/$i.rocm-smi.0.log
            sleep 10
            rocm-smi 2>&1 | tee $LOG_FOLDER_RVS/$i/$i.rocm-smi.10.log
            sleep 10
            rocm-smi 2>&1 | tee $LOG_FOLDER_RVS/$i/$i.rocm-smi.20.log
        fi
    done
fi
