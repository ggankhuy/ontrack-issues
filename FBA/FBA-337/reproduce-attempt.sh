set -x 
# cloned and modified from ../FBA-251/mini-reg/mini-test.sh

<<<<<<< HEAD
OPTION_ENABLE_RVS=1
OPTION_ENABLE_RCCL_TESTS=1
ITERATIONS=3
=======
OPTION_ENABLE_RVS=0
OPTION_ENABLE_RCCL_TESTS=0
OPTION_ENABLE_LLAMA2_TEST=1

>>>>>>> guyen
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
TORCHRUN=`which torchrun`

RVS=/opt/rocm/bin/rvs
ALL_REDUCE=~/extdir/gg/git/rccl-tests/build/all_reduce
RVS_CONFIG_MI300X=/opt/rocm/share/rocm-validation-suite/conf/MI300X/gst_stress.conf
RVS_CONFIG=$RVS_CONFIG_MI300X
LOG_FOLDER_RVS=$LOG_FOLDER/rvs/

LOG_FOLDER_RCCL=$LOG_FOLDER/rccl-tests/

<<<<<<< HEAD
    for i in {1..2}; do

        if [[ $OPTION_ENABLE_RCCL_TESTS == 1 ]] ; then
            if [[ ! -f $ALL_REDUCE ]] ; then echo "Can not find all reduce at $ALL_REDUCE." ; exit 1 ; fi
            mkdir -p $LOG_FOLDER_RCCL_TESTS/$i
            echo "Running rccl-test...all redice..."
            dmesg --clear
            $ALL_REDUCE -b 64 -e 1G -f 2
            dmesg  | tee $LOG_FOLDER_RVS/$i/$i.dmesg.log
            rocm-smi 2>&1 | tee $LOG_FOLDER_RCCL_TESTS/$i/$i.rocm-smi.0.log
            sleep 10
            rocm-smi 2>&1 | tee $LOG_FOLDER_RCCL_TESTS/$i/$i.rocm-smi.10.log
            sleep 10
            rocm-smi 2>&1 | tee $LOG_FOLDER_RCCL_TESTS/$i/$i.rocm-smi.20.log
        fi            
        if [[ $OPTION_ENABLE_RVS == 1 ]] ; then
            if [[ ! -f $RVS ]] ; then echo "Can not find rvs at $RVS." ; exit 1 ; fi
            mkdir -p $LOG_FOLDER_RVS/$i
            echo "Running RVS..."
            dmesg --clear
            $RVS -c $RVS_CONFIG 2>&1 | tee $LOG_FOLDER_RVS/$i/$i.rvs.run.log
            dmesg  | tee $LOG_FOLDER_RVS/$i/$i.dmesg.log
            rocm-smi 2>&1 | tee $LOG_FOLDER_RVS/$i/$i.rocm-smi.0.log
            sleep 10
            rocm-smi 2>&1 | tee $LOG_FOLDER_RVS/$i/$i.rocm-smi.10.log
            sleep 10
            rocm-smi 2>&1 | tee $LOG_FOLDER_RVS/$i/$i.rocm-smi.20.log
=======
LOG_FOLDER_LLAMA2=$LOG_FOLDER/llama2

for i in RVS RCCL LLAMA2; do
    mkdir -p $LOG_FOLDER_$i
done

for i in {1..50}; do
    if [[ $OPTION_ENABLE_LLAMA2_TEST == 1 ]] ; then
        mkdir -p $LOG_FOLDER_LLAMA2/$i
        if [[ -z $TORCHRUN ]] ; then
            echo "Unable to find torch run!. "
            exit 1
>>>>>>> guyen
        fi
        
        $TORCHRUN --standalone --nnodes=1 --nproc-per-node=8 \
        /workspace/vllm-private/benchmarks/benchmark_latency.py \
        --model /data/Llama-2-70b-chat-hf/  --batch-size 8 --input-len 4096 \
        --output-len 32 --tensor-parallel-size 8 --num-iters 1 --report \
        --report-file=$LOG_FOLDER_LLAMA2/$i/report.csv --use-cuda-graph 2>&1 | tee $LOG_FOLDER_LLAMA2/$i/torchrun.llama2.log

        sleep 30
        for i in {1..4} ; do
            rocm-smi 2>&1 | tee -a $LOG_FOLDER_LLAMA2/$i/rocm-smi.log
            sleep 10
        done
    fi
    if [[ $OPTION_ENABLE_RCCL_TESTS == 1 ]] ; then
        if [[ ! -f $ALL_REDUCE ]] ; then echo "Can not find all reduce at $ALL_REDUCE." ; exit 1 ; fi
        mkdir -p $LOG_FOLDER_RCCL_TESTS/$i
        echo "Running rccl-test...all redice..."
        dmesg --clear
        $ALL_REDUCE -b 64 -e 1G -f 2
        dmesg  | tee $LOG_FOLDER_RVS/$i/$i.dmesg.log
        rocm-smi 2>&1 | tee $LOG_FOLDER_RCCL_TESTS/$i/$i.rocm-smi.0.log
        sleep 10
        rocm-smi 2>&1 | tee $LOG_FOLDER_RCCL_TESTS/$i/$i.rocm-smi.10.log
        sleep 10
        rocm-smi 2>&1 | tee $LOG_FOLDER_RCCL_TESTS/$i/$i.rocm-smi.20.log
    fi    
    if [[ $OPTION_ENABLE_RVS == 1 ]] ; then
        if [[ ! -f $RVS ]] ; then echo "Can not find rvs at $RVS." ; exit 1 ; fi
        mkdir -p $LOG_FOLDER_RVS/$i
        echo "Running RVS..."
        dmesg --clear
        $RVS -c $RVS_CONFIG 2>&1 | tee $LOG_FOLDER_RVS/$i/$i.rvs.run.log
        dmesg  | tee $LOG_FOLDER_RVS/$i/$i.dmesg.log
        rocm-smi 2>&1 | tee $LOG_FOLDER_RVS/$i/$i.rocm-smi.0.log
        sleep 10
        rocm-smi 2>&1 | tee $LOG_FOLDER_RVS/$i/$i.rocm-smi.10.log
        sleep 10
        rocm-smi 2>&1 | tee $LOG_FOLDER_RVS/$i/$i.rocm-smi.20.log
    fi
done

