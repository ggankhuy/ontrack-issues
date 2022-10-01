# run kfd test individually and gather logs separately.
KFDTEST_LIST=./kfdtest.list.mi250
KFDTEST=/usr/local/bin/kfdtest


TOPOLOGY_SYSFS_DIR=/sys/devices/virtual/kfd/kfd/topology/nodes
getHsaNodes() {
    for i in $(find $TOPOLOGY_SYSFS_DIR  -maxdepth 1 -mindepth 1 -type d); do
        simdcount=$(cat $i/properties | grep simd_count | awk '{print $2}')
        if [[ $simdcount != 0 ]]; then
            hsaNodeList+="$(basename $i) "
        fi
    done
    echo "$hsaNodeList"
}

if [[ -z $KFDTEST ]] ; then
    echo "Unable to find kfdtest in $KFDTEST. Please install it first."
    exit 1
fi

nodes=$(getHsaNodes)

DATE=`date +%Y%m%d-%H-%M-%S`
LOG_FOLDER=log/$DATE/
mkdir $LOG_FOLDER -p

for node in $nodes; do
    echo "Starting testing node $node"
    while IFS= read -r line; do
        echo $line
        dmesg --clear
        mkdir $LOG_FOLDER/$line -p
        $KFDTEST --gtest_filter=$line 2>&1 | tee $LOG_FOLDER/$line/$line.gpu-$node.log | tee -a $LOG_FOLDER/kfdtest.log
        dmesg | tee $LOG_FOLDER/$line/$line.dmesg.log
    done < $KFDTEST_LIST
    echo "Finished testing node $node"
done
