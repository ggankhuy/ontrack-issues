# run kfd test individually and gather logs separately.
KFDTEST_LIST=./kfdtest.list.mi250
KFDTEST=/usr/local/bin/kfdtest

if [[ -z $KFDTEST ]] ; then
    echo "Unable to find kfdtest in $KFDTEST. Please install it first."
    exit 1
fi

DATE=`date +%Y%m%d-%H-%M-%S`
LOG_FOLDER=log/$DATE/
mkdir $LOG_FOLDER -p

while IFS= read -r line; do
    echo $line
    dmesg --clear
    mkdir $LOG_FOLDER/$line -p
    $KFDTEST --gtest_filter=$line 2>&1 | tee $LOG_FOLDER/$line/$line.log
    dmesg | tee $LOG_FOLDER/$line/$line.dmesg.log
done < $KFDTEST_LIST


