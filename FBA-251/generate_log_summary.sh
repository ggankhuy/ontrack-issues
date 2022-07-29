DATE=`date +%Y%m%d-%H-%M-%S`
LOG_SUMMARY=./log_summary-$DATE
LOG_SUMMARY_FILE=$LOG_SUMMARY/summary.log
mkdir -p $LOG_SUMMARY
egrep -irn "hip" > $LOG_SUMMARY/hip_error.log
ERR0="hipDeviceGetCount"
ERR1="ethernet_read_keys"
ERR2="Unable to read from socket"
ERR3="Completion with error at client"
echo -ne "" > $LOG_SUMMARY_FILE

for i in "$ERR0" "$ERR1" "$ERR2" "$ERR3" ; do
	egrep -irn "$i" ./*.txt > $LOG_SUMMARY/"$i".log
	echo "--------------" | tee -a  $LOG_SUMMARY_FILE
	echo "$i occurence: "| tee -a  $LOG_SUMMARY_FILE
	echo "cmd: egrep -irn '$i' ./*.txt"
	grep -irn "${i}" ./*.txt  | wc -l  2>&1 | tee -a $LOG_SUMMARY_FILE
done
