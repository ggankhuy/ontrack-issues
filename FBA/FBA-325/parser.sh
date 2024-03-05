FILENAME=$1
DEBUG=1
if [[ -z $FILENAME ]] ; then
    echo "Please specify json file to parse"
    exit 1
fi
    

SINGLE_LINE="--------------------------------"
for i in _rms_norm_kernel_0d1d2d; 
do
    echo $SINGLE_LINE    
    echo "Processing $i..."
    echo $SINGLE_LINE    

    arr_durations_sum=0
    arr_durations_avg=0
    arr_durations_max=0
    arr_durations_min=10000000
    arr_durations_len=0
    arr_durations_len=`egrep -irn $i $FILENAME | wc -l`
    arr_durations=`egrep -irn "rms_norm" stream_set_badperf.json  -A 1 | grep dur |  awk '{ print $5 }' | sed 's/,//'`
    for j in $arr_durations; do

        if [[ $DEBUG = 1 ]] ; then echo "Processing $j..." ; fi
        arr_durations_sum=$((arr_durations_sum+j))

        if [[ $j -gt $arr_durations_max ]] ; then 
            if [[ $DEBUG=1 ]] ; then echo "Updating arr_duration_max from $arr_duration_max to $j..."; fi
            arr_durations_max=$j
        fi
        if [[ $j -lt $arr_durations_min ]] ; then 
            if [[ $DEBUG=1 ]] ; then echo "Updating arr_duration_min from $arr_duration_min to $j..."; fi
            arr_durations_min=$j
        fi
    done
    arr_durations_avg=$((arr_durations_sum/arr_durations_len))

    echo "No of occurrences of $i       :      $arr_durations_len"
    echo "Average of $i             (us):      $arr_durations_avg"
    echo "maximum duration of $i    (us):      $arr_durations_max"
    echo "minimum duration of $i    (us):      $arr_durations_min"
            
done
