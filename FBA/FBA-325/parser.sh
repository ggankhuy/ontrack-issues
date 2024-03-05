FILENAME=$1
DEBUG=0
if [[ -z $FILENAME ]] ; then
    echo "Please specify json file to parse"
    exit 1
fi
    
echo "Processing json trace $FILENAME..."
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
    arr_durations=`egrep -irn "rms_norm" $FILENAME -A 1 | grep dur |  awk '{ print $5 }' | sed 's/,//'`

    arr_durations_sub_100=`egrep -irn $i  $FILENAME -A 1| grep -i "dur.*[0-9][0-9]" | grep -v "dur.*[0-9][0-9][0-9]" | wc -l`
    arr_durations_sub_10000=`egrep -irn $i  $FILENAME -A 1| grep -i "dur.*[0-9][0-9][0-9][0-9]" | grep -v "dur.*[0-9][0-9][0-9][0-9][0-9]" | wc -l`
    arr_durations_super_10000=`egrep -irn $i $FILENAME -A 1| grep -i "dur.*[0-9][0-9][0-9][0-9][0-9]" | wc -l`
    iter=0

    if [[ $DEBUG = 0 ]] ; then "echo arr_durations: $arr_durations" ; fi

    for j in $arr_durations; do

        if [[ $DEBUG = 0 ]] ; then echo "iter: $iter" ; fi
        if [[ $DEBUG = 0 ]] ; then echo "Processing $j..." ; fi
        if [[ $DEBUG = 0 ]] ; then echo "Sum: $arr_durations_sum" ; fi
        arr_durations_sum=$((arr_durations_sum+j))

        if [[ $j -gt $arr_durations_max ]] ; then 
            if [[ $DEBUG = 1 ]] ; then echo "Updating arr_duration_max from $arr_durations_max to $j..."; fi
            arr_durations_max=$j
        fi
        if [[ $j -lt $arr_durations_min ]] ; then 
            if [[ $DEBUG = 1 ]] ; then echo "- Updating arr_duration_min from $arr_durations_min to $j..."; fi
            arr_durations_min=$j
        fi
        iter=$((iter+1))
    done
    arr_durations_avg=$((arr_durations_sum/arr_durations_len))

    echo "No of occurrences of $i                  :   $arr_durations_len"
    echo "No of occurrences <100,$i(us)            :   $arr_durations_sub_100"
    echo "No of occurrences >100,<10000,$i(us)     :   $arr_durations_sub_10000"
    echo "No of occurrences >10000,$i(us)          :   $arr_durations_super_10000"
    echo "Average of $i             (us)           :   $arr_durations_avg"
    echo "maximum duration of $i    (us)           :   $arr_durations_max"
    echo "minimum duration of $i    (us)           :   $arr_durations_min"
    echo "total wall duration of $i (ms)           :   $((arr_durations_sum/1000))"

            
done
