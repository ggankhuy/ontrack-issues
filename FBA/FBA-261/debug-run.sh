# this is a modification of linear.py so that, matrix sizes are passed from this script along ROCm debug parameters to associate these two.
set -x
LOG_DIR=log
sudo mkdir -p $LOG_DIR
csv_filename=output.csv

echo -ne "" > $csv_filename

for size in 512 1024 2048 4096 8192 ; do
    curr_line=""
    row_idx=0
    for dtype in float float16 bfloat16 ; do
        #TENSILE_DB=0x8000 python3 linear-arg.py 2>&1 | tee $LOG_DIR/linear-arg.size.$size.$size.$size.$dtype.TENSILE_DB.0x8000.log
        python3 linear-arg.py 2>&1 | tee $LOG_DIR/linear-arg.size.$size.$size.$size.$dtype.log
    done
    for dtype in float float16 bfloat16 ; do
        tflops_curr=`cat $LOG_DIR/linear-arg.size.$size.$size.$size.$dtype.log  | grep TFLOPS |  awk '{print $2}'`

        if [[ $row_idx == 2 ]]  ; then
            echo "$tflops_curr" >> $csv_filename
        else
            echo -ne "$tflops_curr," >> $csv_filename
        fi
        row_idx=$((row_idx+1))
    done
done

cat $csv_filename
