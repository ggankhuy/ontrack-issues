set -x 
OUT_DIR=./out/
LOG_DIR=./log/
mkdir -p $OUT_DIR $LOG_DIR
rm *.out
for param in "-fsanitize=address" "-fsanitize=address -shared-libsan" "-fsanitize=address -shared-libsan -g"  ; do
    param_no_nonprtable=`echo $param | sed 's/ //g'`
    param_no_nonprtable=`echo $param_no_nonprtable | sed 's/\-/./g'`
    param_no_nonprtable=`echo $param_no_nonprtable | sed 's/=/./g'`
    echo param_no_nonprtable: $param_no_nonprtable
    hipcc --offload-arch=gfx90a:xnack+ $param p61.cpp -o $OUT_DIR/p61"$param_no_nonprtable".out 2>&1 | tee $LOG_DIR/hipcc"$param_no_nonprtable".log
done
ls -l $LOG_DIR/hipcc.*.log
ls -l $OUT_DIR/*.out
