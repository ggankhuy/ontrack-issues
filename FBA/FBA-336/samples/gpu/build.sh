set -x 
OUT_DIR=./out/
LOG_DIR=./log/
mkdir -p $OUT_DIR $LOG_DIR
rm *.out
for param in "-fsanitize=address" "-fsanitize=address -shared-libsan" "-fsanitize=address -shared-libsan -g"  ; do
#   param_no_whitespace=`sed -i '/ /c \\' $param`
    echo $param_no_whitespace
    hipcc $param p61.cpp -o $OUT_DIR/p61."$param".out 2>&1 | tee $LOG_DIR/hipcc."$param".log
done
ls -l $LOG_DIR/hipcc.*.log
ls -l $OUT_DIR/*.out
