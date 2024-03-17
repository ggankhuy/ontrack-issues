LOG_DIR=./log/
OUT_DIR=./out/
mkdir -p $LOG_DIR $OUT_DIR

for i in $OUT_DIR/*.out; do
    readelf -d $i 2>&1 | tee  $LOG_DIR/needed.$i.log
    ldd $i 2>&1 | tee $LOG_DIR/ldd.$i.log
done

tree -fs $LOG_DIR
