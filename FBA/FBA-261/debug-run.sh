# this is a modification of linear.py so that, matrix sizes are passed from this script along ROCm debug parameters to associate these two.
LOG_DIR=log
sudo mkdir -p $LOG_DIR

for size in 512 1024 2048 4096 8192 ; do
    for dtype in float float16 bfloat16 ; do
        TENSILE_DB=0x8000 python3 linear-arg.py 2>&1 | tee $LOG_DIR/linear-arg.size.$i.$i.$i.$dtype.TENSILE_DB.0x8000.log
        python3 linear-arg.py 2>&1 | tee $LOG_DIR/linear-arg.size.$i.$i.$i.$dtype.log
    done
done


