set -x 
OUT_DIR=./out/
mkdir -p $OUT_DIR
rm *.out
for params in "-fsanitize=address" "-fsanitize=address -shared-libsan" "-fsanitize=address -shared-libsan -g" ; do
    hipcc $params p61.cpp -o $OUT_DIR/p61.fsanitize.out
done
ls -l $OUT_DIR/*.out
