LOGDIR=log/debugL2
mkdir -p $LOGDIR

TENSILE_DB_VAL="20000"
#for (( i = 2; i <  ;33; i++ )) ; do
for (( i = 2; i <  33; i++ )) ; do
    echo "---- running for $i ... ----"
    python3  03-matrix-multiplication-mod.py --size=$i 2>&1 | tee $LOGDIR/03-matrix-multiplication-mod.$((i * 128)).log
    TENSILE_DB=0x$TENSILE_DB_VAL python3  03-matrix-multiplication-mod.py --size=$i 2>&1 | tee $LOGDIR/03-matrix-multiplication-mod.TENSILE.$TENSILE_DB_VAL.$((i*128)).log

done
