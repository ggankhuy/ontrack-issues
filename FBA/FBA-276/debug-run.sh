mkdir log
for (( i = 2; i <  33; i++ )) ; do
    echo "---- running for $i ... ----"
    python3  03-matrix-multiplication-mod.py --size=$i 2>&1 | tee log/03-matrix-multiplication-mod.log
    TENSILE_DB=0x8000 python3  03-matrix-multiplication-mod.py --size=$i 2>&1 | tee log/03-matrix-multiplication-mod.TENSILE.log

done
