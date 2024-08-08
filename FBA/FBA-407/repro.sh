sudo mkdir log
PY_ANALYZER=`find . -name cdump_analyzer.py | head -1`
for i in ./CPER_files/*.cper ; do 
    sudo python3 $PY_ANALYZER -i $i 2>&1 | \
    sudo tee ./log/`basename $i`.log ; 
done
tree -fs log
