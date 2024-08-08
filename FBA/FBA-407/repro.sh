sudo mkdir log
for i in watchdog_fail/CPER_files/*.cper ; do 
    sudo python3 ./ADDC_CDUMP_Analyzer_INSTINCT_2.1.3/ADDC_CDUMP_Analyzer_INSTINCT/cdump_analyzer.py -i $i 2>&1 | \
    sudo tee ./log/`basename $i`.log ; 
done
tree -fs log
