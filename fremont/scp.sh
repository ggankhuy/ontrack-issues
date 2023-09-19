test_mode=0
rm -rf summary.log

hostnames=('10.216.114.27' '10.216.114.25' '10.216.114.24' '10.216.69.251' '10.216.114.22' '10.216.114.21' '10.216.114.28' '10.216.114.29' \
'10.216.114.30' '10.216.114.31' '10.216.69.252' '10.216.114.33' '10.216.114.34' '10.216.114.36' '10.216.114.37' '10.216.114.38' \
'10.216.114.39' '10.216.114.40' '10.216.114.41' '10.216.114.42' '10.216.114.56' '10.216.114.44' '10.216.114.45' '10.216.114.46' \
'10.216.114.47' '10.216.114.48' '10.216.114.54' '10.216.114.53' '10.216.114.50' '10.216.114.49')
for i in $(seq 0 ${#hostnames[@]}) ; do
    echo "----"
    echo "processing $i: ${hostnames[$i]}"
    ret=`sshpass -p amd123 ssh  -o StrictHostKeyChecking=no amd@${hostnames[$i]} "sudo -S dmidecode | grep 'Serial Number: S' | rev | cut -d ' ' -f1 | rev"`
    echo "ret: $ret"
    dst=$ret-${hostnames[$i]}
    if [[ -d $dst ]] ; then
        echo "Bypassing $dst, directory already exist."
    else
        mkdir $dst
    fi

    if [[ $test_mode -eq 0 ]] ; then
        ret=`sshpass -p amd123 scp -r -o StrictHostKeyChecking=no amd@${hostnames[$i]}://home/amd/Documents/mfg_script $dst`
    else
        echo "test_mode: bypassing actual scp command"
    fi

    du -hs $dst | tee -a summary.log

    if [[ $test_mode -eq 1 ]] && [[ $i > 3 ]] ; then
        echo "test_mode: i: $i.exiting afer 3 iteration..."
        break
    fi
done

du -hs S*
exit 0

