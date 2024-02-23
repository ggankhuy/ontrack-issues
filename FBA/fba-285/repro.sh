# this is cloned from FBA-212 script with necessary mod as needed.

# not working
PW='\#3paleHorse5\#'
USER=master

# working ok.
PW=amd1234
USER=root

DATE=`date +%Y%m%d-%H-%M-%S`
OUTPUT_DIR=fba-212-$DATE
CONFIG_IP_GUEST=gt-pla-u23-18.pla.dcgpu
CONFIG_PORT_GUEST=22
PASSWORDLESS_LOGIN=1
LOOP_COUNT=3
SINGLE_BAR="-----------------------------"
DOUBLE_BAR="============================="
CONFIG_WAIT_HOST_UP_LOOP_TIMES=60
CONFIG_WAIT_HOST_UP_LOOP_INTERVAL=5

# wait until host reboot. 60 loop + 1 second interval

function wait_host_up() {
    sleep 3
    for i in {1..300} ; do
        echo -ne "."
        ping -c 1 -W 3 $CONFIG_IP_GUEST
        if  [[ $? -eq 0 ]] ; then
            echo "Host $CONFIG_IP_GUEST is back online."
            return 0

        if [[ i >= ${CONFIG_WAIT_HOST_UP_LOOP_TIMES} ]] ; then
            echo "host $CONFIG_IP_GUEST is not back online within $((CONFIG_WAIT_HOST_UP_LOOP_TIMES*CONFIG_WAIT_HOST_UP_LOOP_INTERVAL)) \
                seconds, giving up..."
            return 1
        fi
    done
    echo "Host is not unreachable after reboot or not reachable at all."
    exit 1
}

#                "sudo reboot" \
#                "sudo dmesg | sudo tee fba-212/dmesg.after.reboot.'$loop'.log" \
#                "/usr/local/bin/run_kfdtest.sh 2>&1 | sudo tee fba-212/$loop.kfdtest.log " \
#                "/usr/local/bin/kfdtest --gtest_filter=-*LargestSysBuffero* 2>&1 | sudo tee fba-212/$loop.kfdtest.log " \
#                "sudo /opt/rocm-5.0.1/rvs/rvs -c /opt/rocm-5.0.1/rvs/conf/gst_single.conf 2>&1 | sudo tee fba-212/$loop.rvs.gst.test.log" \

wait_host_up

if [[ $? -ne = 0 ]] ; then
    echo "Can not ping host. Exiting..."
fi

for loop in {1..10} ; do
    echo $DOUBLE_BAR
    echo "Current loop: $loop........"

    for cmd in  "sudo mkdir fba-285" \
                "sudo dmesg --clear"  "sudo modprobe amdgpu" \
                "sudo dmesg | sudo tee fba-212/dmesg.after.modprobe.amdgpu.'$loop'.log" \
                "sudo dmesg --clear" \
                "/usr/local/bin/kfdtest --gtest_filter=-*LargestSysBuffero* 2>&1 | sudo tee fba-212/$loop.kfdtest.log " \
                "sudo dmesg | sudo tee fba-212/dmesg.after.workload.'$loop'.log" \
                "sudo dmesg --clear" "sudo rmmod amdgpu" "sudo dmesg | sudo tee fba-212/dmesg.after.rmmod.'$loop'.log" \
        ; do
        echo "$SINGLE_BAR"
        echo --- $cmd ---

        if [[ $PASSWORDLESS_LOGIN -eq  1 ]] ; then
            ssh -p $CONFIG_PORT_GUEST -o StrictHostKeyChecking=no $USER@$CONFIG_IP_GUEST $cmd
        else
            sshpass -p $PW ssh -p $CONFIG_PORT_GUEST -o StrictHostKeyChecking=no $USER@$CONFIG_IP_GUEST $cmd
        fi
        ret=$?
        sleep 3
        
        #if [[ $ret != 0 && $ret != 1 ]] ; then 
        #    echo "code: $ret. Error executing $cmd on remote $CONFIG_IP_GUEST with credentials: $USER/$PW"
        #    exit 1
        #fi
        sleep 1

        if [[ "$cmd" == "sudo reboot" ]] ; then
            echo "Reboot issued. Waiting for system to become online"
            wait_host_up

            if [[ $? -ne = 0 ]] ; then
                echo "Can not ping host. Exiting..."
            fi
        else
            echo "not a reboot cmd"
        fi
    done
done

sudo sshpass -p $PW scp  -p $CONFIG_PORT_GUEST -o StrictHostKeyChecking=no -r $USER@$CONFIG_IP_GUEST:~/fba-212 $OUTPUT_DIR
