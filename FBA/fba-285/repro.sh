# this is cloned from FBA-285 script with necessary mod as needed.
set +x
# not working
PW='\#3paleHorse5\#'
USER=master

# working ok.
PW=amd1234
USER=ggankhuy

DATE=`date +%Y%m%d-%H-%M-%S`
OUTPUT_DIR=./log/fba-285-$DATE
CONFIG_IP_GUEST=gt-pla-u23-18.pla.dcgpu
CONFIG_PORT_GUEST=22
PASSWORDLESS_LOGIN=1
LOOP_COUNT=3
SINGLE_BAR="-----------------------------"
DOUBLE_BAR="============================="
CONFIG_WAIT_HOST_UP_LOOP_TIMES=60
CONFIG_WAIT_HOST_UP_LOOP_INTERVAL=5
CONFIG_PUBKEY=~/.ssh/id_rsa.pub

CONFIG_FLAG_DEBUG_SSH_LOGIN=" -v"
CONFIG_FLAG_DEBUG_SSH_LOGIN=""

CONFIG_FLAG_SSH_LOGIN_EXPLICIT=" -i $CONFIG_PUBKEY"
CONFIG_FLAG_SSH_LOGIN_EXPLICIT=""
CONFIG_TEST_MODE=0
# wait until host reboot. 60 loop + 1 second interval

function wait_host_up() {
    sleep 3
    for i in {1..300} ; do
        echo -ne "."
        ping -c 1 -W $CONFIG_WAIT_HOST_UP_LOOP_INTERVAL $CONFIG_IP_GUEST
        if  [[ $? -eq 0 ]] ; then
            echo "Host $CONFIG_IP_GUEST is back online."
            return 0
        fi
        if [[ ${i} -gt ${CONFIG_WAIT_HOST_UP_LOOP_TIMES} ]] ; then
            echo "host $CONFIG_IP_GUEST is not back online within $((CONFIG_WAIT_HOST_UP_LOOP_TIMES*CONFIG_WAIT_HOST_UP_LOOP_INTERVAL)) seconds, giving up..."
            return 1
        fi
    done
    echo "Host is not unreachable after reboot or not reachable at all."
    exit 1
}

#                "sudo reboot" \
#                "sudo dmesg | sudo tee fba-285/dmesg.after.reboot.$loop.log" \
#                "/usr/local/bin/run_kfdtest.sh 2>&1 | sudo tee fba-285/$loop.kfdtest.log " \
#                "/usr/local/bin/kfdtest --gtest_filter=-*LargestSysBuffero* 2>&1 | sudo tee fba-285/$loop.kfdtest.log " \
#                "sudo /opt/rocm-5.0.1/rvs/rvs -c /opt/rocm-5.0.1/rvs/conf/gst_single.conf 2>&1 | sudo tee fba-285/$loop.rvs.gst.test.log" \

echo "Verifying host is up..."
wait_host_up

if [[ $? -ne 0 ]] ; then
    echo "Can not ping host. Exiting..."
else
    echo "ok."
fi

CONFIG_CMD_KFD_TEST="/usr/local/bin/kfdtest --gtest_filter=-*LargestSysBuffero* 2>&1 | sudo tee fba-285/$loop.kfdtest.log "
CONFIG_CMD_KFD_TEST=""

for loop in {0..300
} ; do
    echo $DOUBLE_BAR
    echo "Current loop: $loop........"

    for cmd in  "sudo mkdir fba-285" \
                "sudo dmesg --clear"  "sudo modprobe amdgpu" \
                "sudo dmesg | sudo tee fba-285/dmesg.after.modprobe.amdgpu.$loop.log" \
                "sudo dmesg --clear" \
                "$CONFIG_CMD_KFD_TEST" \
                "sudo dmesg | sudo tee fba-285/dmesg.after.workload.'$loop'.log" \
                "sudo dmesg --clear" "sudo rmmod amdgpu" "sudo dmesg | sudo tee fba-285/dmesg.after.rmmod.$loop.log" \
        ; do
        echo "$SINGLE_BAR"
        echo --- $cmd ---

        if [[ -z $cmd ]] ; then
            echo "Warning: cmd is empty, bypassing..." 
        else
            if [[ $CONFIG_TEST_MODE -eq 1 ]] ; then
                echo "TEST_MODE: Executing cmd: $cmd"
            else
                if [[ $PASSWORDLESS_LOGIN -eq  1 ]] ; then
                    clear ; echo $SINGLE_BAR ;echo $SINGLE_BAR
                    echo "Using passwordless login: cmd: $cmd"
                    echo $SINGLE_BAR ; echo $SINGLE_BAR
                    sleep 3

                    if [[ ! -f $CONFIG_PUBKEY ]] ; then
                        echo "$CONFIG_PUBKEY does not exit. Likely fail...!"
                    fi

                    ssh -v $CONFIG_FLAG_DEBUG_SSH_LOGIN $CONFIG_FLAG_SSH_LOGIN_EXPLICIT -p $CONFIG_PORT_GUEST -o StrictHostKeyChecking=no $USER@$CONFIG_IP_GUEST $cmd
                else
                    echo "Logging in with password: cmd: $cmd"
                    sshpass -p $PW ssh -p $CONFIG_PORT_GUEST -o StrictHostKeyChecking=no $USER@$CONFIG_IP_GUEST "echo pwd: ;pwd; $cmd"
                fi
            fi
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

            if [[ $? -ne 0 ]] ; then
                echo "Can not ping host. Exiting..."
            fi
        else
            echo "not a reboot cmd"
        fi
    done
done

mkdir $OUTPUT_DIR -p

if [[ $CONFIG_TEST_MODE -eq 1 ]] ; then
    echo "TEST_MODE: Executing ssh copy..."
else
    if [[ $PASSWORDLESS_LOGIN -eq  1 ]] ; then
        
        scp -o StrictHostKeyChecking=no -r $USER@$CONFIG_IP_GUEST:~/fba-285 $OUTPUT_DIR
    else
        sudo sshpass -p $PW scp -p $CONFIG_PORT_GUEST -o StrictHostKeyChecking=no -r $USER@$CONFIG_IP_GUEST:~/fba-285 $OUTPUT_DIR
    fi
fi
