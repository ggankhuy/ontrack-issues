# Platform info script to gather system information.
#!/usr/bin/sh


# Usage:
# techsupport.sh - to display host information.
# techsupport.sh <VMno> - to display both host and guest information.

# Requirements/assumptions:
#   - To run this script, either amdgpu or libgv must be installed and loaded and this script will determine 
#   the presence using dkms status and gather respective logs.
#   - if in instance, both amdgpu and libgv(gim) installed, it will gather only libgv information and ignore amdgpu.
#   - To gather guest log, guest VM must be running and amdgpu on guest must be loaded.
#   - libgv and gim is used interchangeably, meaning kvm based virtualization drivers.
#   - for guest VM, it is assumed rocm is instsalled.
#   - for virtualization host/guest systems, it is assumed host is running gim/libgv and guest is running rocm.

# log file structure:
# ts/
#       <DATE>
#           <DATE>-platform-summary log (containing overall information on host + guest(if applicable)
# <DATE->-techsupport.tar
#           host part: cpu, gpu model, amdgpu/libgv version, kernel versO/Sn, host os version.
#           guest part: cpu, gpu mod(guest) version, kernel version, guest O/S version.
#       host/
#           host logs: dmesg, modi(or libgv) info amdgpu, amdgpu(or libgv) parameters, rocminfo, rocm-smi.
#       guest/
#           guest logs: dmesg, modinfo amdgpu, amdgpu parameters, rocminfo, rocm-smi.


p1=$1
DEBUG=1
SINGLE_BAR=`for i in {1..50}; do echo -n '-'; done`
DOUBLE_BAR=`for i in {1..50}; do echo -n '='; done`
DATE=`date +%Y%m%d-%H-%M-%S`
CONFIG_PATH_PLAT_INFO=`pwd`/ts/$DATE
CONFIG_FILE_TAR=./ts/$DATE/$DATE-techsupport.tar
CONFIG_FILE_PLAT_INFO=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-platform-info.log
CONFIG_SUBDIR_HOST=host
CONFIG_SUBDIR_GUEST=guest

#   If set, copies the sysfiles, if unset, loops over and gather sys file content into CONFIG_FILE_SYSFILES_HOST

CONFIG_SYS_FILE_COPY=1

#   If set, limits gathering non-gpu information as possible.

OPTION_LIMIT_NONGPU_LOG=1

CONFIG_FILE_PARM_AMDGPU_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-parm-amdgpu-host.log
CONFIG_FILE_PARM_LIBGV_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-parm-libgv-host.log
CONFIG_FILE_MODINFO_AMDGPU_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-modinfo-amdgpu-host.log
CONFIG_FILE_MODINFO_LIBGV_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-modinfo-libgv-host.log
CONFIG_FILE_DMESG_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-dmesg-host.log
CONFIG_FILE_SYSLOG_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-syslog-host.log
CONFIG_FILE_KERN_LOG_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-kernlog-host.log
CONFIG_FILE_DMIDECODE_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-dmidecode-host.log
CONFIG_FILE_ROCMINFO_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-rocminfo-host.log
CONFIG_FILE_ROCMSMI_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-rocm-smi-host.log

CONFIG_FILE_LSPCI_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-lspci-host.log
CONFIG_FILE_LSPCI_T_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-lspci-t-host.log
CONFIG_FILE_LSPCI_VVV_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-lspci-vvv-host.log
CONFIG_FILE_LSPCI_VVVXXX_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-lspci-vvvxxx-host.log
CONFIG_FILE_NUMA_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-numa-host.log

CONFIG_FILE_SYSFILES_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-sysfiles-host.log
CONFIG_FILE_SYSFILES_HOST_DIR=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-sysfiles-host/

CONFIG_FILE_PARM_AMDGPU_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-parm-amdgpu-guest.log
CONFIG_FILE_DMESG_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-dmesg-guest-$p1.log
CONFIG_FILE_CLINFO_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-dmesg-clinfo-$p1.log
CONFIG_FILE_MODINFO_AMDGPU_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-modinfo-amdgpu-$p1.log
CONFIG_FILE_SYSLOG_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-syslog-guest.log
CONFIG_FILE_KERN_LOG_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-kernlog-guest.log
CONFIG_FILE_DMIDECODE_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-dmidecode-guest.log
CONFIG_FILE_ROCMINFO_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-rocminfo-guest.log
CONFIG_FILE_ROCMSMI_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-rocm-smi-guest.log

CONFIG_FILE_SYSFILES_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-sysfiles-guest.log
CONFIG_FILE_SYSFILES_GUEST_DIR=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-sysfiles-guest/

CONFIG_FILE_LSPCI_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-lspci-guest.log
CONFIG_FILE_LSPCI_T_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-lspci-t-guest.log
CONFIG_FILE_LSPCI_VVV_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-lspci-vvv-guest.log
CONFIG_FILE_LSPCI_VVVXXX_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-lspci-vvvxxx-guest.log

mkdir -p $CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST
mkdir -p $CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST

if [[ $p1 == "--help" ]] ; then
	clear
	echo "Usage: "
	echo "$0 - get host information for libgv or amdgpu logs"
	echo "$0 <vm_Id> get both host and guest information. Use virsh list to list the VMs and get the corresponding ID."
	exit 0
fi

OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`
echo "OS_NAME: $OS_NAME"
case "$OS_NAME" in
   "Ubuntu")
      echo "Ubuntu is detected..."
      PKG_EXEC=apt
      ;;
   "CentOS Linux")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      ;;
   "CentOS Stream")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      ;;
   *)
     echo "Unsupported O/S, exiting..." ; exit 1
     ;;
esac

$PKG_EXEC install tree virt-what sshpass wcstools numactl -y 
if [[ $?  -ne 0 ]] ; then
	echo "ERROR: Failed to install packages..."
	exit 1
fi

#   Start gathering logs.
#   If p1 is not specified, gather only host system.
#       Determine if gim (libgv) installed (virtualized) or amdgpu installed (baremetal).
#       amdgpu compatible logs (baremetal):
#       libgv compatible logs (virt).
#   else if p1 is specified and libgv is not installed, exit with error or optionally, display host with amdgpu info only.
#       libgv compatible logs (virt) from host.
#       gather guest logs.
#

function ts_amdgpu_compat_full_logs() {
    ts_helper_full_logs amdgpu
}

function ts_libgv_compat_full_logs() {
    ts_helper_full_logs libgv
}

function ts_helper_full_logs() {
    local p1=$1
    if [[ $DEBUG ]] ; then
        echo "ts_helper_summary_logs entered..."
        echo "p1: $p1"
    fi

    # Gather dmesg.

    if [[ $OPTION_LIMIT_NONGPU_LOG == 1 ]] ; then
        dmesg | grep -i amdgpu | tee $CONFIG_FILE_DMESG_HOST
    else
        dmesg | tee $CONFIG_FILE_DMESG_HOST
    fi
    #cat /var/log/syslog | tee $CONFIG_FILE_SYSLOG_HOST
	#cat /var/log/kern.log | tee $CONFIG_FILE_KERN_LOG_HOST

    # Gather smbios information.

    if [[ $OPTION_LIMIT_NONGPU_LOG == 1 ]] ; then
        for i in 0 1 2 3 4 ; do
            dmidecode -t $i | tee -a $CONFIG_FILE_DMIDECODE_HOST
        done
    else
        dmidecode | tee $CONFIG_FILE_DMIDECODE_HOST
    fi

    # Gather pcie information.

    lspci | tee -a $CONFIG_FILE_LSPCI_HOST
    lspci -t | tee -a $CONFIG_FILE_LSPCI_T_HOST
    lspci -vvv | tee -a $CONFIG_FILE_LSPCI_VVV_HOST
    lspci -vvvxxx | tee -a $CONFIG_FILE_LSPCI_VVVXXX_HOST

    # Gather numa info

    numactl --hardware | tee $CONFIG_FILE_NUMA_HOST

    # Gather amdgpu/kfd driver information.

    if [[ $p1 == "amdgpu" ]] ; then
        modinfo amdgpu | tee $CONFIG_FILE_MODINFO_AMDGPU_HOST
        for i in /sys/module/amdgpu/parameters/* ; do echo -n $i: ; cat $i ; done 2>&1 | tee $CONFIG_FILE_PARM_AMDGPU_HOST 
        rocminfo 2>&1 | tee $CONFIG_FILE_ROCMINFO_HOST
        rocm-smi --showall 2>&1 | tee $CONFIG_FILE_ROCMSMI_HOST
        rocm-smi --showtopo 2>&1 | tee $CONFIG_FILE_ROCMSMI_HOST

        # Gather sysfiles.

        if [[ $CONFIG_SYS_FILE_COPY == 1 ]] ; then
            if [[ -d /sys/devices/virtual/kfd/kfd/topology/nodes ]] ; then
                pushd /sys/devices/virtual/kfd/kfd/topology/nodes
                mkdir -p $CONFIG_FILE_SYSFILES_HOST_DIR
                cp -vrP ./* $CONFIG_FILE_SYSFILES_HOST_DIR/
                rm -rf $CONFIG_FILE_SYSFILES_HOST_DIR/*/caches
                popd
            else
                echo "Warning: /sys/devices/virtual/kfd/kfd/topology/nodes does not exist, will not gather sys files. Is kfd driver loaded?"
                sleep 3
            fi
        else
            pushd /sys/devices/virtual/kfd/kfd/topology/nodes

            echo -ne "" > $CONFIG_FILE_SYSFILES_HOST
            for node in ./* ; do
                cd $node
                #echo "current dir: `pwd`"
                #echo "current node: $node"
                #ls -l 
                for filename in gpu_id name properties; do
                    echo "writing: `pwd`/$filename..."
                    #sleep 1

                    echo "-- `pwd`/$filename" | tee -a $CONFIG_FILE_SYSFILES_HOST
                    cat `pwd`/$filename | tee -a $CONFIG_FILE_SYSFILES_HOST
                done
                cd ..
                cd $node/io_links
                for io_link in ./* ; do
                    echo "-- `pwd`/properties" | tee -a $CONFIG_FILE_SYSFILES_HOST
                    cat `pwd`/properties | tee -a $CONFIG_FILE_SYSFILES_HOST
                done
                cd ../..        
    
                cd $node/p2p_links
                for p2p_link in ./* ; do
                    echo "-- `pwd`/properties" | tee -a $CONFIG_FILE_SYSFILES_HOST
                    cat `pwd`/properties | tee -a $CONFIG_FILE_SYSFILES_HOST
                done
                cd ../..
            done            
            popd    
        fi
    elif [[ $p1 == "libgv" ]] ; then

        # Gather libgv information and parameters.

        modinfo gim | tee $CONFIG_FILE_MODINFO_LIBGV_HOST
        for i in /sys/module/gim/parameters/* ; do echo -n $i: ; cat $i ; done 2>&1 | tee $CONFIG_FILE_PARM_LIBGV_HOST 
    elif [[ $p1 = "" ]] ; then
        echo "Warning: p1 is empty, neither amdgpu or libgv specific logs will be gathered."
    else
        echo "warning: Unknown parameter! "
    fi
}

function ts_helper_summary_logs() {
    local p1=$1
    if [[ $DEBUG ]] ; then
        echo "ts_helper_summary_logs entered..."
        echo "p1: $p1"
    fi
    echo $DOUBLE_BAR|  tee -a $COFNIG_FILE_PLAT_INFO
    echo "LIBGV HOST SUMMARY:"
    echo $DOUBLE_BAR|  tee -a $COFNIG_FILE_PLAT_INFO
	echo    "HOSTNAME:  " `hostname` | tee $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO

 	echo -n "IP:        " | tee -a $CONFIG_FILE_PLAT_INFO
    if [[  `which ifconfig` ]] ; then
	    echo `ifconfig | grep inet | grep -v inet6 | grep -v 127.0.0.1` | tee -a $CONFIG_FILE_PLAT_INFO
    else
        echo "ifconfig not available" | tee -a $CONFIG_FILE_PLAT_INFO
    fi

	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "O/S:          "`cat /etc/os-release  | grep PRETTY_NAME` | tee -a $CONFIG_FILE_PLAT_INFO

	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "O/S KERNEL: 	"`uname -r` | tee -a $CONFIG_FILE_PLAT_INFO
	echo "current linux boot grub config: 	"`cat /proc/cmdline` | tee -a $CONFIG_FILE_PLAT_INFO

    if [[ $p1 == "amdgpu" ]] ; then
    	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	    echo "GPU:      " | tee -a $CONFIG_FILE_PLAT_INFO
        if [[ `which rocm-smi` ]] ; then
            rocm-smi --showall | grep "series:" 2>&1 | tee -a $CONFIG_FILE_PLAT_INFO
            elif [[ `rocminfo` ]] ; then
            rocminfo | grep gfx 2>&1 | tee -a $CONFIG_FILE_PLAT_INFO
        elif [[ `lspci` ]] ; then
            lspci | grep Disp 2>&1 | tee -a $CONFIG_FILE_PLAT_INFO
        else
            echo "Unable to determine GPU, neither rocm-smi, rocminfo or lspci utilities available" | tee -a $CONFIG_FILE_PLAT_INFO
        fi

        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
        echo "GPUDRIVER:   "`lsmod | egrep "^amdkfd|^amdgpu"` | tee -a $CONFIG_FILE_PLAT_INFO
        modinfo amdgpu | egrep "^filename|^version" | tee -a $CONFIG_FILE_PLAT_INFO
        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
        echo "GPUDRIVER(DKMS):   "`dkms status | grep amdgpu` | tee -a $CONFIG_FILE_PLAT_INFO

        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO

        echo -n "ROCM VERSION: " | tee -a $CONFIG_FILE_PLAT_INFO
        if [[ -f /opt/rocm/.info/version ]] ; then
            cat /opt/rocm/.info/version | tee -a $CONFIG_FILE_PLAT_INFO
        else
            echo "Unable to determine rocm version. Is ROCm installed?"
        fi
        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
    elif [[ $p1 == "libgv" ]] ; then
        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
        echo "GPUDRIVER:   "`lsmod | egrep "^gim|^gim"` | tee -a $CONFIG_FILE_PLAT_INFO
        modinfo gim | egrep "^filename|^version" | tee -a $CONFIG_FILE_PLAT_INFO
        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
        echo "GPUDRIVER(DKMS):   "`dkms status | grep gim` | tee -a $CONFIG_FILE_PLAT_INFO
        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
    elif [[ $p1 = "" ]] ; then
        echo "Warning: p1 is empty, neither amdgpu or libgv specific logs will be gathered."
    else
        echo "warning: Unknown parameter! "
    fi
}
function ts_amdgpu_compat_summary_logs() {
    ts_helper_summary_logs amdgpu
}
function ts_libgv_compat_summary_logs() {
    ts_helper_summary_logs libgv
}

function ts_helper_guest_summary_logs() {
    vmIp=`virsh domifaddr $p1 | egrep "[0-9]+\.[0-9]+\." | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
    if [[ $DEBUG ]] ; then
        echo "ts_guest entered, p1: $p1"
        echo "guest IP: $vmIp"
    fi
	if [[ -z $/vmIp ]] ; then
        echo "Warning: Unable to determine guest O/S IP, will not gather guest logs."
		echo "Use virsh to determine running VM-s indices."
    	return 1
    fi

	#echo "VM KERNEL:	"`sshpass -p amd1234 ssh root@$vmIp 'uname -r'` | tee -a $CONFIG_FILE_PLAT_INFO

    echo $DOUBLE_BAR|  tee -a $COFNIG_FILE_PLAT_INFO
    echo "LIBGV GUEST SUMMARY: $vmIP"
    echo $DOUBLE_BAR|  tee -a $COFNIG_FILE_PLAT_INFO
	echo "VM OS:"`sshpass -p amd1234 ssh root@$vmIp 'cat /etc/os-release | grep PRETTY_NAME'`| tee -a $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "GPU.rocm-smi:"`sshpass -p amd1234 ssh root@$vmIp 'rocm-smi --showall | grep series:'`| tee -a $CONFIG_FILE_PLAT_INFO
	echo "GPU.rocminfo:"`sshpass -p amd1234 ssh root@$vmIp 'rocminfo | grep gfx'`| tee -a $CONFIG_FILE_PLAT_INFO
	echo "GPU.lspci:"`sshpass -p amd1234 ssh root@$vmIp 'lspci | grep Disp'`| tee -a $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO

	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "GPUDRIVER:"`sshpass -p amd1234 ssh root@$vmIp 'modinfo amdgpu | egrep version'`| tee -a $CONFIG_FILE_PLAT_INFO

	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "GPUDRIVER(DKMS):"`sshpass -p amd1234 ssh root@$vmIp 'dkms status | grep amdgpu'`| tee -a $CONFIG_FILE_PLAT_INFO

	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "ROCM VERSION:"`sshpass -p amd1234 ssh root@$vmIp 'cat /opt/rocm/.info/version'`| tee -a $CONFIG_FILE_PLAT_INFO
}

function ts_helper_guest_full_logs() {
    if [[ $OPTION_LIMIT_NONGPU_LOG == 1 ]] ; then
    	sshpass -p amd1234 ssh root@$vmIp 'dmesg | grep amdgpu' 2>&1 | tee -a $CONFIG_FILE_DMESG_GUEST
    else
    	sshpass -p amd1234 ssh root@$vmIp 'dmesg' 2>&1 | tee -a $CONFIG_FILE_DMESG_GUEST
    fi
    #cat /var/log/syslog | tee $CONFIG_FILE_SYSLOG_GUEST
	#cat /var/log/kern.log | tee $CONFIG_FILE_KERN_GUEST

    sshpass -p amd1234 ssh root@$vmIp 'lspci' 2>&1 | tee $CONFIG_FILE_LSPCI_GUEST
    sshpass -p amd1234 ssh root@$vmIp 'lspci -t' 2>&1 | tee $CONFIG_FILE_LSPCI_T_GUEST
    sshpass -p amd1234 ssh root@$vmIp 'lspci -vvv' 2>&1 | tee $CONFIG_FILE_LSPCI_VVV_GUEST
    sshpass -p amd1234 ssh root@$vmIp 'lspci -vvvxxx' 2>&1 | tee $CONFIG_FILE_LSPCI_VVVXXX_GUEST

    sshpass -p amd1234 ssh root@$vmIp 'dmidecode' 2>&1 | tee $CONFIG_FILE_DMIDECODE_GUEST
    sshpass -p amd1234 ssh root@$vmIp 'modinfo amdgpu' 2>&1 | tee $CONFIG_FILE_MODINFO_AMDGPU_GUEST
    sshpass -p amd1234 ssh root@$vmIp 'for i in /sys/module/amdgpu/parameters/* ; do echo -n $i: ; cat $i ; done' 2>&1 | \
        tee $CONFIG_FILE_PARM_AMDGPU_GUEST
    sshpass -p amd1234 ssh root@$vmIp 'rocminfo' 2>&1 | tee $CONFIG_FILE_ROCMINFO_GUEST
    sshpass -p amd1234 ssh root@$vmIp 'rocm-smi --showall' 2>&1 | tee $CONFIG_FILE_ROCMSMI_GUEST
}

function ts_guest() {
    ts_helper_guest_summary_logs
    ts_helper_guest_full_logs
}

AMDGPU_PRESENCE=`dkms status | grep amdgpu`
LIBGV_PRESENCE=`dkms status | grep gim`
echo "AMDGPU_PRESENCE: $AMDGPU_PRESENCE"
echo "LIBGV_PRESENCE: $LIBGV_PRESENCE"

# If both libgv and amdgpu present, libgv will take presedence...

if [[ $LIBGV_PRESENCE ]] ; then
    echo "Gathering libgv compatible logs..."
    ts_libgv_compat_summary_logs
    ts_libgv_compat_full_logs
    if [[ $1 ]] ; then
        echo "Gathering guest log: "
        ts_libgv_compat_summary_logs
        ts_guest
    fi
elif [[ $AMDGPU_PRESENCE ]] ; then 
    echo "Gathering amdgpu compatible logs..."
    ts_amdgpu_compat_summary_logs
    ts_amdgpu_compat_full_logs
else
    echo "Unable to find either amdgpu or libgv on host system."
fi

echo $DOUBLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
echo LOG FILES: $CONFIG_PATH_PLAT_INFO:
#tar -cvf $CONFIG_FILE_TAR $CONFIG_PATH_PLAT_INFO
tar -cvf $CONFIG_FILE_TAR ./ts/$DATE
tree -fs $CONFIG_PATH_PLAT_INFO

