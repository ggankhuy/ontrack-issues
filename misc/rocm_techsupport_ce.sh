#!/bin/sh
# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
#
# rocm_techsupport.sh
# This script collects ROCm and system logs on a Debian OS installation.
# It requires 'sudo' supervisor privileges for some log collection
# such as dmidecode, dmesg, lspci -vvv to read capabilities.
# Author: srinivasan.subramanian@amd.com
# Fork: ggankhuy@amd.com
# Revision: V1.30
# V1.30: added tar option, simplified many commands, add log outputs.
# V1.29: check 5.0 version
# V1.28: add timestamp
# V1.27: grab repo setup info
# V1.26: grep amdgpu-dkms packages (new in 4.5)
# V1.25: grab amdgpu udev rules, lsinitrd/ramfs
# V1.24: add dkms status
# V1.23: add 4.0 check
# V1.22: workaround ROCm 3.9 rocm-smi bug
# V1.21: fix 3.10 detect
# V1.20: check openmp-extras package install
# V1.19: support for 3.10 release
# V1.18: add pnp filter
# V1.17: dump pids /sys/class/kfd/kfd/proc/
# V1.16: Add nmi to grep, lspci -t
# V1.15: Add rdc ROCm package filter, env, ldcache
# V1.14: List PCIe current link width/speed
# V1.12: List available ROCm versions under /opt
# V1.11: Run dmesg in addition to journalctl
# V1.10: get ROCm related ldconf entries
# V1.9: Add kfd to message filter
# V1.8: Add vfio to message filter
# V1.7: Check persistent logging status
#       add panic, oop, fail, xgmi to grep filter
# V1.6: In docker, use dmesg
# V1.5: Add error and fault to system log filter
# V1.4: Collect logs from the last 3 boots
#       Add power to grep
# V1.3: Add rocm-bandwidth-test -t to get topology
# V1.2: Add ECC and rask_mask
#       Show ras info, xgmierr
# V1.1: Detect OS type
#       Check paths for lspci, lshw
# V1.0: Initial version
#

DATE=`date +%Y%m%d-%H-%M-%S`
LOGFILE_KERNEL_OS=$DATE-kernel-os.log
LOGFILE_CPU=$DATE-cpu.log
LOGFILE_DMESG=$DATE-dmesg.log
LOGFILE_MEM==$DATE-mem.log
LOGFILE_LSPCI=$DATE-lspci.log
LOGFILE_ROCM_SMI=$DATE-rocm-smi.log
LOGFILE_ROCMINFO=$DATE-rocminfo.log
LOGFILE_DMIDECODE=$DATE-dmidecode.log
LOGFILE_AMDGPU=$DATE-amdgpu-params.log
LOGFILE_DKMS=$DATE-dkms.log
LOGFILE_ROCM=$DATE-rocm.log
CONFIG_ENABLE_HISTORY=1
LOGFILE_HISTORY=$DATE-history.log
LOGFILE_BOOTINFO=$DATE-bootinfo.log
LOG_FOLDER=./log-$DATE
sudo mkdir $LOG_FOLDER
echo "=== ROCm TechSupport Log Collection Utility: V1.29 ==="
/bin/date

ret=`/bin/grep -i -E 'debian|ubuntu' /etc/os-release`
if [ $? -ne 0 ]
then
    pkgtype="rpm"
else
    pkgtype="deb"
fi
echo "===== Section: OS Distribution         ==============="
# Print OS type
/bin/uname -a 2>&1 | sudo tee $LOG_FOLDER/$LOGFILE_KERNEL_OS
# OS release
/bin/cat /etc/os-release 2>&1 | sudo tee $LOG_FOLDER/$LOGFILE_KERNEL_OS

# Kernel boot parameters
echo "===== Section: Kernel Boot Parameters  ==============="
/bin/cat /proc/cmdline 2>&1 | sudo tee $LOG_FOLDER/$LOGFILE_CPU

# System log related to GPU
echo "===== Section: dmesg GPU/DRM/ATOM/BIOS ==============="
if [ -f /var/log/journal ]
then
    echo "Persistent logging enabled."
else
    echo "WARNING: Persistent logging possibly disabled."
    echo "WARNING: Please run: "
    echo "       sudo mkdir -p /var/log/journal "
    echo "       sudo systemctl restart systemd-journald.service "
    echo "WARNING: to enable persistent boot logs for collection and analysis."
fi

dockerchk=`/bin/grep cpuset /proc/1/cgroup | /usr/bin/awk 'BEGIN {FS=":"} {print $3}'`
echo dockercheck: $dockercheck...
sleep 5
if [ "$dockerchk" != "/" ]
then
    echo "Section: Current boot logs"
    /bin/dmesg -T | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|nmi|dazed|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER'

elif [ -f /bin/journalctl ]
then
    echo "Section: dmesg boot logs"
    /bin/dmesg -T | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|nmi|dazed|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER'
    echo "Section: Current boot logs"
    /bin/journalctl -b | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|nmi|dazed|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER'
    echo "Section: Previous boot logs"
    /bin/journalctl -b -1 | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|nmi|dazed|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER'
    echo "Section: Second Previous boot logs"
    /bin/journalctl -b -2 | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|nmi|dazed|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER'
    echo "Section: Third Previous boot logs"
    /bin/journalctl -b -3 | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|nmi|dazed|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER'
elif [ -f /usr/bin/journalctl ]
then
    echo "Section: dmesg boot logs"
    /bin/dmesg -T | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|nmi|dazed|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER'
    echo "Section: Current boot logs"
    /usr/bin/journalctl -b | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|nmi|dazed|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER'
    echo "Section: Previous boot logs"
    /usr/bin/journalctl -b -1 | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|nmi|dazed|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER'
    echo "Section: Second Previous boot logs"
    /usr/bin/journalctl -b -2 | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|nmi|dazed|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER'
    echo "Section: Third Previous boot logs"
    /usr/bin/journalctl -b -3 | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|nmi|dazed|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER'
else
    echo "Section: dmesg boot logs"
    /bin/dmesg -T | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|nmi|dazed|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER'
    echo "ROCmTechSupportNotFound: journalctl utility not found!"
fi

dmesg | sudo tee $LOG_FOLDER/$LOGFILE_DMESG

# CPU information
echo "===== Section: CPU Information         ==============="
/usr/bin/lscpu | sudo tee -a $LOG_FOLDER/$LOGFILE_CPU

# Memory information
echo "===== Section: Memory Information      ==============="
/usr/bin/lsmem $LOG_FOLDER/$LOGFILE_MEM

# Hardware Information
echo "===== Section: Hardware Information    ==============="
if [ -f /usr/bin/lshw ]
then
    /usr/bin/lshw
elif [ -f /usr/sbin/lshw ]
then
    /usr/sbin/lshw
elif [ -f /sbin/lshw ]
then
    /sbin/lshw
else
    echo "Note: Install lshw to get lshw hardware listing information"
    echo "    Ex: sudo apt install lshw "
    echo "ROCmTechSupportNotFound: lshw utility not found!"
fi

# Kernel Modules loaded
echo "===== Section: lsmod loaded module     ==============="
/sbin/lsmod

# amdgpu modinfo
echo "===== Section: amdgpu modinfo          ==============="
modinfo amdgpu | tee -a $LOG_FOLDER/$LOGFILE_AMDGPU

# dkms status
echo "===== Section: dkms status             ==============="
dkms status | tee -a $LOG_FOLDER/$LOGFILE_DKMS

# amdgpu udev rules
echo "===== Section: amdgpu udev rule        ==============="
cat /etc/udev/rules.d/70-amdgpu.rules

# lsinitrd or lsinitramfs dump
echo "===== Section: lsinitrd lsinitramfs    ==============="
if [ "$pkgtype" = "deb" ]
then
    /usr/bin/lsinitramfs /boot/initrd.img-$(/bin/uname -r)
else
    sudo /usr/bin/lsinitrd
fi

# Hardware Topology
echo "===== Section: Hardware Topology       ==============="
if [ -f /usr/bin/lstopo-no-graphics ]
then
    /usr/bin/lstopo-no-graphics
else
    echo "lstopo command not found. Skipping hardware topology information."
    echo "Note: Install hwloc to get lstopo hardware topology information"
    echo "    Ex: sudo apt install hwloc "
fi

# DMIdecode information - BIOS etc
echo "===== Section: dmidecode Information   ==============="

if [[ `which dmidecode` ]] ; then
    sudo dmidecode | sudo tee $LOG_FOLDER/$LOGFILE_DMIDECODE
fi

# PCI peripheral information
echo "===== Section: lspci verbose output    ==============="


if [[ `which lspci` ]] ; then
    lspci -t | sudo tee $LOG_FOLDER/$LOGFILE_LSPCI
    lspci -vvv  | sudo tee -a $LOG_FOLDER/$LOGFILE_LSPCI    
else
    echo "ROCmTechSupportNotFound: lspci utility not found!"
fi

# Print ROCm repo setup
echo "===== Section: ROCm Repo Setup         ==============="
/bin/grep -i -E 'rocm|amdgpu' /etc/apt/sources.list.d/* /etc/zypp/repos.d/* /etc/yum.repos.d/*

# Print ROCm installed packages
echo "===== Section: ROCm Packages Installed ==============="
if [ "$pkgtype" = "deb" ]
then
    /usr/bin/dpkg -l | /bin/grep -i -E 'ocl-icd|kfdtest|llvm-amd|miopen|half|^ii  hip|hcc|hsa|rocm|atmi|^ii  comgr|aomp|amdgpu|rock|mivision|migraph|rocprofiler|roctracer|rocbl|hipify|rocsol|rocthr|rocff|rocalu|rocprim|rocrand|rccl|rocspar|rdc|openmp' | /usr/bin/sort
else
    /usr/bin/rpm -qa | /bin/grep -i -E 'ocl-icd|kfdtest|llvm-amd|miopen|half|hip|hcc|hsa|rocm|atmi|comgr|aomp|amdgpu|rock|mivision|migraph|rocprofiler|roctracer|rocblas|hipify|rocsol|rocthr|rocff|rocalu|rocprim|rocrand|rccl|rocspar|rdc|openmp' | /usr/bin/sort
fi

# Log ROCm related ldconfig entries
echo "===== Section: ROCm ldconfig entries   ==============="
/bin/grep -i 'rocm' /etc/ld.so.conf.d/*

# Dump ROCm cached ldconfig entries
echo "===== Section: ROCm ldcache entries      ============="
ldconfig -p | grep rocm

# Dump ROCm related environmental variables
echo "===== Section: ROCm environment variables============="
env | /bin/grep -i 'rocm'

# Select latest ROCM installed version: only supports 3.1 or newer
echo "===== Section: Available ROCm versions ==============="
if [ "$ROCM_VERSION"x = "x" ]; then
    /bin/ls -v -d /opt/rocm*
    ROCM_VERSION=`/bin/ls -v -d /opt/rocm-[3-5]* | /usr/bin/tail -1`
    if [ "$ROCM_VERSION"x = "x" ]
    then
        ROCM_VERSION=`/bin/ls -v -d /opt/rocm* | /usr/bin/tail -1`
    fi
fi 
echo "==== Using $ROCM_VERSION to collect ROCm information.==== "

# RBT Topology
echo "===== Section: rocm-bandwidth-test Topology       ==============="
if [ -f $ROCM_VERSION/bin/rocm-bandwidth-test ]
then
    $ROCM_VERSION/bin/rocm-bandwidth-test -t
else
    echo "$ROCM_VERSION/bin/rocm-bandwidth-test command not found. Skipping topology information."
    echo "Note: Install rocb=m-bandwidth-test ROCm package to get topology information"
    echo "    Ex: sudo apt install rocm-bandwidth-test "
fi

# ROCm SMI # why not just use show all?
if [ -f `which rocm-smi` ] ; then
    echo "===== Section: ROCm SMI                ==============="
    rocm-smi --showall | sudo tee -a $LOG_FOLDER/$LOGFILE_ROCM_SMI
else
    echo " rocm-smi NOT FOUND !!! "
fi

# should get dynamically the number of gpus.

echo "===== Section: GPU PCIe Link Config    ==============="
GPUs=`lspci | grep Disp`
for i in $(seq 0 $GPUs)
do
    echo "GPU $i PCIe Link Width Speed: "
    cat /sys/class/drm/card$i/device/current_link_width
    cat /sys/class/drm/card$i/device/current_link_speed
done

echo "===== Section: KFD PIDs sysfs kfd proc ==============="
ls /sys/class/kfd/kfd/proc/

# ROCm Agent Information

if [[ -f `which rocminfo` ]]
then
    echo "===== Section: rocminfo                ==============="
    rocminfo | sudo tee $LOG_FOLDER/$LOGFILE_ROCMINFO
else	
	echo "rocminfo not found!!!"
fi

# OpenCL Agent Information
if [[ `which clinfo` ]]
then
    echo "===== Section: clinfo                  ==============="
    clinfo
fi

#command history

if [[ $CONFIG_ENABLE_HISTORY -eq 1 ]] ; then
    echo "Gathering history..."
    export HISTTIMEFORMAT="%F %T " ; history 2>&1 | sudo tee $LOG_FOLDER/$LOGFILE_HISTORY
else
    echo "Gather history is disabled."
fi

#grub

echo " ===== Section: boot information          ================" 
echo "grub: /etc/default/grub" | sudo tee  $LOG_FOLDER/$LOGFILE_BOOTINFO
cat /etc/default/grub | sudo tee  $LOG_FOLDER/$LOGFILE_BOOTINFO
echo "/sys/firmware/efi: " | sudo tee  $LOG_FOLDER/$LOGFILE_BOOTINFO
ls -l /sys/firmware/efi | sudo tee -a $LOG_FOLDER/$LOGFILE_BOOTINFO
echo "grub.cfg for efi mode: " | sudo tee  $LOG_FOLDER/$LOGFILE_BOOTINFO
cat /boot/efi/EFI/centos/grub.cfg | sudo tee -a $LOG_FOLDER/$LOGFILE_BOOTINFO
cat /boot/grub2/grub.cfg | sudo tee -a $LOG_FOLDER/$LOGFILE_BOOTINFO

echo " ====== kmod /amdgpu/ driver parameters =================="
kmod_params=`modinfo amdgpu | grep parm | tr -d ' ' | cut -d ':' -f2`

for i in $kmod_params ; do
    filename=`find /sys -name $i`
    if [[ -f $filename ]] ; then
        value=`cat $filename`
        echo $filename: $value1: $value | tee -a $LOG_FOLDER/$LOGFILE_AMDGPU 
    else
        echo "bypassing $i /not a file/"
    fi

done

echo "Creating tar..."
tar -cvf $DATE.tar $LOG_FOLDER/*

echo "Watchdog info: /proc/sys/kernel/watchdog_thres: " | sudo tee -a $LOG_FOLDER/$LOGFILE_CPU
cat /proc/sys/kernel/watchdog_thresh | sudo tee -a $LOG_FOLDER/$LOGFILE_CPU

yum list installed | grep -i rocm | sudo tee -a $LOG_FOLDER/$LOGFILE_ROCM
