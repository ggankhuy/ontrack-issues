#!/bin/bash

modprobe amdgpu

PCIE_DEVICE=0000:2f:00.0
lspci -t $PCIE_DEVICE
echo ------
lspci -t $PCIE_DEVICE
echo "Capturing rocm_techsupport before SERR injection"
../rocm_techsupport.sh > before_rocm_techsupport_$pci_bus_number.log

echo "Capturing SEL logs after SERR injection on $pci_bus_number"
ipmitool -H 10.96.30.225 -U ADMIN -I lanplus -P ADMIN sel elist > after_SEL_logs_$pci_bus_number.log


echo "Injecting SERR on pci bus $pci_bus_number"
#setpci -s $pci_bus_number 60.L=00092981f
setpci -s $PCIE_DEVICE CAP_EXP+08.L=0009291f
echo "Capturing rocm_techsupport after SERR injection on $pci_bus_number"
../rocm_techsupport.sh > after_rocm_techsupport_$pci_bus_number.log

echo "Capturing SEL logs after SERR injection on $pci_bus_number"
ipmitool -H 10.96.30.225 -U ADMIN -I lanplus -P ADMIN sel elist > after_SEL_logs_$pci_bus_number.log

echo "Running TransferBench"
/root/rccl/tools/TransferBench/TransferBench example.cfg


echo "Capturing rocm_techsupport after workload"
../rocm_techsupport.sh > after_transferbench_rocm_techsupport_$pci_bus_number.log

echo "Capturing SEL logs after SERR injection on $pci_bus_number"
ipmitool -H 10.96.30.225 -U ADMIN -I lanplus -P ADMIN sel elist > after_transferbench_SEL_logs_$pci_bus_number.log
