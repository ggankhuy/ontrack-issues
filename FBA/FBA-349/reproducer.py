import sys

sys.path.append('/opt/rocm/share/amd_smi/')
for i in sys.path:
    print(i)

import amdsmi
amdsmi.amdsmi_init()
devices = amdsmi.amdsmi_get_processor_handles()
processes = amdsmi.amdsmi_get_gpu_process_list(devices[0])
ecc_count = amdsmi_get_gpu_total_ecc_count()
'''
try:
    process_info = amdsmi.amdsmi_get_gpu_process_info(devices[0], processes[0])
    print(process_info)
except Exception as msg:
    print("Make sure workload is running on gpu otherwise processes[0] will None and cause this.")
    print(msg)
'''