import sys

sys.path.append('/opt/rocm/share/amd_smi/')
for i in sys.path:
    print(i)

import amdsmi
amdsmi.amdsmi_init()
try:
    processes = amdsmi.amdsmi_get_gpu_process_list(devices[0])
    process_info = amdsmi.amdsmi_get_gpu_process_info(devices[0], processes[0])
    print(process_info)
except Exception as msg:
    print("Exception: Make sure workload is running on gpu otherwise processes[0] will None and cause this.")
    print(msg)

try:
    devices = amdsmi.amdsmi_get_processor_handles()
    if len(devices) == 0:
        print("No GPUs on machine")
    else:
        for device in devices:
            ecc_error_count = amdsmi.amdsmi_get_gpu_total_ecc_count(device)
            xgmi_stat = amdsmi.amdsmi_gpu_xgmi_error_status(device)
            print("corr count: (amdsmi.amdsmi_get_gpu_total_ecc_count):     ", ecc_error_count["correctable_count"])
            print("uncorr count: (amdsmi.amdsmi_get_gpu_total_ecc_count):   ", ecc_error_count["uncorrectable_count"])
            print("xgmi stat: (amdsmi.amdsmi_gpu_xgmi_error_status):        ", xgmi_stat)

except Exception as msg:
    print("Exception: ")
    print(msg)
