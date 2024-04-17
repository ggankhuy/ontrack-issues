import sys

sys.path.append('/opt/rocm/share/amd_smi/')
for i in sys.path:
    print(i)

import amdsmi
amdsmi.amdsmi_init()
try:
    devices = amdsmi_get_processor_handles()
    if len(devices) == 0:
        print("No GPUs on machine")
    else:
        for device in devices:
            ecc_error_count = amdsmi_get_gpu_total_ecc_count(device)
            print(ecc_error_count["correctable_count"])
            print(ecc_error_count["uncorrectable_count"])
except Exception as e:
    print(e)

