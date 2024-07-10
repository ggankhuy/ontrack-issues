import sys

sys.path.append('/opt/rocm/share/amd_smi/')
for i in sys.path:
    print(i)

import amdsmi
amdsmi.amdsmi_init()

devices = amdsmi.amdsmi_get_processor_handles()

print("--- test: amdsmi.amdsmi_get_process_info.")
for device in devices:
    try:
        processes = amdsmi.amdsmi_get_gpu_process_list(device)
        process_info = amdsmi.amdsmi_get_gpu_process_info(device, processes[0])
        print(process_info)

    except Exception as msg:
        print(msg)

fields=[\
    ['average_socket_power','gfx_voltage','power_limit'],\
    ['vram_used','vram_total'],\
    ['correctable_count','uncorrectable_count'],\
    ['gfx_activity','umc_activity','mm_activity'],\
    []\
    ]

idx=0
for i in [\
    amdsmi.amdsmi_get_power_info, \
    amdsmi.amdsmi_get_gpu_vram_usage, \
    amdsmi.amdsmi_get_gpu_total_ecc_count,\
    amdsmi.amdsmi_get_gpu_activity, \
    amdsmi.amdsmi_gpu_xgmi_error_status\
    ]:

    print(" ---- TESTING : ", i)
    try:
        for device in devices:
            result=i(device)

            if fields[idx]:
                for j in fields[idx]:
                    print(j, ": ", result[j])
            else:
                print("!! no fields.")
                print(j, ": ", result)
               
    except Exception as msg:
        print("Error:")
        print(msg)
    idx = idx + 1

