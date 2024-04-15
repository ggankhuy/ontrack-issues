import amdsmi

amdsmi.amdsmi_init()
devices = amdsmi.amdsmi_get_processor_handles()
processes = amdsmi.amdsmi_get_gpu_process_list(devices[0])
process_info = amdsmi.amdsmi_get_gpu_process_info(devices[0], processes[0])
print(process_info)
