hipcc ipc.hip


    compiler_flags = [
        "-Wno-error=switch",
        "-Wno-error=int-in-bool-context",
    ],
    hip_flags = get_rocm_arch_args() + [
        "-Wno-error=switch",
        "-Wno-error=int-in-bool-context",
        "-Wno-error=unused-result",
        "-Wno-error=format",
