hipcc reproduce.cpp/
#rocprofv2 --hip-trace ./a.out
rocprofv2 --hip-trace --plugin perfetto -d out ./a.out
