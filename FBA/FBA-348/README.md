to build:

1) ./ROCm-6.2/hip-tests/perftests

(should have following files in the hip-tests repo:
a.out    dispatch                    memory  stream           test_common.h  timer.h
compute  hipPerfBufferCopySpeed.cpp  module  test_common.cpp  timer.cpp

2) hipcc hipcc test_common.cpp timer.cpp -o hipPerfBufferCopySpeed.out


