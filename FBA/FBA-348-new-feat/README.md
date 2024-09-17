to build:

1) Place file in  ./ROCm-6.2/hip-tests/perftests

1a) repo is in: https://github.com/ROCm/hip-tests
1b) should have following files in the hip-tests repo:
a.out    dispatch                    memory  stream           test_common.h  timer.h
compute  hipPerfBufferCopySpeed.cpp  module  test_common.cpp  timer.cpp

2) hipcc test_common.cpp timer.cpp -o hipPerfBufferCopySpeed.out


