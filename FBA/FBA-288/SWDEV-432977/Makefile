MPI_DIR=/usr/mpi/gcc/openmpi-4.1.2rc2
MPI_DIR=/usr/include/openmpi-x86_64/
MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi/
all: hipIpcTest

hipIpcTest: hipIpcTest.cpp
#	ubuntu, intest.
	/opt/rocm/bin/hipcc hipIpcTest.cpp -o hipIpcTest -I$(MPI_DIR)/include -L$(MPI_DIR)/lib -lmpi
# 	worked for c9s but failed to run.
# 	/opt/rocm/bin/hipcc hipIpcTest.cpp -o hipIpcTest -I$(MPI_DIR)/ -L$(MPI_DIR_LIB) -lmpi

test: hipIpcTest
	LD_LIBRARY_PATH=$(MPI_DIR)/lib $(MPI_DIR)/bin/mpirun -n 4 ./hipIpcTest

clean:
	rm -f ./hipIpcTest
