MPI_DIR=/usr/mpi/gcc/openmpi-4.1.2rc2
MPI_DIR=/usr/include/openmpi-x86_64/
MPI_DIR_LIB=/usr/lib64/openmpi/lib/
all: hipIpcTest

hipIpcTest: hipIpcTest.cpp
	/opt/rocm/bin/hipcc hipIpcTest.cpp -o hipIpcTest -I$(MPI_DIR)/ -L$(MPI_DIR_LIB) -lmpi
#	/opt/rocm/bin/hipcc hipIpcTest.cpp -o hipIpcTest -I$(MPI_DIR)/ -L$(MPI_DIR)/lib -lmpi

test: hipIpcTest
	LD_LIBRARY_PATH=$(MPI_DIR)/lib $(MPI_DIR)/bin/mpirun -n 4 ./hipIpcTest

clean:
	rm -f ./hipIpcTest
