#!/usr/bin/bash
FILENAME=4-streams-vector
hipcc $FILENAME.cpp -o $FILENAME.out

for i in {1..32} ; do
    sudo mkdir ln/$i
    ln -s `pwd`/$FILENAME.out ln/$i/$FILENAME.out
    cd ln/$i 
    sudo rocprof --sys-trace ./$FILENAME.out &
    cd ../..
done
