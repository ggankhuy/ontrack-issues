#!/bin/bash
p1=$1

if [[ -z $p1  ]] ; then
    echo "Please specify container name...."
    exit 1
fi

container=docker.gpuperf:5000/dcgpu/ansys:v23r2_1

docker run -it --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --name=$p1 \
      -v ./out:/benchmark/results \
      -w /benchmark \
      ${container} \
      bash

