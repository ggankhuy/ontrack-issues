#!/bin/bash

set -ex

[ -n "$CMAKE_VERSION" ]

install_ubuntu() {
    # Remove system cmake install so it won't get used instead
    apt-get remove cmake -y
}

install_centos() {
    # Remove system cmake install so it won't get used instead
    yum remove cmake -y
}
# Install Python packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  centos)
    install_centos
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac


    # Turn 3.6.3 into v3.6
path=$(echo "${CMAKE_VERSION}" | sed -e 's/\([0-9].[0-9]\+\).*/v\1/')
file="cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz"

# Download and install specific CMake version in /usr/local
pushd /tmp
curl -Os --retry 3 "https://cmake.org/files/${path}/${file}"
tar -C /usr/local --strip-components 1 --no-same-owner -zxf cmake-*.tar.gz
rm -f cmake-*.tar.gz
popd
