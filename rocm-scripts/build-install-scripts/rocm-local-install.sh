set -x
yum install createrepo -y
yum install epel-release epel-next-release -y 
yum config-manager --set-enabled crb
yum install epel-release epel-next-release -y

for i in rocm amdgpu; do
    path=`find $i -name repodata`
    dir=`dirname $path`
    pushd $dir
    pwd
    createrepo .
    echo -ne "" > $i.repo
    echo "[$i]" >> $i.repo
    echo "name=$i repository" >> $i.repo
    echo "baseurl=//`pwd`" >> $i.repo
    echo "enabled=1" >> $i.repo
    echo "gpgcheck=0" >> $i.repo

    sudo yum remove $i -y
    amdgpu-install --uninstall -y
    rm -rf /etc/yum.repos.d/$i*.repo

    cp $i.repo /etc/yum.repos.d/
    yum install $i -y
    if [[ $i -eq rocm ]] ; then
        echo "Creating llvm soft links..."
        sleep 3
        for $j in clang ld.lld ; do
            sudo rm -rf /usr/bin/$i
            sudo ln -s /opt/rocm/llvm/bin/$i /usr/bin/
        done
    fi
    popd
done
