#set -x
SINGLE_LINE=---------------------------------------------------------------------------------------------

function usage()
{
    clear
    echo $SINGLE_LINE
    echo "examples: $0  # install rocm + amdgpu"
    echo "examples: $0 --no-dkms # install rocm only, no dkms, usually for docker env. "
    echo $SINGLE_LINE
}

function error() {
    echo "Error! $error_string"
    exit 1
}

for var in "$@"
do
    if [[ $DEBUG -eq 1 ]] ; then echo var: $var ; fi

    if [[ $var == *"--help"* ]]  ; then
        usage
        exit 0
    fi

    if [[ $var == *"--no-dkms"* ]]  ; then
        if [[ $DEBUG -eq 1 ]] ; then eecho "Will bypass dkms installation usually for docker env..." ; fi
         nodkms=1
    fi
done

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

    if [[ $i -eq amdgpu ]] && [[ ! -z nodkms ]] ; then
        echo "Will bypass dkms installation."
        continue
    fi

    yum install $i -y

    if [[ $i -eq rocm ]] ; then
        echo "Creating llvm soft links..."
        sleep 3
        for j in clang ld.lld ; do
            sudo rm -rf /usr/bin/$j
            sudo ln -s /opt/rocm/llvm/bin/$j /usr/bin/
        done
    fi
    popd
done
