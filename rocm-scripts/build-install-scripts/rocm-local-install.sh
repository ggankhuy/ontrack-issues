#set -x
for i in rocm amdgpu; do
    path=`find $i -name repodata`
    dir=`dirname $path`
    pushd $dir
    pwd
    createrepo .
    popd
done
