rm -rf /root/rpmbuild/ ; 
make clean ; make -j`nproc` binrpm-pkg
mv /root/rpmbuild/RPMS/x86_64/kernel-*  /home/nonroot/gg/rpm/rock-kernel/config.meta/1
ls -l /home/nonroot/gg/rpm/rock-kernel/config.meta/1

make clean ; make -j`nproc` rpm-pkg
mv /root/rpmbuild/RPMS/x86_64/kernel-*  /home/nonroot/gg/rpm/rock-kernel/config.meta/2
ls -l /home/nonroot/gg/rpm/rock-kernel/config.meta/1
