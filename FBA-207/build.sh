# install rocm4.5.2
# build and install rocThrust: cd rocThrust ; mkdir cd build ; cd build ; cmake .. ; make -j8 install
hipcc code.hip -std=c++14 -I/opt/rocm/include -c -o code.o -Xclang -debug-info-kind=standalone -fdebug-types-section
