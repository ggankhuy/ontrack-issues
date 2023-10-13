mkdir build
cd build
rm -rf ./* ; 
log_subfolder=time.trace+debug-gdwarf-aranges-gdwarf-4-g2
cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_FLAGS="-ftime-trace -gdwarf-aranges -gdwarf-4 -g2" -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS="gfx908;gfx90a"  .. 2>&1 | tee ../log/$log_subfolder/cmake.log
t1=$((SECONDS)) 
make -j`nproc` 2>&1 | tee ../log/$log_subfolder/make.log  
t2=$((SECONDS))
echo "build time, make nproc ck: $((t2-t1))" 2>&1 | tee ../log/$log_subfolder/build.time.log
