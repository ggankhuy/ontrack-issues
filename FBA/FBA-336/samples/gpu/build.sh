rm *.out
hipcc -fsanitize=address p61.cpp -o p61.fsanitize.out
hipcc -fsanitize=address -g  -shared-libsan  p61.cpp -o p61.fsanitize.shared-libsan.g.out
hipcc -fsanitize=address -g  -shared-libsan --offload-arch=gfx908:xnack+ p61.cpp -o p61.fsanitize.shared-libsan.g.offload.908.out
ls -l *.out
