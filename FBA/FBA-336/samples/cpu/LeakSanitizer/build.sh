#clang++ -fsanitize=address -g memory-leak.c
clang -fsanitize=leak -g -fno-omit-frame-pointer memory-leak.c
