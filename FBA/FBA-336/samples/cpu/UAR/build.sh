clang++ -O1 -g -fsanitize-address-use-after-return=always -fno-omit-frame-pointer example_UseAfterFree.cc
