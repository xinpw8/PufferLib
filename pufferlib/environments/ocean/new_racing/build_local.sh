#!/bin/bash

# new build_local.sh
cython new_cy_racing.pyx --cplus -3 --fast-fail -v
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.8 -o new_cy_racing.so new_cy_racing.cpp new_racing.h
