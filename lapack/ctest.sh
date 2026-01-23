#!/bin/bash

# pushd to ctest
# run make run

pushd ctest
make run
if [ $? -ne 0 ]; then
    echo "Make run failed"
    rm -f *.o *.s ctest
    exit 1
fi
