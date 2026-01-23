#!/bin/bash

script_dir="$(cd "$(dirname "$0")" && pwd)"
pushd "$script_dir/ctest" >/dev/null
make run
if [ $? -ne 0 ]; then
    echo "Make run failed"
    rm -f *.o *.s ctest
    exit 1
fi
popd >/dev/null
