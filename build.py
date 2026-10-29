#!/usr/bin/env python3
"""Pass the explicit native toolchain configuration from env.sh to Moon."""
import json
import os
import sys

json.load(sys.stdin)
json.dump({"vars": {
    "CC": os.environ.get("CC", "cc"),
    "CC_FLAGS": os.environ.get("CC_FLAGS", ""),
    "CC_LINK_FLAGS": os.environ.get("CC_LINK_FLAGS", "-lopenblas -llapacke"),
}}, sys.stdout)
