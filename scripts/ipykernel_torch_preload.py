"""Launch ipykernel after preloading torch on Windows.

PyZMQ can load native DLLs before notebook code runs. On some Windows
installs, loading pyzmq before PyTorch makes torch fail while loading c10.dll with
WinError 1114. This wrapper imports torch before ipykernel imports zmq, then
hands control to the normal ipykernel launcher.
"""

from __future__ import annotations

import os
import runpy

os.environ.setdefault("KERAS_BACKEND", "torch")

# Import torch before ipykernel/pyzmq initialize their native DLLs.
import torch  # noqa: F401, E402

runpy.run_module("ipykernel_launcher", run_name="__main__", alter_sys=True)
