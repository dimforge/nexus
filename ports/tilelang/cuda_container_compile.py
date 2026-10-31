#!/usr/bin/env python3
"""Compatibility entrypoint; device_container_compile.py also supports Ascend."""

from device_container_compile import main

if __name__ == "__main__":
    raise SystemExit(main())
