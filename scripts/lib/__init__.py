#!/usr/bin/env python3
"""
Shared library of the KANIT workflow.

Code shared by the numbered scripts.

Submodules:
    registry      — organisms.yaml / antibiotics.yaml access (single source)
    config        — global config loader + {organism}/{antibiotic} path resolver
    io_utils      — run_command (shlex-based, never shell=True)
    run_metadata  — git hash / version capture, run_id generation (MLOps)
"""
