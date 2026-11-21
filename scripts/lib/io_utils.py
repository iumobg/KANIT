#!/usr/bin/env python3
"""Safe subprocess execution — canonical single copy.

run_command tokenises with shlex and runs with shell=False, eliminating the
shell-injection vector: special characters in genome IDs / paths
are passed as literal argument tokens, never interpreted by a shell.
"""

import shlex
import subprocess
import sys


def run_command(command, exit_on_error=True):
    """
    Execute an external command safely (NO shell interpretation).

    Stdout is suppressed to keep console output clean; stderr is captured and
    printed on failure to aid debugging external tool errors.

    Args:
        command (str):       Command line to execute.
        exit_on_error (bool): If True (default), call sys.exit(1) on failure.
                              If False, return False on failure instead.

    Returns:
        bool: True on success. False on failure when ``exit_on_error`` is False.
    """
    try:
        subprocess.run(
            shlex.split(command),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed: {command}")
        print(f"Return code: {e.returncode}")
        if e.stderr:
            for line in e.stderr.strip().splitlines()[:5]:
                print(f"  STDERR: {line}")
        if exit_on_error:
            sys.exit(1)
        return False


def run_logged(args, log_path, cwd=None):
    """Run ``args`` (a list, no shell) in ``cwd`` with stdout and stderr written to
    ``log_path``; exit with the log's last lines if the command fails."""
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(str(a) for a in args) + "\n")
        log.flush()
        rc = subprocess.run([str(a) for a in args], stdout=log, stderr=subprocess.STDOUT,
                            cwd=cwd).returncode
    if rc != 0:
        with open(log_path, encoding="utf-8", errors="replace") as log:
            tail = log.read().splitlines()[-10:]
        sys.exit(f"ERROR: {args[0]} failed (exit {rc}); last lines of {log_path}:\n  "
                 + "\n  ".join(tail))
