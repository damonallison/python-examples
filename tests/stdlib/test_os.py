"""Example stdlib usage.

Important standard library modules:

* sys - command line arguments, environment (stderr, environment variables)
* os - interact with the operating system.
* shutil - high level interface for dealing with files
* glob - make list of filenames using wildcards
* re - regular expressions
* urllib.request - HTTP
* datetime
* timeit, profile, pstats - performance measurement
* json
* threading
* logging
* decimal - more precise 'float' - use for financial apps, when you need to
  control precision and/or rounding.

"""
# os abstracts away OS platform differences, like path separator characters. Use
# os.path functions to build and manipulate file paths in an OS agnostic way.
#
# Using `os` to perform path related activities will make your code more
# portable across platforms. All pathname manipulation should be done using
# os.path.
import os
import logging
import shutil


def test_needs_tmp_dir() -> None:
    assert 1


def test_os() -> None:
    """os and shutil provide platform agnostic I/O operations."""

    assert len(os.getcwd()) > 1
    assert shutil.disk_usage(os.getcwd()).total > 0
