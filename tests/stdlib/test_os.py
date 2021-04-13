"""Example stdlib usage."""
# os abstracts away OS platform differences, like path separator characters. Use
# os.path functions to build and manipulate file paths in an OS agnostic way.
#
# Using `os` to perform path related activities will make your code more
# portable across platforms. All pathname manipulation should be done using
# os.path.
import os

import shutil


class TestOS:
    def test_os(self) -> None:
        """os and shutil provide platform agnostic I/O operations."""

        assert len(os.getcwd()) > 1

    def test_shutil(self) -> None:
        """shutil provides simple file and directory management functions"""
        assert shutil.disk_usage(os.getcwd()).total > 0
