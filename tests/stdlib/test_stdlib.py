"""Example stdlib usage."""

import unittest

#
# Abstracts away OS platform differences, like path separator characters.
# Use os.path functions to build and manipulate file paths in an OS agnostic way.
#
import os
import shutil
import logging


class TestStdLib(unittest.TestCase):
    """Examples showing stdlib usage."""

    @classmethod
    def setUpClass(cls) -> None:
        # warm up the root logger
        logging.basicConfig()
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self) -> None:

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def tearDown(self) -> None:
        pass

    def test_os(self) -> None:
        """os and shutil provide platform agnostic I/O operations."""

        self.assertTrue(len(os.getcwd()) > 1, msg="Retrieve current working directory failed")
        self.assertTrue(shutil.disk_usage(os.getcwd()).total > 0, "Disk usage for the volume returned 0")
