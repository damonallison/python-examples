#!/usr/bin/env python3

"""An example python program that shows how python interacts with it's environment"""

# The sys module holds information about how the script was executed. Including
# it's command line arguments.
import logging
import os
import sys
from tests.algorithms.fibonacci import fib

logger = logging.getLogger(__name__)

if not getattr(logger, "handler_set", False):
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.handler_set = True


def print_environment():
    """Prints relevant information about the current runtime environment"""

    # sys.argv will *always* contain 1 element, depending on how the script was
    # executed.
    #
    # If executed as a script (python simple.py), sys.argv[0] == "simple.py"
    # If executed as a module, (python -m simple), sys.argv[0] is the full path
    # to the script. (i.e., sys.argv[0] == "/Users/dra/projects/python-examples/simple.py"

    logger.info(f"argv len={len(sys.argv)} = {sys.argv}")

    # The module search path is the list of:
    # * Current directory
    # * $PYTHONPATH environment variable. Takes same format as $PATH (/usr/local:/usr/bin).
    # * Installation dependent default
    logger.info(f"path == {str(sys.path)}")
    logger.info(f"platform == {sys.platform}")


# If the file is being executed as a script, i.e. `python3 hellp.py`
# the module's __name__ property is set to __main__.
#
if __name__ == "__main__":
    logger.debug("hello from " + __name__)
    print_environment()
    fib.fib_to(1000)
