#!/usr/bin/env python3

"""An example python program that shows how python interacts with it's environment"""

# The sys module holds information about how the script was executed. Including
# it's command line arguments.
import logging
import sys
import pkg_resources

logger = logging.getLogger(__name__)

if not getattr(logger, "handler_set", False):
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.handler_set = True


def print_environment():
    """Prints relevant information about the current runtime environment"""
    print(f"python version: {sys.version}")

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

    print("--- packages ---")
    print(sorted([f"{i.key} = {i.version}" for i in pkg_resources.working_set]))
    print("--- END packages ---")


# If the file is being executed as a script, i.e. `python3 hellp.py`
# the module's __name__ property is set to __main__.
#
if __name__ == "__main__":
    logger.debug("hello from " + __name__)
    print_environment()
