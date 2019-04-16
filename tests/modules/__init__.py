"""
An example __init__.py file

When a package is first loaded, code in __init__.py is executed.

There is a special __all__variable that can be set which lists a set of modules
which will be imported when doing a "wildcard" import:

from pkg import *

This type of import should *not* be used from code, only from an interactive
shell.
"""

print("Initializing package " + __name__)
#
# Packages support a special __path__ atrribute.
#
# Updating this variable will affect future searches for modules and
# subpackages contained in this package. You *could*
#
print("Path == " + str(__path__))

__all__ = ["calculator"]
