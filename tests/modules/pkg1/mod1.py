
#
# A "global" variable.
#
# Note that python does not have truly *global* variables, only module level
# variables.
#
# Anyone wanting to access the variable must import a reference to the *module*.
# Importing the symbol (from mod1 import name_call_count) will create a *new*
# variable with the initial value of `name_call_count`. They will be different,
# independent variables!
#
# import mod1
# mod1.name_call_count += 1
#
call_count = 0


class Mod1Calculator:
    def add(self, x: int, y: int) -> int:
        global call_count
        call_count += 1
        return x + y
