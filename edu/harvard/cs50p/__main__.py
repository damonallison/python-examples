"""
Week 0: Introduction to programming in python

Main topics: data types, variables, functions

Environment
-----------
* purpose of programming: solving problems by moving data
* engineering mindset
* command line
* running vs. debugging
    * python file.py
    * debugger python file.py
* syntax vs. runtime bugs

Data types, variables, and functions
------------------------------------
* data types
    * https://docs.python.org/3/library/stdtypes.html
* strict vs dynamic typing
* strings 's' vs "s" vs \"""\"""
* variables (value vs. reference)
* function declaration
* arguments (calling) vs. parameters (definition)
* functions (declared function) vs. methods (object)
* passing variables (immutable types)

Program structure
-----------------
* objects
* object oriented vs. procedural vs. functional
* scoping
* modules
* built-ins vs packages

Logistics / environment
------------------------
* python --version
* github / git
* command line
* homebrew
* pip / poetry


Types:


* int
* float
* bool
* complex
* str

* sequence types
    * list
    * tuple (immutable)
    * range

* mapping types
    * set
    * dict


"""

from edu.harvard.cs50p import week0

if __name__ == "__main__":
    # print(f"indoor voice: {week0.indoor_voice()}")
    # print(f"playback speed: {week0.playback_speed()}")
    # print(f"making faces: {week0.making_faces()}")
    # print(f"eninsten: {week0.einstein()}")
    print(f"tip calculator: {week0.tip_calculator()}")
