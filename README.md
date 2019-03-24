# Python

## Likes

* Flexible. Simple to embed into other programs.

## Dislikes

* Dynamic. No type checking, type information.

* The core data structures (except tuple) are mutable by default.
    * Swift's `val` and `var` are much more clear, safe.

* Built-in functions vs. objects. Built-ins should be on objects.
    * `del` seems like a complete hack. Put `delete` on all data structure types.
    * `sorted`
    * `zip`
    * `enumerate`

* Heterogeneous lists.
    * `array()` stores homogeneous lists. Specifying an array type feels awkaard.

* Exceptions are all named with `Error` rather than `Exception`.

* Mutliple base classes

* `self`

* Data encapsulation is impossible in python. It's done by convention. Yuk.

> In fact, nothing in Python makes it possible to enforce data hiding — it is
> all based upon convention.

> “Private” instance variables that cannot be accessed except from inside an
> object don’t exist in Python. However, there is a convention that is followed
> by most Python code: a name prefixed with an underscore (e.g. _spam) should be
> treated as a non-public part of the API (whether it is a function, a method or
> a data member). It should be considered an implementation detail and subject
> to change without notice.`

## Comments

* Primary Collection Types : List, Tuple (immutable), Set, Dictionary
    * Any hashable object can be a dictionary key. (How does it enforce?)

## Namespaces

* Built-ins (`abs()`)
* Module global
* Function

### Scopes (Lexical)

* Innermost scope (block or function)
* Enclosing scope (outer function or class)
* Module global scope
* Built-ins

* `nonlocal` binds a variable higher in the enclosing scope to the current scope.
* `global` binds a variable in the global scope to the current scope.


## Classes

* Visibility rules / modification.
