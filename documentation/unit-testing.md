# Unit Testing

## Links

* [unittest](https://docs.python.org/3/library/unittest.html)

## Running Unit Tests

By default, `unittest` will look for tests in all packages starting in the root
directory. You can run a subset of tests in a number of ways:

* Run a single test / module.
* Run a subset of modules from a particular directory.


```sh
-v              : verbose
-s START        : the directory (or package) to start discovery

# run all tests in the ./tests directory
$ python3 -m unittest discover -v -s tests

# run all tests in the test.exceptions package.
$ python3 -m unittest discover -v -s tests.exceptions

```