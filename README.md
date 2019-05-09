# Python

This repo contains python examples as `unittest` test cases. All tests are in
the `tests` top level directory.

## Running Tests

Tests can be executed from the command line or from VS Code.
### Command Line

```sh
# Run all unit tests starting in the `tests` directory.
$ python3 -m unittest
```

## Visual Studio Code

The `.vscode/settings.json` file should contain the following config for unit
testing to work.

```json
{
    /**
     * settings.json
     *
     * VS Code workspace settings
     */
    "python.pythonPath": "python3",
    "python.unitTest.unittestArgs": [
        "-v",
        "-p",
        "test_*.py"
    ],
    "python.unitTest.unittestEnabled": true,
    "python.unitTest.pyTestEnabled": false,
    "python.unitTest.nosetestsEnabled": false
}
```