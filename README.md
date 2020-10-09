# Python

This repo contains python examples as `unittest` test cases. All tests are in
the `tests` top level directory.

## Running Tests

Tests can be executed from the command line or from VS Code.

### Command Line

```sh
# Run all unit tests
$ pip3 install pytest
$ pytest
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
    "python.unitTest.unittestEnabled": false,
    "python.unitTest.pyTestEnabled": true,
    "python.unitTest.nosetestsEnabled": false
}
```