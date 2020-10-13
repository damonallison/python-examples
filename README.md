# Python

This repo contains python examples as `unittest` test cases. All tests are in
the `tests` top level directory.

## Running Tests

Tests can be executed from the command line or from VS Code.

### Create a virtual environment (venv)

```shell
python3 -m pip install --user virtualenv

# The second argument is the env name
python3 -m venv myenv

# Activate
source ./env/bin/activate.fish

# Deactivate
deactivate

# Freezing dependencies
pip freeze > requirements.txt

# Installing requirements
pip install -r requirements.txt

```
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