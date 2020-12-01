# Python Examples

Simple python examples as unit tests.

## Running Tests

```shell
pip install -U pytest
pytest

```

### Create a virtual environment (venv)

```shell
pip install virtualenv

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