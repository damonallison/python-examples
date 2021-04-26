# Python Environment

Setting up a clean python environment isn't as simple as it should be. There are
multiple environment managers (`pyenv`, `virtualenv`, `conda`, others).

I use [pyenv](https://github.com/pyenv/pyenv) to manage python versions and
[pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) to create virtual
environments. `pyenv` simplifies installing and managing multiple python
versions, and `virtualenv` allows for creating multiple environments, each which
uses a different python version.

All python development should be done within a `virtualenv` to prevent polluting
the global environment.

## Installation

```shell

# Install pyenv
brew install pyenv pyenv-virtualenv

# Install the pyenv plugin for fish command line completion
omf update
omf install pyenv
```

## Configuration

Enable auto-activation of `pyenv` and `pyenv-virtualenv` by adding this to your
`config.fish`.

```shell
#
# python
#
# Enables auto-activation of pyenv and virtual environments.
# https://github.com/pyenv/pyenv-virtualenv
#
status --is-interactive; and pyenv init - | source
status --is-interactive; and pyenv virtualenv-init - | source
```

## pyenv

```shell

# List / install available python versions
pyenv install --list
pyenv install 3.9.0

# List the installed python versions
pyenv versions

# Specify a python version to use in the current directory (creates .python-version)
pyenv local 3.9.0

# Set the global version of python to use
pyenv global 3.9.0

# Create a virtual environment
pyenv virtualenv test

# Activate / deactivate a virtual environment
pyenv activate test
pyenv deactivate

# Delete a virtualenv (both commands are identical)
pyenv virtualenv-delete `env`
pyenv uninstall `env`

````