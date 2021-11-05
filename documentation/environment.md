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

[Poetry](https://python-poetry.org) is a packaging and dependency management
system that many projects use. Poetry creates it's own virtual environments
*outside* of pyenv. (run `poetry env info` to determine what virtual environment
is in use.)


## Installation

```shell

# Install pyenv
brew install pyenv pyenv-virtualenv

# Install the pyenv plugin for fish command line completion
omf update
omf install pyenv

# Install poetry via the poetry installation instructions (not via homebrew)
# https://python-poetry.org/docs/

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

```

## Poetry

Poetry installs itself to `~/.poetry`. This is *just* where the `poetry`
executable is installed. Add `~/.poetry/bin` to your path. You can also set the
`$POETRY_HOME` environment variable to have poetry install into a different
location.

```shell
# Add `poetry` to fish's $fish_user_paths
fish_add_path -g "~/.poetry/bin"

# Poetry will create virtualenvs in it's `cache-dir` configuration variable.
poetry config --list
poetry config cache-dir ~/.cache/pypoetry
```

Poetry works by creating and executing it's commands in virtual environments. A
virtual environment is created using the current python version. Therefore,
before creating the virtual environment (using `poetry init` or `poetry
install`, you need to be running a python version compatible with the project.

```shell
pyenv install 3.9.7
#
# Set the python version for the local directory.
# pyenv will switch to this python version when
# entering the local directory.
#
pyenv local 3.9.7

# Verify we are running 3.9.7
python --version

# Create a new virtual environment
poetry install
```

Note that if a python virtualenv is current active, poetry will *not* attempt to
create a virtualenv. It will use the active virtualenv. This allows you to
control the virtualenv used py poetry and also prevents the need to use `poetry
run` to run commands in the virtualenv. However it requires you to manually
active the virtualenv when you switch between projects.
