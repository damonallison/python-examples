# Python Environment

Setting up a clean python environment isn't as simple as it should be. There are
multiple environment managers (`pyenv`, `virtualenv`, `conda`, others).

I use [asdf](https://asdf-vm.com/) to manage python and other dev tool versions
and [poetry](https://python-poetry.org/) to manage python virtual environments.

All python development should be done within a `virtualenv` to prevent polluting
the global environment.

## Installation

If you are running both x86_64 (intel) and Apple Silicon (arm64), ensure you set
your `$ASDF_DATA_DIR` env variable according to the architecture.

```shell
# NOTE: fish shell
if [ $ARCH = "i386" ]
    set -gx ASDF_DATA_DIR $HOME/.asdfx86
else
    set -gx ASDF_DATA_DIR $HOME/.asdf
end

# Install asdf
brew install asdf
```

## Installing plugins and versions

```shell
# search for a plugin
asdf plugin list all | grep dasel

# add plugins
asdf plugin add python

# update plugins
asdf plugin update --all

# find a plugin version
asdf list all python | grep 3.8

# installing a plugin version
asdf install python 3.8.15
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
