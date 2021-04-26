# Poetry

[Poetry](https://python-poetry.org/docs/)

Packaging and dependency manager.

## Environment

* Ensure the XDG directories are set (in `~/.config/fish/config.fish`)

```shell

set -U XDG_CACHE_HOME $HOME/.cache
set -U XDG_CONFIG_HOME $HOME/.config
set -U XDG_DATA_HOME $HOME/.local/share

```

* Enable tab completion:
  * `poetry completions fish > ~/.config/fish/completions/poetry.fish`

* Config file:
  * `$HOME/Library/Application Support/pypoetry/`


## Usage

Before using poetry in a project, create or set the `pyenv` virtualenv you want
to use with the project.

```shell

# create a new poetry project
poetry new [project-name]

# initialize an existing project
cd project-dir
poetry init

# Add a dependency to pyproject.toml
#
# You can add packages and version constraints to pyproject.toml manually (recommended)
#
# The following is equal to (>= 1.4.0 <2.0.0)
# pendulum = ^1.4
poetry add [dep-name]

# Install dependencies (will create poetry.lock)
# --no-root will prevent installing the project as a package.
poetry install --no-root


# Updates all out of date package (respecting semver as defined in pyproject.toml) and updates poetry.lock
poetry update

# Run a script / project

poetry run python script.py
poetry run pytest

curl --silent https://raw.githubusercontent.com/shipt/pyshipt/master/requirements.txt?token=AAAH5JJJGGF77I5F6GFVUI273IAL4 --output pyshipt-reqs.txt`
    - Run `pip install --no-deps --require-hashes --requirement pyshipt-reqs.txt
```


## Libraries

```shell

# Builds ./dist with sdist and wheel builds
poetry build

# Publishes to PyPi or a private repository
poetry publish [-r my-repository]poet
```