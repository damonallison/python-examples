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

Before entering a poetry shell, set `pyenv` to a virtualenv with the version of
python you want to use with poetry. `poetry shell` will use that Python version
in it's virtualenv.



```shell

# create a new poetry project
poetry new [project-name]

# initialize an existing project
cd project-dir
poetry init

# Start a new poetry shell. This will create a new virtualenv in
# ~/.config/pypoetry/virtualenvs

poetry shell

# Add a dependency to pyproject.toml
poetry add [dep-name]

# Install dependencies (will create poetry.lock)
poetry install

# Updates all out of date package (respecting semver as defined in pyproject.toml) and updates poetry.lock
poetry update

# Run a script / project

poetry run python script.py
poetry run pytest

curl --silent https://raw.githubusercontent.com/shipt/pyshipt/master/requirements.txt?token=AAAH5JJJGGF77I5F6GFVUI273IAL4 --output pyshipt-reqs.txt`
    - Run `pip install --no-deps --require-hashes --requirement pyshipt-reqs.txt
```



