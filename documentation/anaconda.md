# Anaconda

## Installation / Configuration

```shell
# Install anaconda
$ brew cask install anaconda

# Config files are stored in ~/.conda/

# Initialize conda with the fish shell
#
# Add the following block to fish.config *before*
# the anaconda entered block
#
#######################################

set PATH /usr/local/anaconda3/bin $PATH

#
# Added because activate / deactivate wasn't working correctly
#
# See:
# https://github.com/kalefranz/conda/blob/5dd547f754a52350199dcc975848c1969d6c8931/shell/conda.fish
#
source (conda info --root)/etc/fish/conf.d/conda.fish

#####################################################################
```


## Working with the environment

```shell
# What version of Anaconda am I running?
$ conda --version

# Update anaconda
$ conda update conda

# Create a conda environment with the global python version
$ conda create --name snowflake

# Create a conda environment (use python=2.7)
$ conda create --name snowflake python=2.7

# Show environments
$ conda info --envs

# Remove a conda environment
$ conda remove -n snowflake --all

# Activate an environment
$ conda activate snowflake


```
