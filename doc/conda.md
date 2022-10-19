# Conda

Packaging tool and installer for Python virtual environments with separate versions of Python and python package manager.

## Understanding conda & pip

From [StackOverflow](https://stackoverflow.com/questions/20994716/what-is-the-difference-between-pip-and-conda)

Quoting from the [Conda blog](http://web.archive.org/web/20170415041123/www.continuum.io/blog/developer-blog/python-packages-and-environments-conda):

> Having been involved in the python world for so long, we are all aware of pip, easy_install, and virtualenv, but these tools did not meet all of our specific requirements. The main problem is that they are focused around Python, neglecting non-Python library dependencies, such as HDF5, MKL, LLVM, etc., which do not have a setup.py in their source code and also do not install files into Python’s site-packages directory.

So Conda is a packaging tool and installer that aims to do more than what `pip` does; handle library dependencies *outside* of the Python packages as well as the Python packages themselves. Conda also creates a virtual environment, like `virtualenv` does.

As such, Conda should be compared to [Buildout](http://www.buildout.org/en/latest/) perhaps, another tool that lets you handle both Python and non-Python installation tasks.

Because Conda introduces a new packaging format, you cannot use `pip` and Conda interchangeably; `pip` cannot install the Conda package format. You can use the two tools side by side (by installing `pip` with `conda install pip`) but they do not interoperate either.

Since writing this answer, Anaconda has published a [new page on *Understanding Conda and Pip*](https://www.anaconda.com/understanding-conda-and-pip/), which echoes this as well:

> This highlights a key difference between conda and pip. Pip installs Python packages whereas conda installs packages which may contain software written in any language. For example, before using pip, a Python interpreter must be installed via a system package manager or by downloading and running an installer. Conda on the other hand can install Python packages as well as the Python interpreter directly.

and further on

> Occasionally a package is needed which is not available as a conda package but is available on PyPI and can be installed with pip. In these cases, it makes sense to try to use both conda and pip.

## install conda

Get a newer version of Python. Follow the [guide here](https://linuxize.com/post/how-to-install-python-3-9-on-ubuntu-20-04/).

Download `conda`:

1. `sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6`
2. Download `conda` installer from [the official website](https://www.anaconda.com/products/individual#linux).
3. Run `conda` installer: `bash Anaconda3-2021.05-Linux-x86_64.sh`.
4. Reload terminal configs: `source ~/.bashrc`.
5. Check installation:

```shell
(base) onyr@laerys:~/Downloads$ conda --version
conda 4.10.1
```

6. Update `conda`: `conda update conda`


## Conda-forge

> [conda-forge](https://github.com/conda-forge) is a GitHub organization containing repositories of conda recipes. Thanks to some awesome continuous integration providers (AppVeyor, Azure Pipelines, CircleCI and TravisCI), each repository, also known as a feedstock, automatically builds its own recipe in a clean and repeatable way on Windows, Linux and OSX.

[conda-forge](https://conda-forge.org/)

Setup `conda-forge`:

1. `conda config --add channels conda-forge`
2. `conda config --set channel_priority strict`
3. `conda install <package-name>`

## export & create conda env

`conda env export > environment.yml`: Export an environment to a YAML file that can be read on Windows, macOS, and Linux.

`conda env create --file environment.yml`: Create an environment from YAML file.

## Download new packages

Always prefer to use the `pip` from the conda environment:

1. Activate conda: `conda activate <env-name>`
2. Install package with `pip`: `pip install <package-name>`

You can then update the `requirements.txt` and `environment.yml` directly from within the current working conda env:

1. Recreate environment.yml for conda: `conda env export > environment.yml`
2. Create requirements.txt for pip: `pip freeze > requirements.txt`

## Conda commands

> See [here](https://conda.io/projects/conda/en/latest/user-guide/cheatsheet.html) for `conda` command cheatsheet.

`conda -V`: check that `conda` is installed.

`conda update conda`: update `conda`.

`conda create -n venv1 python=3.9.5`: create a virtual environment with specified version of Python.

`conda activate venv1`: work on specified `venv`.

`conda deactivate`: return to base.

`conda info -e`: list virtual environments.

`conda create --name venv3.8 python=3.8`: create a virtual environment with specified version of python.

`while read requirement; do conda install --yes $requirement; done < requirements.txt`: install packages from a `requirements.txt` file using conda.

> with pip: `while read requirement; do pip3 install $requirement; done < requirements.pip.txt`

`conda install <package>`: install a package into a conda environment. Make sure to be inside the right environment.

`conda list`: list installed packaged.

`conda env list` | `conda info --envs` : list all virtual environments available on the machine.

`conda env export --name ENVNAME > envname.yml`: Export an environment to a YAML file that can be read on Windows, macOS, and Linux.

`conda env create --file envname.yml`: Create an environment from YAML file.

`conda env remove --name <envname>`: Delete an environment and everything in it.

## Troubleshooting

##### dissable conda adding (base) before command prompt

> This (base) thing is actually quite handy if you are using Python and conda a lot so I advice you to keep it.

[How to remove (base) from terminal prompt after updating conda](https://stackoverflow.com/questions/55171696/how-to-remove-base-from-terminal-prompt-after-updating-conda)

By default, `auto_activate_base` is set to `True` when installing anaconda. To check this, run:

```
$ conda config --show | grep auto_activate_base
auto_activate_base: True
```

To set it `False`

```
conda config --set auto_activate_base False
```

and vice-versa.

Note, if `changeps1` is kept `False`, it will hide `(env)` completely, and in case you want to show `(env)` only when it's activated, you can set `changeps1` to `True`:

```
conda config --set changeps1 True
```

> Setting `changeps1` to `False` will hide `(env)` even if the `env` is activated and will keep hiding `(base)` even after `auto_activate_base` is set to `True`.

```shell
(base) onyr@aezyr:~/anaconda3/etc$ conda config --show | grep auto_activate_base
auto_activate_base: True
(base) onyr@aezyr:~/anaconda3/etc$ conda config --set auto_activate_base False
(base) onyr@aezyr:~/anaconda3/etc$ source ~/.bashrc 
onyr@aezyr:~/anaconda3/etc$ conda config --show | grep auto_activate_base
auto_activate_base: False
```

##### conda with VSCode

It seems that VSCode stays at (base) level, and not inside the right venv, even when we `conda activate venv1` in a VSCode integrated terminal.

Need to click on the Python version and select the one whose path is the one of the virtual environment.

##### adding a new interpreter path (env) to VSCode

See [StackOverflow](https://stackoverflow.com/questions/66869413/visual-studio-code-does-not-detect-virtual-environments/68169595#68169595):

1. In VSCode open your command palette — `Ctrl+Shift+P` by default
2. Look for `Python: Select Interpreter`
3. In `Select Interpreter` choose `Enter interpreter path...` and then `Find...`
4. Navigate to your `venv` folder — eg, `~/pyenvs/myenv/` or `\Users\Foo\Bar\PyEnvs\MyEnv\`
5. In the virtual environment folder choose `<your-venv-name>/bin/python` or `<your-venv-name>/bin/python3`
