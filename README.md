# CUDA optimisation

### Useful links

## Install & setup

> This project uses `python 3.10` version and `conda` (Anaconda) as python package manager on linux (Ubuntu 22.04 LTS).

[Install conda (Anaconda) on Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) - there is also a script for it in `config/`. Make sure NOT to run the script with `sudo`.

1. Install `conda` and `jupyter notebook`
2. Create conda env: `conda env create --file environment.yml`
3. Go in env with `conda activate py310omp`

## TODO

* [X] debug segfault in pi_multithread
