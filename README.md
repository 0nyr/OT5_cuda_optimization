# CUDA optimisation

### Useful links

## Install & setup

> This project uses `python 3.10` version and `conda` (Anaconda) as python package manager on linux (Ubuntu 22.04 LTS).

[Install conda (Anaconda) on Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) - there is also a script for it in `config/`. Make sure NOT to run the script with `sudo`.

1. Install `conda` and `jupyter notebook`
2. Create conda env: `conda env create --file environment.yml`
3. Go in env with `conda activate py310omp`

## Install cuda toolkit

> Ubuntu 22.04 LTS

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install nvidia-cuda-toolkit
sudo apt install libtbb-doc libvdpau-doc opencl-clhpp-headers-doc
snap install nvtop
```

If following error: `system has unsupported display driver / cuda driver combination` or nvtop displays `No GPU to monitor`. This is due to Ubuntu having different drivers.

```
sudo ubuntu-drivers install 
```

Doing so will remove the softlink to nvcc. Check that nvcc is still installed with `apt list --installed | grep nv`. Then determine the location of the nvcc executable with `dpkg -L cuda-nvcc-11-8 | grep nvcc`. Then create a softlink inside `/usr/bin` with `ln -s /usr/local/cuda-11.8/bin/nvcc nvcc`. Now you should have a working nvcc. 

### sh:1 `<stuff>`: not found

> If you can't compile with nvcc, it's because some executables are not in the PATH.

Easy fix, to make cuda tools available: edit `~/.bashrc` with following lines. If you can't find something, go to `/usr/local` and use `find . -name <stuff_to_find>`:

```
### CUDA for Nvidia card
export "PATH=$PATH:/usr/local/cuda/bin:/usr/local/cuda-11.8/bin:/usr/local/cuda/nvvm/bin"
```

## TODO

* [X] debug segfault in pi_multithread
