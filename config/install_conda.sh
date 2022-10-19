#!/bin/bash
# This file install Anaconda for Linux
# WARN: do NOT run with sudo, else it will install conda for root

# Go to home directory
cd ~

### Download Anaconda and install it
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
chmod +x Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh -b -p ~/anaconda
echo 'export PATH=~/anaconda/bin:$PATH' >> ~/.bashrc # modify PATH in 
# tell leandre thre are many of these lines inside

#ls -alt
#ls -alt ~/anaconda/bin/

### get conda in path
echo "Last line of .bashrc:"
tail -1 ~/.bashrc # cat last line of ~/.bashrc
# source not working: https://stackoverflow.com/questions/48785324/source-command-in-shell-script-not-working
. ~/.bashrc # seems not to work... conda not detected
echo $PATH

# put conda in current PATHw
export PATH=~/anaconda/bin:$PATH # this always works
echo $PATH

~/anaconda/bin/conda --version # works ok
conda env list 
conda update conda

# finish to setup conda
conda init
conda config --set auto_activate_base true

echo ">>> deleting install script..."
rm Anaconda3-2022.05-Linux-x86_64.sh