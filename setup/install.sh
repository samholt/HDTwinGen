#!/bin/bash
conda create --name reg python=3.9.7
conda activate reg
pip3 install torch torchvision torchaudio
pip install --upgrade pip
pip install hydra-core --upgrade
# https://github.com/google/jax#installation
pip install -r requirements.txt
pip install hydra-core --upgrade
pip install evosax
pip install gymnax
mkdir repos
cd repos
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
# Could install gymnax again
# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
sudo apt-get install xvfb
sudo apt-get install python-opengl