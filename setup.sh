#!/usr/bin/env bash
# TriFlow environment setup. Usage: bash setup.sh
set -e

conda create -n triflow python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate triflow

git submodule update --init --recursive
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit -y
CUDA_HOME=$CONDA_PREFIX pip install --no-build-isolation git+https://github.com/mit-han-lab/torchsparse.git@385f5ce8718fcae93540511b7f5832f4e71fd835
pip install -r requirements.txt
pip install -e third_party/pyfqmr-triflow

echo "PYTHONPATH=\$PYTHONPATH:third_party/TRELLIS:third_party/Direct3D-S2" > .env
echo "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib" >> .env
