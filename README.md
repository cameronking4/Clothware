# Digital-clothing
This repository contains a code for obtaining a 3D reconstuction of a garment. This is a "proof-of-concept" version of a potential solution for a garment reconstruction task by monocular images. Currently the code supports 2 types of garment ('shirt', 'pants'). 

For reconstruction you need to have an image of clothing lying on a flat surface. There are 2 variants of usage:

*  using front side image of a garment;
*  using both: front and back side images.

The reconstruction lies in transfering texture from the image to the corresponding 3D template-mesh of the garment type. The garment type needs to be specified with the input. 

# Installation
## Prerequisites
*  OS: Ubuntu 20.04.
*  [conda](https://docs.conda.io/en/latest/miniconda.html)
*  [cuda-toolkit](https://docs.nvidia.com/cuda/index.html#installation-guides)

## Setup
``` shell
conda create -n grecon python=3.7.9
conda activate grecon

# For CUDA 10
# conda install -c pytorch -c conda-forge pytorch=1.10.1 torchvision=0.11.2 cudatoolkit=10.2
# For CUDA 11
conda install -c pytorch -c conda-forge pytorch=1.10.1 torchvision=0.11.2 cudatoolkit=11.1

pip install -r requirements.txt
mkdir input
```

## KDGet installation
``` shell
cd kgdet/deepfashion2_api/PythonAPI
pip install -e .
cd ../../mmdetection
python setup.py develop
cd ../../
```

# Checkpoints and Meshes
Download [checkpoints](https://drive.google.com/file/d/1aBLAkpBHRL39x1ecdlS5rvfMJmCVUGkJ/view?usp=share_link) and [meshes](https://drive.google.com/file/d/18Ln_YN3RAaK9ZmifGhD1mzPhOpGIxQ6K/view?usp=share_link) and unzip them at the root folder.

# Input data
Place a garment images at `./input` folder. You can also use another folder for that purpose, but you will need to specify it for `demo.py` script via `--root` argument.

You can dowload demo-data [here](https://drive.google.com/file/d/11-NJZNpYJv5sWhZxGGNfgS925LqYePQU/view?usp=share_link) and extract it at the root folder.
# Usage
Any of scripts needs to invoked from the root directory. For now there are available 2 possible classes for the reconstruction: `shirt` and `pants`.

Note, that a folder for output needs to be empty, otherwise its content will be removed automatically.

After invoking `demo.py` script check results at the folder specified via `--out` argument.
``` shell
usage: demo.py [-h] [--root ROOT] [--out OUT] [--mesh MESH]
               [--checkpoints CHECKPOINTS] --type {pants,shirt} --front FRONT
               [--back BACK] [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           Directory, where images are placed.
  --out OUT             Directory for output results.
  --mesh MESH           Directory of a folder containing pattern-meshes.
  --checkpoints CHECKPOINTS
                        Directory of a folder containing checkpoints for
                        kgdet.
  --type {pants,shirt}  Target type of garment for reconstruction.
  --front FRONT         Path to image of a front side of a garment.
  --back BACK           Path to image of a front side of a garment.
```
## Example
```
python demo.py --root ./input --type shirt --front shirt6.jpg --back shirt6_b.jpg 
python demo.py --root ./input/pants1 --type pants --front 1.png
python demo.py --root ./input/pants3 --type pants --front 3_f.png --back 3_b.png 
```
