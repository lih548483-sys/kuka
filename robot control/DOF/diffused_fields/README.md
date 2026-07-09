# Diffused Fields

This package is supplementary material for the paper **"Object-centric Task Representation and Transfer using Diffused Orientation Fields"**.

This is the **core package** that provides fundamental diffusion algorithms and geometric manifold operations for computing **Diffused Orienation Fields (DOF)**.

For robot manipulation experiments using **DOF**, see the companion package:
- **[diffused_fields_robotics](https://github.com/idiap/diffused_fields_robotics)** (depends on this package) - Object-centric robot manipulation applications: local action primitives (peeling, slicing and tactile coverage), trajectory optimization, and reinforcement learning using DOF.

## Features

**Core package for solving orientation valued diffusion PDE (i.e., heat equation) on geometric manifolds.**

This package provides implementations of both traditional diffusion solvers and walk-on-spheres methods on point clouds for various data types (scalars, vectors, quaternions).

- **Diffusion on Manifolds**: Scalar, vector, and quaternion diffusion on point clouds 
- **Walk-on-Spheres Methods**: Monte Carlo-based diffusion solvers for point clouds and various geometric primitives
- **Visualization Tools**: Interactive 3D visualization with Polyscope
- **Comparisons to Baselines**: Comparison to nearest frame projection, tangent vector projection and Euclidean diffusion baselines

## Installation

### Complete Installation (Recommended) 

Follow the installation instructions of the **[diffused_fields_robotics](https://github.com/idiap/diffused_fields_robotics)**


### Stand-alone Installation Using Python 3.12 virtual environment

This repository uses [Git LFS](https://git-lfs.github.com/) to store large files
(e.g. data, models, point clouds).


Install Git LFS (Ubuntu)
```bash
sudo apt install git-lfs
```
Install Git LFS (macOS) using homebrew
```bash
brew install git-lfs
```
run once to enable LFS
```bash
git lfs install 
```

If you cloned the repository before installing and enabling git-lfs pull the large files
```bash
git lfs pull
```

Clone this repository
```bash
git clone https://github.com/idiap/diffused_fields.git
```
Create a virtual environment and install the package in editable mode:
```bash
# go to cloned directory
cd diffused_fields
# Create a virtual environment named 'df' with Python 3.12
python3.12 -m venv df

# Activate the virtual environment
source df/bin/activate

# Install the package in editable mode
pip install -e .
```

## Repository Structure

```
diffused_fields/
├── src/diffused_fields/     # Main package source code
│   ├── core/                # Core algorithms and data structures
│   ├── diffusion/           # Diffusion solvers (scalar, vector, quaternion)
│   ├── manifold/            # Manifold operations and geometry
│   ├── utils/               # Utility functions
│   ├── visualization/       # Visualization tools
│   └── baselines/           # Baseline methods for comparison
├── scripts/                 # Example scripts and demonstrations
├── data/                    # Sample point clouds and meshes
│   ├── pointclouds/         # .ply and .pcd files
│   └── meshes/              # .obj and .stl files
└── config/                  # Configuration files
```

## Paper and Citation

If you use this package in your research, please cite:

```bibtex
@online{bilalogluTactileErgodicControl2024,
  title = {Tactile {{Ergodic Control Using Diffusion}} and {{Geometric Algebra}}},
  author = {Bilaloglu, Cem and Löw, Tobias and Calinon, Sylvain},
  date = {2024-02-07},
  eprint = {2402.04862},
  eprinttype = {arxiv},
  eprintclass = {cs},
  url = {http://arxiv.org/abs/2402.04862}
}
```


## Reproducing Paper Results

Simulation data and plots from the paper can be generated using the scripts in this repository. 

## Dependencies

To compute the discrete Laplacian on point clouds (and also meshes if you want)
robust_laplacian: https://github.com/nmwsharp/robust-laplacians-py

For basic point cloud operations (another library can be easily used instead):
open3d: https://www.open3d.org

For visualizations:
plotly: https://polyscope.run/py/

For sparse matrix operations:
scipy: https://scipy.org

For linear algebra operations:
numpy: https://numpy.org


This code is maintained by Cem Bilaloglu and licensed under the MIT License.

Copyright (c) 2025 Idiap Research Institute contact@idiap.ch