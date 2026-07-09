"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

from setuptools import find_packages, setup


# Read version from __init__.py
def get_version():
    with open("src/diffused_fields/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip("\"'")
    return "0.0.1"


setup(
    name="diffused_fields",
    version=get_version(),
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        # Core dependencies
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "PyYAML>=6.0",
        "tqdm",
        # Point cloud processing
        "open3d>=0.13.0",
        "potpourri3d",
        "robust_laplacian",
        "pcdiff",
        # Visualization
        "polyscope",
        # Reinforcement Learning
        "stable-baselines3",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    description="A Python package for diffusion-based methods on geometric manifolds",
    long_description="""
    Diffused Fields provides diffusion algorithms on geometric manifolds including
    point clouds, meshes, and other geometric structures. It implements both
    traditional diffusion solvers and walk-on-spheres methods.
    
    This package provides tools for:
    - Point cloud and mesh diffusion algorithms
    - Geometric manifold operations
    - Walk-on-spheres diffusion methods
    - Scalar, vector, and quaternion diffusion
    - Visualization utilities
    
    Developed at the Robot Learning and Interaction group of the Idiap Research Institute.
    """,
    long_description_content_type="text/plain",
    author="Cem Bilaloglu",
    author_email="cem.bilaloglu@idiap.ch",
    url="",
    project_urls={
        "Paper": "http://arxiv.org/abs/2402.04862",
        "Source": "https://github.com/idiap/diffused_fields",  # Update when available
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.12",
    keywords="diffusion, point-cloud, mesh, manifolds, walk-on-spheres, geometric-algebra",
)
