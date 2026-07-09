"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Diffused Fields - A Python package for diffusion-based methods on manifolds.

This package implements diffusion algorithms on geometric manifolds including
point clouds, meshes, and other geometric structures. It provides both
traditional diffusion solvers and walk-on-spheres methods.

The package is organized into focused submodules:
- manifold: Geometric structures (Pointcloud, Mesh, Grid, Manifold)
- diffusion: Diffusion algorithms and solvers (scalar, vector, walk-on-spheres)
- visualization: Plotting and visualization utilities

Copyright (c) 2024 Idiap Research Institute
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

# Version information
__version__ = "0.1.0"
__author__ = "Cem Bilaloglu"
__email__ = "cem.bilaloglu@idiap.ch"
__license__ = "MIT"

# Import from organized submodules with error handling
_CORE_IMPORTS = []
_MANIFOLD_IMPORTS = []
_DIFFUSION_IMPORTS = []
_VISUALIZATION_IMPORTS = []
_UTILS_IMPORTS = []

# Core submodule - configuration and utilities
try:
    from .core import Config
    _CORE_IMPORTS = ["Config"]
except ImportError as e:
    print(f"Warning: Core submodule import failed: {e}")

# Manifold submodule - geometric structures
try:
    from .manifold import (
        Manifold, Plane, Line, Sphere, Pointcloud, Mesh, Grid
    )
    _MANIFOLD_IMPORTS = [
        "Manifold", "Plane", "Line", "Sphere", "Pointcloud", "Mesh", "Grid"
    ]
except ImportError as e:
    print(f"Warning: Manifold submodule import failed: {e}")

# Diffusion submodule - diffusion algorithms and solvers
try:
    from .diffusion import (
        DiffusionSolver, PointcloudScalarDiffusion,
        WalkOnSpheresDiffusion
    )
    _DIFFUSION_IMPORTS = [
        "DiffusionSolver", "PointcloudScalarDiffusion",
        "WalkOnSpheresDiffusion"
    ]

except ImportError as e:
    print(f"Warning: Diffusion submodule import failed: {e}")

# Visualization submodule - plotting utilities
try:
    from .visualization import *
    _VISUALIZATION_IMPORTS = ["visualization_utils"]  # Placeholder for * imports
except ImportError as e:
    print(f"Note: Visualization submodule not available (likely missing dependencies): {e}")

# Utils submodule - utility functions
try:
    from .utils import (
        find_endpoints_via_diffusion,
        find_endpoints_via_extremal_projection,
        visualize_detected_endpoints
    )
    _UTILS_IMPORTS = [
        "find_endpoints_via_diffusion",
        "find_endpoints_via_extremal_projection",
        "visualize_detected_endpoints"
    ]
except ImportError as e:
    print(f"Note: Utils submodule not available: {e}")

# Combine all successful imports
__all__ = _CORE_IMPORTS + _MANIFOLD_IMPORTS + _DIFFUSION_IMPORTS + _VISUALIZATION_IMPORTS + _UTILS_IMPORTS

# Provide easy access to submodules
from . import core
from . import manifold
from . import diffusion

# Optional submodules (may fail)
try:
    from . import visualization
except ImportError:
    pass

try:
    from . import utils
except ImportError:
    pass

# No automatic print statements on import - keep the console clean