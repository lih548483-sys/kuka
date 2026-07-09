"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Visualization utilities for diffused fields.

This submodule contains all plotting and visualization functionality:
- Polyscope integration for 3D visualization
- Matplotlib helpers
- Animation and trajectory visualization

Dependencies: diffused_fields.core, polyscope, matplotlib
"""

# Try to import visualization modules (they may have missing dependencies)
_AVAILABLE_MODULES = []

try:
    from .plotting_ps import *  # Import all plotting functions
    _AVAILABLE_MODULES.append('plotting_ps')
except ImportError:
    pass

try:
    from .plotting_utils import *  # Import all utility functions
    _AVAILABLE_MODULES.append('plotting_utils')
except ImportError:
    pass

__all__ = _AVAILABLE_MODULES