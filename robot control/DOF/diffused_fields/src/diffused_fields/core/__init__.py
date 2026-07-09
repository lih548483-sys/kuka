"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Core utilities and shared components for diffused_fields.

This module contains:
- Centralized configuration management
- Base classes and protocols
- Shared utilities
- Path handling
"""

from .config import Config

__all__ = ["Config"]
