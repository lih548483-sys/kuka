"""
Copyright (c) 2024 Idiap Research Institute
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Centralized configuration management for diffused_fields.

This module provides a single source of truth for:
- Project paths (data, config, results)
- Configuration file loading
- Path resolution utilities
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Centralized configuration management for diffused_fields."""

    _project_root: Optional[Path] = None

    @classmethod
    def get_project_root(cls) -> Path:
        """
        Get the project root directory.

        This method calculates the project root once and caches it.
        The project root is determined by going up from this file's location
        until we find the directory containing 'src/diffused_fields'.

        Returns:
            Path: The project root directory
        """
        if cls._project_root is None:
            # This file is in: src/diffused_fields/core/config.py
            # So project root is 3 levels up: ../../../
            cls._project_root = Path(__file__).resolve().parents[3]

        return cls._project_root

    @classmethod
    def get_data_dir(cls) -> Path:
        """Get the data directory path."""
        return cls.get_project_root() / "data"

    @classmethod
    def get_pointclouds_dir(cls) -> Path:
        """Get the pointclouds data directory path."""
        return cls.get_data_dir() / "pointclouds"

    @classmethod
    def get_meshes_dir(cls) -> Path:
        """Get the meshes data directory path."""
        return cls.get_data_dir() / "meshes"

    @classmethod
    def get_results_dir(cls) -> Path:
        """Get the results directory path."""
        return cls.get_data_dir() / "results"

    @classmethod
    def get_policy_dir(cls) -> Path:
        """Get the policy directory path."""
        return cls.get_data_dir() / "policy"

    @classmethod
    def get_config_dir(cls) -> Path:
        """Get the config directory path."""
        return cls.get_project_root() / "config"

    @classmethod
    def get_pointclouds_config_path(cls) -> Path:
        """Get the pointclouds configuration file path."""
        return cls.get_config_dir() / "pointclouds.yaml"

    @classmethod
    def get_meshes_config_path(cls) -> Path:
        """Get the meshes configuration file path."""
        return cls.get_config_dir() / "meshes.yaml"

    @classmethod
    def load_pointcloud_config(cls, object_name: str) -> Dict[str, Any]:
        """
        Load configuration for a specific pointcloud object.

        Args:
            object_name: Name of the pointcloud object to load config for

        Returns:
            Dict containing the configuration parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            KeyError: If object_name not found in config
        """
        config_path = cls.get_pointclouds_config_path()

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        if object_name not in config:
            raise KeyError(f"Object '{object_name}' not found in pointclouds config")

        return config[object_name]

    @classmethod
    def load_mesh_config(cls, object_name: str) -> Dict[str, Any]:
        """
        Load configuration for a specific mesh object.

        Args:
            object_name: Name of the mesh object to load config for

        Returns:
            Dict containing the configuration parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            KeyError: If object_name not found in config
        """
        config_path = cls.get_meshes_config_path()

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        if object_name not in config:
            raise KeyError(f"Object '{object_name}' not found in meshes config")

        return config[object_name]

    @classmethod
    def resolve_pointcloud_path(cls, filename: str) -> Path:
        """
        Resolve a pointcloud filename to its full path.

        Args:
            filename: Name of the pointcloud file

        Returns:
            Path: Full path to the pointcloud file
        """
        return cls.get_pointclouds_dir() / filename

    @classmethod
    def resolve_mesh_path(cls, filename: str) -> Path:
        """
        Resolve a mesh filename to its full path.

        Args:
            filename: Name of the mesh file

        Returns:
            Path: Full path to the mesh file
        """
        return cls.get_meshes_dir() / filename

    @classmethod
    def ensure_directories_exist(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.get_data_dir(),
            cls.get_pointclouds_dir(),
            cls.get_meshes_dir(),
            cls.get_results_dir(),
            cls.get_config_dir(),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_info(cls) -> Dict[str, str]:
        """
        Get configuration information for debugging.

        Returns:
            Dict with current path configurations
        """
        return {
            "project_root": str(cls.get_project_root()),
            "data_dir": str(cls.get_data_dir()),
            "pointclouds_dir": str(cls.get_pointclouds_dir()),
            "meshes_dir": str(cls.get_meshes_dir()),
            "config_dir": str(cls.get_config_dir()),
            "pointclouds_config": str(cls.get_pointclouds_config_path()),
            "meshes_config": str(cls.get_meshes_config_path()),
        }
