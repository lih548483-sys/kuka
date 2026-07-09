"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Nearest frame baseline for orientation field computation.
Computes orientation field by projecting vectors to planes defined by surface normals.
"""

import numpy as np
import yaml
from scipy.spatial import KDTree


class NearestFrameBaseline:
    """
    Computes orientation fields using nearest frame baseline method.

    This method finds the closest point on the surface for each query point,
    then computes a local coordinate frame by projecting a reference vector
    onto the tangent plane defined by the surface normal.
    """

    def __init__(self, pointcloud, source_vertices=None, wos_diffusion=None):
        """
        Initialize the nearest frame baseline.

        Args:
            pointcloud: The pointcloud object containing vertices and normals
            source_vertices: List of source vertex indices. If None, will load from config.
            wos_diffusion: Optional WalkOnSpheresDiffusion instance for diffused normals
        """
        self.pointcloud = pointcloud
        self.source_vertices = source_vertices
        self.kdtree = None
        self.source_vector = None
        self.wos_diffusion = wos_diffusion

    def load_source_vertices(self):
        """Load source vertices from config file if not provided."""
        if self.source_vertices is not None:
            return

        config_filepath = self.pointcloud.config

        # Load the YAML file
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)

        # Retrieve the section for the object
        params = config.get(self.pointcloud.object_name, {})
        scalar_diffusion_params = params.get("scalar_diffusion", {})

        if "source_vertices" in scalar_diffusion_params:
            self.source_vertices = scalar_diffusion_params["source_vertices"]
        else:
            # Default fallback - use center vertex if available, otherwise random
            if hasattr(self.pointcloud, "center_vertex"):
                self.source_vertices = [self.pointcloud.center_vertex]
            else:
                # Use two random vertices as fallback
                n_vertices = len(self.pointcloud.vertices)
                self.source_vertices = [
                    np.random.randint(0, n_vertices),
                    np.random.randint(0, n_vertices),
                ]

    def set_sources(self):
        """Setup the source vertices and compute reference vector."""
        # Ensure normals are computed
        self.pointcloud.get_normals()

        # Load source vertices from config if not provided
        self.load_source_vertices()

        # Handle empty source vertices by using all boundary points
        if not self.source_vertices:  # This catches both None and []
            boundary_mask = self.pointcloud.get_boundary()
            boundary_indices = np.where(boundary_mask)[0]
            if len(boundary_indices) == 0:
                raise ValueError(
                    "No source vertices specified and no boundary points found"
                )
            self.source_vertices = boundary_indices.tolist()
            print(
                f"[BASELINE] Auto-detected {len(self.source_vertices)} boundary vertices as sources"
            )

        # Build KDTree for finding closest points
        self.kdtree = KDTree(self.pointcloud.vertices)

        # Compute source vector from first two source vertices
        if len(self.source_vertices) >= 2:
            self.source_points = self.pointcloud.vertices[self.source_vertices[:2]]
            # Compute initial source vector between the two points
            self.source_vector = self.source_points[1] - self.source_points[0]
        else:
            boundary_mask = self.pointcloud.get_boundary()
            boundary_indices = np.where(boundary_mask)[0]
            if len(boundary_indices) == 0:
                raise ValueError(
                    "No source vertices specified and no boundary points found"
                )
            self.source_vertices = boundary_indices.tolist()
            print(
                f"[BASELINE] Auto-detected {len(self.source_vertices)} boundary vertices as sources"
            )

    def compute_local_frame(self, query_point):
        """
        Compute local coordinate frame for a single query point.

        Args:
            query_point: 3D point to compute frame for

        Returns:
            3x3 local basis matrix [x_axis, y_axis, z_axis]
        """
        if self.kdtree is None:
            raise RuntimeError("Must call setup_diffusion() first")

        # Get current (deformed) source point positions from pointcloud
        source_point_0 = self.pointcloud.vertices[self.source_vertices[0]]
        source_point_1 = self.pointcloud.vertices[self.source_vertices[1]]

        # Option 1
        # self.source_vector = source_point_1 - source_point_0
        # Option 3: Interpolate between pushing and pulling based on distance
        pushing_vector = query_point - source_point_1
        pulling_vector = query_point - source_point_0
        distance1 = np.linalg.norm(pushing_vector)
        distance2 = np.linalg.norm(pulling_vector)
        total_distance = distance1 + distance2
        push_weight = distance2 / (
            total_distance + 1e-8
        )  # Closer to source1 -> more pulling
        pull_weight = distance1 / (
            total_distance + 1e-8
        )  # Closer to source2 -> more pushing
        self.source_vector = push_weight * (
            -pushing_vector / (distance1 + 1e-8)
        ) + pull_weight * (pulling_vector / (distance2 + 1e-8))

        # Find closest point on the surface
        _, closest_idx = self.kdtree.query(query_point)

        # Get normal at closest point (z-axis) - flip to point towards object
        z_axis = self.pointcloud.normals[closest_idx]

        # Project source vector onto the plane defined by the normal
        projected_vector = (
            self.source_vector - np.dot(self.source_vector, z_axis) * z_axis
        )

        # Normalize to get x-axis
        x_axis = projected_vector / (np.linalg.norm(projected_vector) + 1e-8)

        # Compute y-axis as cross product (right-handed system)
        y_axis = np.cross(z_axis, x_axis)

        # Return local basis
        return np.column_stack([x_axis, y_axis, z_axis])

    def compute_local_frame_all_sources(self, query_point):
        """
        Compute local coordinate frame using all source points as pushing vectors.

        This method is designed for cases where we have many boundary source points
        (like in lawnmower with empty source_vertices) and want to consider the
        aggregate effect of all sources rather than just the first two.

        Args:
            query_point: 3D point to compute frame for

        Returns:
            3x3 local basis matrix [x_axis, y_axis, z_axis]
        """
        if self.kdtree is None:
            raise RuntimeError("Must call set_sources() first")

        # Compute aggregate pushing vector from all source points
        all_source_points = self.pointcloud.vertices[self.source_vertices]

        # Calculate weighted pushing vectors based on distance
        pushing_vectors = query_point - all_source_points
        distances = np.linalg.norm(pushing_vectors, axis=1)

        # Weight by inverse distance (closer sources have more influence)
        weights = 1.0 / (distances + 1e-8)
        weights = weights / np.sum(weights)  # Normalize weights

        # Aggregate source vector as weighted sum of pushing vectors
        self.source_vector = np.sum(weights.reshape(-1, 1) * pushing_vectors, axis=0)

        # Find closest point on the surface
        _, closest_idx = self.kdtree.query(query_point)

        # Get normal at closest point (z-axis) - flip to point towards object
        z_axis = -self.pointcloud.normals[closest_idx]

        # Project source vector onto the plane defined by the normal
        projected_vector = (
            self.source_vector - np.dot(self.source_vector, z_axis) * z_axis
        )

        # Normalize to get x-axis
        x_axis = projected_vector / (np.linalg.norm(projected_vector) + 1e-8)

        # Compute y-axis as cross product (right-handed system)
        y_axis = np.cross(z_axis, x_axis)

        # Return local basis
        return np.column_stack([x_axis, y_axis, z_axis])

    def compute_local_frame_wos(self, query_point):
        """
        Compute local coordinate frame using walk-on-spheres diffused normals.

        Args:
            query_point: 3D point to compute frame for

        Returns:
            3x3 local basis matrix [x_axis, y_axis, z_axis]
        """
        if self.kdtree is None:
            raise RuntimeError("Must call set_sources() first")
        if self.wos_diffusion is None:
            raise RuntimeError(
                "WalkOnSpheresDiffusion instance required for this method"
            )

        # Get current (deformed) source point positions from pointcloud
        source_point_0 = self.pointcloud.vertices[self.source_vertices[0]]
        source_point_1 = self.pointcloud.vertices[self.source_vertices[1]]

        # Compute source vector using current keypoint locations
        pushing_vector = query_point - source_point_1
        pulling_vector = query_point - source_point_0
        distance1 = np.linalg.norm(pushing_vector)
        distance2 = np.linalg.norm(pulling_vector)
        total_distance = distance1 + distance2
        push_weight = distance2 / (total_distance + 1e-8)
        pull_weight = distance1 / (total_distance + 1e-8)
        self.source_vector = push_weight * (
            -pushing_vector / (distance1 + 1e-8)
        ) + pull_weight * (pulling_vector / (distance2 + 1e-8))

        # Use walk-on-spheres to get diffused normal instead of closest point normal
        batch_points = self.wos_diffusion.get_batch_from_point(query_point)
        z_axis, _, _ = self.wos_diffusion.diffuse_vectors(
            batch_points, vector_type="normal"
        )
        z_axis = z_axis / np.linalg.norm(z_axis)  # normalize

        # Project source vector onto the plane defined by the diffused normal
        projected_vector = (
            self.source_vector - np.dot(self.source_vector, z_axis) * z_axis
        )

        # Normalize to get x-axis
        x_axis = projected_vector / (np.linalg.norm(projected_vector) + 1e-8)

        # Compute y-axis as cross product (right-handed system)
        y_axis = np.cross(z_axis, x_axis)

        # Return local basis
        return np.column_stack([x_axis, y_axis, z_axis])

    def compute_orientation_field(self, query_points):
        """
        Compute orientation field for multiple query points.

        Args:
            query_points: Nx3 array of query points

        Returns:
            Nx3x3 array of local basis matrices
        """
        if self.kdtree is None:
            self.set_sources()

        orientations = []
        for point in query_points:
            local_basis = self.compute_local_frame(point)
            orientations.append(local_basis)

        return np.array(orientations)

    def compute_orientation_field_wos(self, query_points):
        """
        Compute orientation field using walk-on-spheres diffused normals.

        Args:
            query_points: Nx3 array of query points

        Returns:
            Nx3x3 array of local basis matrices
        """
        if self.kdtree is None:
            self.set_sources()

        orientations = []
        for point in query_points:
            local_basis = self.compute_local_frame_wos(point)
            orientations.append(local_basis)

        return np.array(orientations)

    def get_source_vertices(self):
        """Get the source vertices used for the baseline."""
        if self.source_vertices is None:
            self.load_source_vertices()
        return self.source_vertices

    def get_source_points(self):
        """Get the source points used for computing the reference vector."""
        if self.source_vertices is None:
            self.load_source_vertices()
        return self.pointcloud.vertices[self.source_vertices[:2]]
