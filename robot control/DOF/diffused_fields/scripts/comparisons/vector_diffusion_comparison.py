"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Compare DOF to independent vector diffusion and orthonormalization.
"""

import time

import numpy as np
from tqdm import tqdm

from diffused_fields.diffusion import PointcloudScalarDiffusion, WalkOnSpheresDiffusion
from diffused_fields.manifold import *
from diffused_fields.manifold.manifold import extract_plane
from diffused_fields.visualization.plotting_ps import *

# Select the object
# ==========================================
filename = "spot.ply"

pcloud = Pointcloud(filename=filename)
scalar_diffusion = PointcloudScalarDiffusion(pcloud, diffusion_scalar=1000)
scalar_diffusion.get_local_bases()

# the object itself is the boundary condition for the diffusion
# on the ambient space (robot's workspace)
boundaries = [pcloud]

# Monte Carlo diffusion solver for the ambient space
wos_diffusion = WalkOnSpheresDiffusion(
    boundaries=boundaries,
    convergence_threshold=pcloud.get_mean_edge_length() * 2,
)

# We will compute the diffused field at the grid points for visualizing the result
grid = pcloud.get_bounding_box_grid(bounding_box_scalar=1, nb_points=21)

# Get the center coordinates of the grid
grid.get_center()

# Extract cross-sections at mid-points using grid.center
grid.vertices = extract_plane(grid.vertices, axis="x", value=grid.center[0])

# Initialize arrays for diffused vectors
grid.tangent_x_vectors = np.zeros((len(grid.vertices), 3))
grid.tangent_y_vectors = np.zeros((len(grid.vertices), 3))
grid.normal_vectors = np.zeros((len(grid.vertices), 3))

# Initialize arrays for normalized vectors
grid.tangent_x_normalized = np.zeros((len(grid.vertices), 3))
grid.tangent_y_normalized = np.zeros((len(grid.vertices), 3))
grid.normal_normalized = np.zeros((len(grid.vertices), 3))

# Initialize arrays for orthonormalized local bases
grid.local_bases = np.zeros((len(grid.vertices), 3, 3))
grid.local_bases_svd = np.zeros((len(grid.vertices), 3, 3))

# Initialize arrays for orientation diffusion results
grid.orientation_diffusion = np.zeros((len(grid.vertices), 3, 3))

# Diffuse each vector type independently
print("Diffusing vectors...")
start_time = time.time()

for vertex_idx in tqdm(
    range(len(grid.vertices)), desc="Computing diffused vectors", unit="vertex"
):
    # Get the batch of points for parallel computation of the diffusion
    batch_points = wos_diffusion.get_batch_from_point(grid.vertices[vertex_idx])

    # Diffuse x-tangent vector
    grid.tangent_x_vectors[vertex_idx], _, _ = wos_diffusion.diffuse_vectors(
        batch_points, vector_type="tangent_x"
    )

    # Diffuse y-tangent vector
    grid.tangent_y_vectors[vertex_idx], _, _ = wos_diffusion.diffuse_vectors(
        batch_points, vector_type="tangent_y"
    )

    # Diffuse normal vector
    grid.normal_vectors[vertex_idx], _, _ = wos_diffusion.diffuse_vectors(
        batch_points, vector_type="normal"
    )

    # Normalize the vectors
    grid.tangent_x_normalized[vertex_idx] = grid.tangent_x_vectors[
        vertex_idx
    ] / np.linalg.norm(grid.tangent_x_vectors[vertex_idx])
    grid.tangent_y_normalized[vertex_idx] = grid.tangent_y_vectors[
        vertex_idx
    ] / np.linalg.norm(grid.tangent_y_vectors[vertex_idx])
    grid.normal_normalized[vertex_idx] = grid.normal_vectors[
        vertex_idx
    ] / np.linalg.norm(grid.normal_vectors[vertex_idx])

    # Method 1: Orthonormalization keeping z-axis fixed
    # Start with the normalized vectors
    x_vector = grid.tangent_x_normalized[vertex_idx].copy()
    z_vector = grid.normal_normalized[vertex_idx].copy()

    # Compute y-axis as cross product to ensure orthogonality
    y_vector = np.cross(z_vector, x_vector)
    y_vector = y_vector / np.linalg.norm(y_vector)  # normalize

    # Recompute x-axis to ensure perfect orthogonality
    x_vector = np.cross(y_vector, z_vector)
    x_vector = x_vector / np.linalg.norm(x_vector)  # normalize

    # Store the orthonormal basis
    grid.local_bases[vertex_idx, :, 0] = x_vector  # x-axis
    grid.local_bases[vertex_idx, :, 1] = y_vector  # y-axis
    grid.local_bases[vertex_idx, :, 2] = z_vector  # z-axis

    # Method 2: SVD-based orthonormalization without specifying which vector to keep
    # Stack the three normalized vectors as columns of a matrix
    A = np.column_stack(
        [
            grid.tangent_x_normalized[vertex_idx],
            grid.tangent_y_normalized[vertex_idx],
            grid.normal_normalized[vertex_idx],
        ]
    )

    # Compute SVD: A = U @ S @ Vt
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # The orthonormal basis is given by U @ Vt (closest orthogonal matrix to A)
    R_svd = U @ Vt

    # Store the SVD-based orthonormal basis
    grid.local_bases_svd[vertex_idx, :, 0] = R_svd[:, 0]  # x-axis
    grid.local_bases_svd[vertex_idx, :, 1] = R_svd[:, 1]  # y-axis
    grid.local_bases_svd[vertex_idx, :, 2] = R_svd[:, 2]  # z-axis

    # Compute orientation diffusion for comparison
    grid.orientation_diffusion[vertex_idx], _, _ = wos_diffusion.diffuse_rotations(
        batch_points
    )

print(f"Process finished --- {time.time() - start_time} seconds ---")

# Visualize results
ps.init()
set_camera_and_plane()
plot_sources(pcloud.vertices[scalar_diffusion.source_vertices])
# Plot the original pointcloud with its local bases
ps_pcloud = plot_orientation_field(
    pcloud.vertices, pcloud.local_bases, name="pcloud", enable_x=False
)

# Create local bases arrays for raw and normalized vectors for use with plot_orientation_field
raw_bases = np.zeros((len(grid.vertices), 3, 3))
raw_bases[:, :, 0] = (
    grid.tangent_x_vectors
    / np.linalg.norm(grid.tangent_x_vectors, axis=1)[:, np.newaxis]
)  # x-axis
raw_bases[:, :, 1] = (
    grid.tangent_y_vectors
    / np.linalg.norm(grid.tangent_y_vectors, axis=1)[:, np.newaxis]
)  # y-axis
raw_bases[:, :, 2] = (
    grid.normal_vectors / np.linalg.norm(grid.normal_vectors, axis=1)[:, np.newaxis]
)  # z-axis

norm_bases = np.zeros((len(grid.vertices), 3, 3))
norm_bases[:, :, 0] = grid.tangent_x_normalized  # x-axis
norm_bases[:, :, 1] = grid.tangent_y_normalized  # y-axis
norm_bases[:, :, 2] = grid.normal_normalized  # z-axis


# 1. Orthonormalized local bases (z-axis fixed)
z_fixed_grid = ps.register_point_cloud(
    points=grid.vertices, name="z_fixed", radius=1e-8
)
z_fixed_grid.add_vector_quantity(
    "local_x", grid.local_bases[:, :, 0], color=[0, 0, 1], enabled=True
)

# 2. SVD-based orthonormalized local bases
svd_grid = ps.register_point_cloud(points=grid.vertices, name="svd", radius=1e-8)
svd_grid.add_vector_quantity(
    "local_x", grid.local_bases_svd[:, :, 0], color=[1, 0, 0], enabled=True
)

# 3. Orientation diffusion (for comparison)
orientation_grid = ps.register_point_cloud(
    points=grid.vertices, name="orientation_diffusion", radius=1e-8
)
orientation_grid.add_vector_quantity(
    "local_x", grid.orientation_diffusion[:, :, 0], color=[0, 1, 0], enabled=True
)


# Uncomment for angular Deviation analysis
# ==============================================================================
# orientation_methods = {
#     "normalized_vectors": norm_bases,
#     "orthonormal_z_fixed": grid.local_bases,
#     "orthonormal_svd": grid.local_bases_svd,
#     "orientation_diffusion": grid.orientation_diffusion,
# }
# grid.visualize_angular_deviations(orientation_methods)
# ==============================================================================

ps.show()
