"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Compute the diffused rotation field on a pointcloud using random sources
"""

import numpy as np

from diffused_fields.diffusion.pointcloud_quaternion_diffusion import *
from diffused_fields.manifold import Pointcloud

# Select the object
# ==========================================
filename = "rectangular_grid.ply"


pcloud = Pointcloud(filename=filename)

quaternion_diffusion_solver = PointcloudQuaternionDiffusion(
    pcloud, diffusion_scalar=20  # increase for smoothness
)


# Uncomment for planar oriented keypoints
# ==============================================================================
source_vertices = np.array([223, 520, 721])  # far away vertices

z_angle = np.array([180, 270, 90])  #  far away rotations

quaternion_diffusion_solver.set_random_planar_sources(
    source_vertices=source_vertices, z_angle=z_angle
)
# ==============================================================================


# Uncomment for non-planar oriented keypoints
# ==============================================================================
# q1 = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x
# q2 = np.array([0.0, 0.0, 1.0, 0.0])  # 180 deg about y
# q3 = np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about z
# quats = [q1, q2, q3]
# quaternion_diffusion_solver.source_vertices = np.array([223, 520, 721])
# quaternion_diffusion_solver.source_pure_quaternions, _ = pure_quaternions_for_dirichlet(
#     quats
# )

# quaternion_diffusion_solver.source_pure_quaternions = np.array(
#     quaternion_diffusion_solver.source_pure_quaternions
# )
# ==============================================================================


quaternion_diffusion_solver.diffuse_quaternions()
quaternion_diffusion_solver.visualize_diffused_quaternions()
