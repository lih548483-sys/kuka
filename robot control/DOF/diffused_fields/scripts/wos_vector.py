"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Diffuses normal vectors of a point cloud into the ambient space using
Walk on Spheres (WOS) method.
"""

from diffused_fields.diffusion import WalkOnSpheresDiffusion
from diffused_fields.manifold import *
from diffused_fields.manifold.manifold import extract_plane
from diffused_fields.visualization.plotting_ps import *

# Select the object
# ==========================================
filename = "spot.ply"

pcloud = Pointcloud(filename=filename)
pcloud.get_normals()

# the object itself is the boundary condition for the diffusion
# on the ambient space (robot's workspace)
boundaries = [pcloud]

# Monte Carlo diffusion solver for the ambient space
wos_diffusion = WalkOnSpheresDiffusion(
    boundaries=boundaries,
    convergence_threshold=pcloud.get_mean_edge_length() * 2,
)

# We will compute the diffused field at the grid points for visualizing the result
grid = pcloud.get_bounding_box_grid(bounding_box_scalar=1, nb_points=11)

# Extract cross-sections at mid-points using grid.center
# comment if you want the full grid
grid.get_center()
grid.vertices = extract_plane(grid.vertices, axis="x", value=grid.center[0])

# Compute diffused vectors on the grid using WOS
wos_diffusion.diffuse_vectors_on_grid(grid)

ps.init()
set_camera_and_plane()
ps_pcloud = ps.register_point_cloud("pcloud", pcloud.vertices, color=[0, 0, 0])
ps_pcloud.add_vector_quantity("vectors", pcloud.normals, color=[1, 0, 1])

ps_grid = ps.register_point_cloud("grid", grid.vertices, color=[0, 0, 0])
ps_grid.add_vector_quantity(
    "diffused_vectors", grid.vectors, color=[0, 0, 1], enabled=True
)

ps.show()
