"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Compare DOF to nearest frame baseline.
"""

from diffused_fields.baselines import NearestFrameBaseline
from diffused_fields.diffusion import PointcloudScalarDiffusion, WalkOnSpheresDiffusion
from diffused_fields.manifold import *
from diffused_fields.manifold.manifold import extract_plane
from diffused_fields.visualization.plotting_ps import *

# Select the object
filename = "spot.ply"

pcloud = Pointcloud(filename=filename)
scalar_diffusion = PointcloudScalarDiffusion(pcloud, diffusion_scalar=1000)
scalar_diffusion.get_local_bases()

# Set up walk-on-spheres diffusion
boundaries = [pcloud]
wos_diffusion = WalkOnSpheresDiffusion(
    boundaries=boundaries,
    convergence_threshold=pcloud.get_mean_edge_length() * 2,
)

# Create grid for comparison
grid = pcloud.get_bounding_box_grid(bounding_box_scalar=1, nb_points=21)
grid.get_center()
grid.vertices = extract_plane(grid.vertices, axis="x", value=grid.center[0])


# Initialize baseline methods
baseline_original = NearestFrameBaseline(pcloud)

# Compute orientation fields using both methods
orientations_original = baseline_original.compute_orientation_field(grid.vertices)
wos_diffusion.diffuse_orientations_on_grid(grid)
orientations_diffusion = grid.local_bases


ps.init()
set_camera_and_plane()
ps_pcloud = plot_orientation_field(
    pcloud.vertices, pcloud.local_bases, name="pcloud", enable_x=False
)
plot_sources(pcloud.vertices[scalar_diffusion.source_vertices])


# Uncomment for visualizing the local frames you can activate/deactivate in GUI
# ==============================================================================
plot_orientation_field(
    grid.vertices,
    orientations_original,
    name="nearest_frame_baseline",
    enable_z=True,
    point_radius=0,
)

plot_orientation_field(
    grid.vertices,
    orientations_diffusion,
    name="DOF",
    enable_z=True,
    point_radius=0,
    enable=False,
)

# ==============================================================================

#  Uncomment for analyzing and visualizing angular deviations between methods
# ==============================================================================
# orientation_methods = {
#     "nearest_frame_original": orientations_original,
#     "wos_orientation_diffusion": orientations_diffusion,
# }
# grid.visualize_angular_deviations(orientation_methods)
# ==============================================================================
ps.show()
