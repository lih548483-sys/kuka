"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Compare DOF to vector projection baseline.
"""

from diffused_fields.baselines import NearestFrameBaseline
from diffused_fields.diffusion import PointcloudScalarDiffusion
from diffused_fields.manifold import *
from diffused_fields.visualization.plotting_ps import *

# Select the object
filename = "spot.ply"

pcloud = Pointcloud(filename=filename)
scalar_diffusion = PointcloudScalarDiffusion(pcloud, diffusion_scalar=1000)
scalar_diffusion.get_local_bases()

# Initialize baseline method (original nearest frame, no WoS)
baseline = NearestFrameBaseline(pcloud)

# Compute orientation fields on pointcloud using both methods
orientations_baseline = baseline.compute_orientation_field(pcloud.vertices)
orientations_diffusion = pcloud.local_bases

ps.init()
# Analyze and visualize angular deviations between methods
orientation_methods = {
    "vector_projection": orientations_baseline,
    "scalar_diffusion": orientations_diffusion,
}
pcloud.visualize_angular_deviations(orientation_methods)

ps.show()
