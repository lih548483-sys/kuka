"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Test and visualize automatic endpoint detection using two-stage diffusion on batch deformed bananas.

This script demonstrates:
1. Generate 50 randomly deformed bananas (scaling + twist)
2. Detect endpoints automatically for each using two-stage diffusion
3. Visualize all results in a grid layout
"""

import numpy as np
import polyscope as ps

from diffused_fields.manifold import Pointcloud
from diffused_fields.utils import find_endpoints_via_diffusion


def generate_deformed_banana(filename="banana_half.ply", seed=None):
    """
    Generate a randomly deformed banana with scaling, twist, and bend.

    Args:
        filename: Path to banana PLY file
        seed: Random seed for reproducibility

    Returns:
        tuple: (pcloud, scale_factors, twist_strength, curvature, bend_axis)
    """
    if seed is not None:
        np.random.seed(seed)

    # Load fresh pointcloud
    pcloud = Pointcloud(filename=filename)

    # Apply random scaling with more drastic variations
    scale_factors = np.random.uniform(0.5, 1.5, size=3)
    pcloud.apply_scaling(scale_factors.tolist())

    # Apply random twist with more drastic angles
    twist_strength = np.random.uniform(-10.0, 10.0)
    pcloud.apply_twist(axis=2, twist_strength=twist_strength)

    # Apply random bend deformation
    curvature = np.random.uniform(-0.05, 0.05)  # Curvature parameter
    bend_axis = np.random.choice([0, 1, 2])  # Random axis to bend along
    pcloud.apply_bend(bend_axis=bend_axis, curvature=curvature)

    # Fix normal orientation and rebuild spatial structures after deformations
    pcloud.get_normals()
    pcloud.get_kd_tree()

    return pcloud, scale_factors, twist_strength, curvature, bend_axis


def visualize_batch_endpoints(results, experiments_per_row=10):
    """
    Visualize all deformed bananas with detected endpoints in a grid layout.

    Args:
        results: List of dicts with keys: 'pcloud', 'endpoint1', 'endpoint2',
                 'scale_factors', 'twist_strength', 'curvature', 'bend_axis'
        experiments_per_row: Number of experiments per row
    """
    print("\n✓ Visualizing all experiments in polyscope...")

    # Initialize polyscope
    ps.init()

    ps.set_up_dir("x_up")
    ps.set_front_dir("y_front")
    # Offset distances
    x_offset_distance = 0.15  # 15 cm offset between experiments in a row
    z_offset_distance = 0.25  # 15 cm offset between rows

    for i, result in enumerate(results):
        # Calculate row and column for this experiment
        row = i // experiments_per_row
        col = i % experiments_per_row

        # Calculate offset for this experiment
        offset = np.array([col * x_offset_distance, 0, row * z_offset_distance])

        # Add pointcloud with offset (black color)
        pcloud = result["pcloud"]
        vertices_offset = pcloud.vertices + offset
        ps.register_point_cloud(
            f"banana_{i}",
            vertices_offset,
            color=[0.0, 0.0, 0.0],  # Gray color
            enabled=True,
            radius=0.01,
        )

        # Add detected endpoints with offset
        endpoint1_idx = result["endpoint1"]
        endpoint2_idx = result["endpoint2"]

        endpoint1_pos = pcloud.vertices[endpoint1_idx] + offset
        endpoint2_pos = pcloud.vertices[endpoint2_idx] + offset

        ps.register_point_cloud(
            f"endpoint1_{i}",
            endpoint1_pos[np.newaxis, :],
            color=[0, 0, 1],  # Red for first endpoint
            radius=0.03,
            enabled=True,
        )
        ps.register_point_cloud(
            f"endpoint2_{i}",
            endpoint2_pos[np.newaxis, :],
            radius=0.03,
            color=[0, 0, 1],  # Red for first endpoint
            enabled=True,
        )

    num_rows = (len(results) + experiments_per_row - 1) // experiments_per_row

    axis_names = ["X", "Y", "Z"]
    for i, result in enumerate(results):
        row = i // experiments_per_row
        col = i % experiments_per_row
        scale = result["scale_factors"]
        twist = result["twist_strength"]
        curvature = result["curvature"]
        bend_axis = result["bend_axis"]
        scale_str = f"[{scale[0]:.2f},{scale[1]:.2f},{scale[2]:.2f}]"
        bend_str = f"{curvature:.3f}({axis_names[bend_axis]})"
        print(
            f"  Exp {i} (Row {row}, Col {col}): Scale:{scale_str}, Twist:{twist:.2f}°, Bend:{bend_str}"
        )

    print("\n✓ Showing polyscope (close window to exit)...")
    ps.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("BATCH AUTOMATIC ENDPOINT DETECTION TEST")
    print("=" * 60)

    filename = "banana_half.ply"
    num_experiments = 50

    print(f"\nGenerating {num_experiments} deformed bananas...")
    print(f"Object: {filename}")

    results = []

    for i in range(num_experiments):
        print(f"\nExperiment {i+1}/{num_experiments}")
        print("-" * 60)

        # Generate deformed banana
        pcloud, scale_factors, twist_strength, curvature, bend_axis = (
            generate_deformed_banana(
                filename=filename, seed=i  # Different seed for each experiment
            )
        )

        axis_names = ["X", "Y", "Z"]
        print(
            f"  Scale factors: [{scale_factors[0]:.2f}, {scale_factors[1]:.2f}, {scale_factors[2]:.2f}]"
        )
        print(f"  Twist strength: {twist_strength:.2f}°")
        print(f"  Bend curvature: {curvature:.3f} along {axis_names[bend_axis]}-axis")
        print(f"  Number of vertices: {len(pcloud.vertices)}")

        # Detect endpoints using two-stage diffusion
        print("  Detecting endpoints via two-stage diffusion...")
        endpoint1, endpoint2 = find_endpoints_via_diffusion(
            pcloud, min_distance_ratio=0.5
        )

        print(f"  ✓ Endpoint 1 (from center): vertex {endpoint1}")
        print(f"  ✓ Endpoint 2 (from endpoint 1): vertex {endpoint2}")

        # Store results
        results.append(
            {
                "pcloud": pcloud,
                "endpoint1": endpoint1,
                "endpoint2": endpoint2,
                "scale_factors": scale_factors,
                "twist_strength": twist_strength,
                "curvature": curvature,
                "bend_axis": bend_axis,
            }
        )

    print("\n" + "=" * 60)
    print(f"✓ Completed endpoint detection for {num_experiments} deformed bananas")
    print("=" * 60)

    # Visualize all results in a grid
    visualize_batch_endpoints(results, experiments_per_row=10)


if __name__ == "__main__":
    main()
