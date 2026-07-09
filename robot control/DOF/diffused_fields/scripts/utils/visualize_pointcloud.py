"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Visualize point cloud with original colors using Pointcloud class and polyscope.
"""

import numpy as np
import polyscope as ps

from diffused_fields.manifold import Pointcloud


def visualize_pointcloud_with_colors(filename):
    """
    Load a point cloud and visualize it with original colors using polyscope.

    Parameters:
    - filename (str): Name of the point cloud file (e.g., "spot.ply")
    """
    # Load the point cloud using the Pointcloud class (no voxel downsampling)
    print(f"Loading point cloud: {filename}")
    pcloud = Pointcloud(filename=filename, voxel_size=None)

    # Initialize polyscope
    ps.init()

    # Register the point cloud with polyscope
    ps_cloud = ps.register_point_cloud("pointcloud", pcloud.vertices)

    # Check if the point cloud has colors
    if pcloud.colors is not None and pcloud.colors.size > 0:
        # Check if colors are valid (not all zeros)
        if not np.all(pcloud.colors == 0):
            print(f"Point cloud has {len(pcloud.colors)} color values")
            print(
                f"Color range: [{pcloud.colors.min():.3f}, {pcloud.colors.max():.3f}]"
            )

            # Add colors as a vector quantity to the point cloud
            ps_cloud.add_color_quantity("original_colors", pcloud.colors, enabled=True)
        else:
            print("Point cloud has colors but they are all zero (black)")
            # Use a default gradient based on position
            heights = pcloud.vertices[:, 2]  # Z-coordinates
            ps_cloud.add_scalar_quantity("height", heights, enabled=True)
    else:
        print("Point cloud has no color information")
        # Use a default gradient based on position
        heights = pcloud.vertices[:, 2]  # Z-coordinates
        ps_cloud.add_scalar_quantity("height", heights, enabled=True)

    # Add additional information as scalar quantities
    if hasattr(pcloud, "normals") and pcloud.normals is not None:
        # Add normal magnitude as scalar quantity
        normal_magnitudes = np.linalg.norm(pcloud.normals, axis=1)
        ps_cloud.add_scalar_quantity(
            "normal_magnitude", normal_magnitudes, enabled=False
        )

        # Add normals as vector quantity
        ps_cloud.add_vector_quantity("normals", pcloud.normals, enabled=False)

    # Print point cloud information
    print(f"\nPoint cloud information:")
    print(f"  Vertices: {len(pcloud.vertices)}")
    print(f"  Original vertices: {pcloud.num_vertices_original}")
    if hasattr(pcloud, "voxel_size") and pcloud.voxel_size:
        print(f"  Voxel size: {pcloud.voxel_size}")
    print(
        f"  Bounding box: [{pcloud.vertices.min(axis=0)}, {pcloud.vertices.max(axis=0)}]"
    )

    # Show the visualization
    print("\nLaunching polyscope visualization...")
    ps.show()


def main():
    """Main function to run the visualization."""

    filename = "pear.ply"

    try:
        visualize_pointcloud_with_colors(filename)
    except FileNotFoundError:
        print(f"File {filename} not found. Please check the file path.")
    except Exception as e:
        print(f"Error loading point cloud: {e}")


if __name__ == "__main__":
    main()
