"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from ..core.config import Config
from .grid import Grid
from .manifold import Manifold, Sphere


# child class
class Mesh(Manifold):
    def __init__(
        self,
        vertices=None,
        faces=None,
        filename=None,
        voxel_size=None,
        scale=None,
        translation=None,
        rotation=None,
        normal_orientation=None,
        file_directory=None,
        center_position=None,
        center_vertex=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            type=type(self), scale=scale, translation=translation, rotation=rotation
        )
        self.config = self.config_dir / "meshes.yaml"

        self.file_directory = file_directory
        self.voxel_size = voxel_size
        self.normal_orientation = normal_orientation

        if self.file_directory is None:
            # Use centralized config management for mesh directory
            self.file_directory = str(Config.get_meshes_dir()) + "/"

        # Construct the point cloud
        # ==============================================================================
        object_name = "default"
        if (
            vertices is not None and faces is not None
        ):  # initialize from vertices and faces
            self.mesh = o3d.geometry.TriangleMesh()
            self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self.mesh.triangles = o3d.utility.Vector3iVector(faces)
        elif filename is not None:  # initialize from file
            object_name = filename.split(".")[0]
            filepath = self.file_directory + filename
            # print(f"Reading mesh from {filepath}")
            self.mesh = o3d.io.read_triangle_mesh(filepath)
            # Extract vertices and faces as NumPy arrays
            vertices = np.asarray(self.mesh.vertices)  # Nx3 numpy array
            faces = np.asarray(self.mesh.triangles)  # Mx3 numpy array

        self.object_name = object_name
        self.load_object_parameters(object_name)

        # Transform the point cloud
        # ==============================================================================
        self.mesh.scale(self.scale, center=self.mesh.get_center())
        self.mesh.rotate(self.rotation.as_matrix(), center=self.mesh.get_center())
        self.mesh.translate(self.translation, relative=False)

        self.vertices = np.asarray(self.mesh.vertices)
        self.faces = np.asarray(self.mesh.triangles)

        if center_position is None:
            if center_vertex is not None:
                self.center_position = self.vertices[center_vertex]
            else:
                self.center_position = self.mesh.get_center()
        else:
            self.center_position = center_position

        self.center_offset = self.mesh.get_center() - self.center_position

        self.colors = np.asarray(self.mesh.vertex_colors)
        if self.colors.size == 0:  # pointcloud have no color
            self.colors = np.zeros_like(self.vertices)

    def translate_center(self, position):
        target_pos = position + self.center_offset
        self.mesh.translate(target_pos, relative=False)
        self.vertices = np.asarray(self.mesh.vertices)
        self.faces = np.asarray(self.mesh.triangles)

    def get_kd_tree(self):
        # Construct the KD-tree and adjacency graph
        # ==============================================================================
        self.kd_tree = cKDTree(self.vertices)
        # self.knn_graph = knn_graph(self.vertices, self.num_neighbors)
        return self.kd_tree

    def get_boundary_kd_tree(self):
        # Construct the KD-tree and adjacency graph
        # ==============================================================================
        self.boundary_kd_tree = cKDTree(self.vertices[self.is_boundary_arr])
        # self.knn_graph = knn_graph(self.vertices, self.num_neighbors)
        return self.boundary_kd_tree

    def load_object_parameters(self, object_name):
        config_filepath = self.config
        # Load the YAML file
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)

        # Retrieve the parameters for the object
        params = config.get(object_name, {})

        if self.scale is None:
            if "scale" in params:
                self.scale = params["scale"]
            else:
                self.scale = 1.0
        if self.translation is None:
            if "translation" in params:
                self.translation = params["translation"]
            else:
                self.translation = np.array([0.0, 0.0, 0.0])
        if self.rotation is None:
            if "rotation" in params:
                euler_params = params["rotation"]
                self.rotation = R.from_euler("xyz", euler_params, degrees=True)
            else:
                self.rotation = R.from_euler("xyz", [0, 0, 0], degrees=True)

        if self.normal_orientation is None:
            if "normal_orientation" in params:
                self.normal_orientation = params["normal_orientation"]
            else:
                self.normal_orientation = 1

        # # Now, you can use scale_factor, rot, and voxel_size
        # print(f"scale_factor: {self.scale}")
        # # print(f"rot: {self.rotation.as_euler('xyz', degrees=True)}")
        # print(f"voxel_size: {self.voxel_size}")
        # print(f"translation: {self.translation}")
        # print(f"normal_orientation: {self.normal_orientation}")

    def load_parameters(self, dictionary_key):
        config_filepath = self.config

        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)

        all_params = config.get(self.object_name, {})
        key_params = all_params[dictionary_key]

        def set_attributes(obj, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    sub_obj = getattr(obj, key, type("SubParams", (), {})())
                    set_attributes(sub_obj, value)
                    setattr(obj, key, sub_obj)
                else:
                    setattr(obj, key, value)
                    print(f"Set {key} as {value}")

        set_attributes(self, key_params)

    def get_center(self):
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        self.center_point = np.mean(self.vertices, axis=0)
        _, center_vertex = self.kd_tree.query(np.array([self.center_point]), k=1)
        self.center_vertex = center_vertex[0]
        return self.center_point, self.center_vertex

    def get_closest_points(self, points):
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        distances, indices = self.kd_tree.query(points, k=1)
        return distances, indices

    def get_bounding_box(self, scale=1.0):
        self.oriented_bounding_box = self.mesh.get_oriented_bounding_box()
        self.oriented_bounding_box.scale(
            scale, center=self.oriented_bounding_box.get_center()
        )
        self.oriented_bounding_box_corners = np.asarray(
            self.oriented_bounding_box.get_box_points()
        )
        return self.oriented_bounding_box

    def get_bounding_sphere(self):
        if not hasattr(self, "center_point"):
            self.get_center()
        if not hasattr(self, "oriented_bounding_box"):
            self.get_bounding_box(scale=1.5)
        sphere_center = self.center_point

        enclosing_sphere_radius = np.max(
            np.linalg.norm(
                self.oriented_bounding_box_corners - self.oriented_bounding_box.center,
                axis=1,
            )
        )
        self.bounding_sphere = Sphere(
            radius=enclosing_sphere_radius, center=sphere_center
        )
        return self.bounding_sphere

    def get_bounding_box_grid(self, bounding_box_scalar=2.0, nb_points=5):
        self.get_bounding_box(bounding_box_scalar)

        min_vals = np.min(self.oriented_bounding_box_corners, axis=0)
        max_vals = np.max(self.oriented_bounding_box_corners, axis=0)

        x_min, y_min, z_min = min_vals
        x_max, y_max, z_max = max_vals
        grid = Grid(
            Nx=nb_points,
            Ny=nb_points,
            Nz=nb_points,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
        return grid

    def get_normals(self, num_neighbors=30):
        if not hasattr(self, "center_vertex"):
            self.get_center()
        self.mesh.compute_vertex_normals()
        normals = np.asarray(self.mesh.vertex_normals)
        self.normals = normals * self.normal_orientation
        return self.normals

    def get_signed_distance(self, position):
        distance, point_index = self.kd_tree.query(position, k=1)
        projected_point = self.vertices[point_index]
        projected_normal = self.normals[point_index]

        sign = np.sign(np.dot(position - projected_point, projected_normal))
        signed_distance = distance * sign
        return signed_distance, point_index

    def correct_distance_smooth(
        self, position, distance_target, epsilon=5e-4, max_iterations=15, max_error=1e-1
    ):
        position = np.float32(position)
        for i in range(max_iterations):
            signed_distance, point_index = self.get_signed_distance(position)
            error = distance_target - signed_distance
            if np.abs(error) < epsilon:
                return position, signed_distance, point_index
            elif np.abs(error) > max_error:
                error_sign = np.sign(error)
                error = error_sign * max_error

            local_normal = self.normals[point_index]
            correction = error * 0.2 * local_normal
            position += correction
        return position, signed_distance, point_index

    def get_local_basis(self):
        if not hasattr(self, "normals"):
            self.get_normals()
        self.tangent_vectors_u = np.column_stack(
            [self.normals[:, 1], -self.normals[:, 0], np.zeros(len(self.vertices))]
        )
        self.tangent_vectors_v = np.cross(self.normals, self.tangent_vectors_u)

    def get_k_edges(self, num_neighbors=3):
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        self.d_kdtree, idx = self.kd_tree.query(self.vertices, k=num_neighbors)

        idx = idx[:, 1:]
        point_numbers = np.arange(len(self.vertices))
        point_numbers = np.repeat(point_numbers, num_neighbors - 1)
        idx_flatten = idx.flatten()
        edges = np.vstack((point_numbers, idx_flatten)).T

        return edges

    def get_mean_edge_length(self):
        edges = self.get_k_edges()
        edge_vectors = self.vertices[edges[:, 1], :] - self.vertices[edges[:, 0], :]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        self.mean_edge_length = np.mean(edge_lengths)

        print(f"Mean edge length: {self.mean_edge_length*1e3:.1f} mm")
        return self.mean_edge_length

    def get_boundary(self, max_neighbors=30, angle_threshold=np.pi / 2):
        def is_boundary(
            vertex,
            neighbor_vertices,
            tangent_vector_u,
            tangent_vector_v,
            angle_threshold=3 * np.pi / 2,
        ):
            angles = np.zeros(len(neighbor_vertices))
            for j in range(len(neighbor_vertices)):
                neighbor = neighbor_vertices[j]
                delta = neighbor - vertex
                angles[j] = np.arctan2(
                    np.dot(delta, tangent_vector_u), np.dot(delta, tangent_vector_v)
                )

            angles = np.sort(angles)
            diff = np.diff(angles)
            diff = np.append(diff, 2 * np.pi - angles[-1] + angles[0])
            if np.max(diff) > angle_threshold:
                return True
            return False

        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        if not hasattr(self, "tangent_vectors_u"):
            self.get_local_basis()
        is_boundary_arr = np.zeros(len(self.vertices)).astype(bool)
        for i in range(len(self.vertices)):
            vertex = self.vertices[i]
            distances, indices = self.kd_tree.query(vertex, k=max_neighbors)
            neighbor_vertices = self.vertices[indices[0:]]
            is_boundary_arr[i] = is_boundary(
                vertex,
                neighbor_vertices,
                self.tangent_vectors_u[i],
                self.tangent_vectors_v[i],
                angle_threshold,
            )
        self.is_boundary_arr = is_boundary_arr
        return is_boundary_arr

    def get_boundary_normals(self):
        if not hasattr(self, "is_boundary_arr"):
            self.get_boundary()
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()

        boundary_vertices = self.vertices[self.is_boundary_arr]
        boundary_normal_vectors = np.zeros_like(boundary_vertices)

        for i, vertex in enumerate(boundary_vertices):
            distances, indices = self.kd_tree.query(vertex, k=10)
            neighbor_vertices = self.vertices[indices[1:]]

            inward_direction = np.mean(neighbor_vertices - vertex, axis=0)
            inward_direction /= np.linalg.norm(inward_direction)

            tangent_vector_u = self.tangent_vectors_u[self.is_boundary_arr][i]
            tangent_vector_v = self.tangent_vectors_v[self.is_boundary_arr][i]

            projected_u = np.dot(inward_direction, tangent_vector_u) * tangent_vector_u
            projected_v = np.dot(inward_direction, tangent_vector_v) * tangent_vector_v
            tangent_projection = projected_u + projected_v

            tangent_projection /= np.linalg.norm(tangent_projection)
            boundary_normal_vectors[i] = tangent_projection
        self.boundary_normals = boundary_normal_vectors
        self.boundary_tangents = np.cross(
            boundary_normal_vectors, self.normals[self.is_boundary_arr]
        )
        return boundary_normal_vectors

    def get_bases_from_tangent_vector_and_normal(self, tangent_vector):
        if not hasattr(self, "normals"):
            self.get_normals()

        # Ensure normals are normalized
        normal_norms = np.linalg.norm(self.normals, axis=1)[:, np.newaxis]
        normal_norms[normal_norms == 0] = 1  # Avoid division by zero
        normals = self.normals / normal_norms

        # Normalize tangent vector safely
        tangent_norms = np.linalg.norm(tangent_vector, axis=1)[:, np.newaxis]
        zero_tangent_mask = tangent_norms.flatten() == 0
        tangent_norms[tangent_norms == 0] = 1  # Avoid division by zero
        tangent_vector = tangent_vector / tangent_norms

        # For vertices with zero tangent vector, create arbitrary orthogonal vector
        if np.any(zero_tangent_mask):
            # Create a default tangent vector orthogonal to normal
            default_tangent = np.zeros_like(tangent_vector)
            # Use [1,0,0] unless normal is parallel to x-axis, then use [0,1,0]
            for i in np.where(zero_tangent_mask)[0]:
                n = normals[i]
                if abs(n[0]) < 0.9:  # not parallel to x-axis
                    default_tangent[i] = [1, 0, 0]
                else:  # parallel to x-axis, use y-axis
                    default_tangent[i] = [0, 1, 0]
                # Make it orthogonal to normal
                default_tangent[i] = (
                    default_tangent[i] - np.dot(default_tangent[i], n) * n
                )
                default_tangent[i] /= np.linalg.norm(default_tangent[i])
            tangent_vector[zero_tangent_mask] = default_tangent[zero_tangent_mask]

        # Create orthogonal y-vector using cross product
        y_vector = np.cross(normals, tangent_vector)

        # Normalize y_vector safely
        y_norms = np.linalg.norm(y_vector, axis=1)[:, np.newaxis]
        y_norms[y_norms == 0] = 1  # Avoid division by zero
        y_vector = y_vector / y_norms

        # Recompute tangent vector to ensure orthogonality (Gram-Schmidt)
        tangent_vector = np.cross(y_vector, normals)

        # Normalize the recomputed tangent vector
        tangent_norms = np.linalg.norm(tangent_vector, axis=1)[:, np.newaxis]
        tangent_norms[tangent_norms == 0] = 1  # Avoid division by zero
        tangent_vector = tangent_vector / tangent_norms

        if np.any(np.isnan(tangent_vector)):
            print("ERROR! NaN in tangent vector")
        if np.any(np.isnan(y_vector)):
            print("ERROR! NaN in y vector")
        if np.any(np.isnan(normals)):
            print("ERROR! NaN in normals")

        # Create right-handed coordinate system: [tangent, y, normal]
        local_bases = np.stack([tangent_vector, y_vector, normals], axis=2)
        self.local_bases = local_bases
