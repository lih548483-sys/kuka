"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import random

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from ..visualization.plotting_ps import *
from .pointcloud_scalar_diffusion import PointcloudScalarDiffusion

# ========== Quaternion helpers ==========


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_from_axis_angle(axis, angle_deg):
    ang = np.deg2rad(angle_deg)
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    u = axis / n
    c = np.cos(ang / 2.0)
    s = np.sin(ang / 2.0)
    return quat_normalize(np.array([c, *(u * s)]))


def quat_dot(q1, q2):
    return float(np.dot(q1, q2))


def quat_log(q, eps=1e-12):
    """Stable principal log: returns a 3 vector (pure quaternion)."""
    q = quat_normalize(q)
    w, x, y, z = q
    v = np.array([x, y, z], dtype=float)
    vn = np.linalg.norm(v)
    if vn < eps:
        return np.zeros(3)
    phi = np.arctan2(vn, w)  # in [0, pi)
    return v * (phi / vn)


def quat_exp(v, eps=1e-12):
    """Exponential map from a 3 vector to a unit quaternion."""
    v = np.asarray(v, dtype=float)
    phi = np.linalg.norm(v)
    if phi < eps:
        return np.array([1.0, 0.0, 0.0, 0.0])
    u = v / phi
    return np.array([np.cos(phi), *(u * np.sin(phi))])


def expquat_2_rotated_frame(pure_quaternions):
    """
    Convert pure quaternions (N x 3 array) to rotation matrices (N x 3 x 3 array).
    Each 3-vector is converted to a quaternion via quat_exp, then to a rotation matrix.
    """
    pure_quaternions = np.asarray(pure_quaternions)
    if pure_quaternions.ndim == 1:
        pure_quaternions = pure_quaternions.reshape(1, -1)

    N = pure_quaternions.shape[0]
    bases = np.zeros((N, 3, 3))

    for i in range(N):
        quat = quat_exp(pure_quaternions[i])
        # Convert quaternion to rotation matrix using scipy
        r = R.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]
        )  # scipy uses [x,y,z,w] format
        bases[i] = r.as_matrix()

    return bases


def get_quat_between_vectors(v1, v2):
    """
    Get the quaternion that rotates v1 to v2.
    Both vectors should be unit vectors or will be normalized.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Compute rotation axis and angle
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    # Handle parallel or anti-parallel vectors
    if axis_norm < 1e-12:
        dot = np.dot(v1, v2)
        if dot > 0:  # Same direction
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:  # Opposite direction - find perpendicular axis
            if abs(v1[0]) < 0.9:
                axis = np.cross(v1, [1, 0, 0])
            else:
                axis = np.cross(v1, [0, 1, 0])
            axis = axis / np.linalg.norm(axis)
            return np.array([0.0, axis[0], axis[1], axis[2]])

    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    # Convert to quaternion
    half_angle = angle / 2.0
    return np.array([np.cos(half_angle), *(axis * np.sin(half_angle))])


# ========== Sign search on S^3 ==========


def best_sign_assignment(quats):
    """
    Try all 2^{N-1} sign patterns relative to q0.
    Maximize the minimum absolute dot among all pairs.
    """
    Q = [quat_normalize(q) for q in quats]
    N = len(Q)
    if N <= 1:
        return Q

    best = None
    best_min_dot = -np.inf
    for mask in range(1 << (N - 1)):
        cand = [Q[0].copy()]
        for i in range(1, N):
            flip = -1.0 if (mask & (1 << (i - 1))) else 1.0
            cand.append(flip * Q[i])
        # score this assignment
        m = +np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = abs(quat_dot(cand[i], cand[j]))
                if d < m:
                    m = d
        if m > best_min_dot:
            best_min_dot = m
            best = [c.copy() for c in cand]
    return best


# ========== Your entry points ==========


def quats_from_z_angles(z_deg_list):
    """Make unit quaternions for pure z rotations."""
    return [quat_from_axis_angle([0, 0, 1], deg) for deg in z_deg_list]


def pure_quaternions_for_dirichlet(Q, antipode_tol_deg=179.9):
    """
    Input: list of unit quaternions Q (N x 4 array-like)
    Output: list of pure quaternion 3 vectors for Dirichlet data
    """

    # 2) global sign choice on S^3
    Q = best_sign_assignment(Q)

    # 3) optional guard for near pi pairs
    def rel_angle_deg(q1, q2):
        d = abs(quat_dot(q1, q2))
        d = np.clip(d, -1.0, 1.0)
        return np.degrees(2.0 * np.arccos(d))

    worst = 0.0
    for i in range(len(Q)):
        for j in range(i + 1, len(Q)):
            worst = max(worst, rel_angle_deg(Q[i], Q[j]))
    if worst >= antipode_tol_deg:
        print("Warning: a boundary pair is near one hundred eighty degrees.")

    # 4) logs for Dirichlet data
    return [quat_log(q) for q in Q], Q  # returns logs and the lifted quats


# child class
class PointcloudQuaternionDiffusion(PointcloudScalarDiffusion):
    def __init__(
        self,
        pcloud,
        diffusion_scalar=100,
        # method="laplace",
        method="LU",
        num_eigen=None,
        num_integration_steps=1,
    ):

        super().__init__(
            pcloud,
            diffusion_scalar=diffusion_scalar,
            method=method,
            num_eigen=num_eigen,
            num_integration_steps=num_integration_steps,
        )

        self.pcloud = pcloud

    def load_parameters(self):
        config_filepath = self.pcloud.file_directory + "config.yaml"
        # Load the YAML file
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)

        # Retrieve the parameters for the object
        params = config.get(self.pcloud.object_name, {})

        quaternion_diffusion_dict = params.get("quaternion_diffusion", {})
        if "source_vertices" in quaternion_diffusion_dict:
            self.source_vertices = quaternion_diffusion_dict["source_vertices"]
        else:
            self.source_vertices = None
        print(f"Source Vertices: {self.source_vertices}")

    def diffuse_quaternions(self):
        if not hasattr(self, "source_pure_quaternions"):
            self.get_random_source_vertices()
        # In the Euclidean space we can diffuse the scalar components independently
        # then combine to a vector field
        vf = np.zeros((len(self.pcloud.vertices), 3))  # Final vector field
        for i in range(3):
            # Consider each component of the vector field separately as a scalar field
            u0 = np.zeros(len(self.pcloud.vertices))
            for j, vertex in enumerate(self.source_vertices):
                u0[vertex] = self.source_pure_quaternions[j, i]  # mind the i,j ordering
            vf[:, i] = self.integrate_diffusion(u0)

        # Note that the vector diffusion finds the correct direction but changes
        # the magnitude of the vectors. Diffuse the magnitudes seperately to
        # recover the correct magnitudes after diffusion
        u0 = np.zeros(len(self.pcloud.vertices))
        for i, vertex in enumerate(self.source_vertices):
            u0[vertex] = np.linalg.norm(self.source_pure_quaternions[i, :])
        uf = self.integrate_diffusion(u0)

        phi_0 = np.zeros(len(self.pcloud.vertices))
        for i, vertex in enumerate(self.source_vertices):
            phi_0[vertex] = 1
        # Solve the linear system
        phi_f = self.integrate_diffusion(phi_0)

        # Normalize the vector field
        vf = vf / np.linalg.norm(vf, axis=1)[:, None]
        vf = vf * uf[:, None] / phi_f[:, None]
        self.diffused_pure_quaternions = vf

    def steady_state_diffuse_quaternions(self):
        if not hasattr(self, "source_pure_quaternions"):
            self.get_random_source_vertices()
        self.pcloud.boundary_points = self.source_vertices
        # In the Euclidean space we can diffuse the scalar components independently
        # then combine to a vector field
        vf = np.zeros((len(self.pcloud.vertices), 3))  # Final vector field
        for i in range(3):
            # Consider each component of the vector field separately as a scalar field
            u0 = np.zeros(len(self.pcloud.vertices))
            for j, vertex in enumerate(self.source_vertices):
                u0[vertex] = self.source_pure_quaternions[j, i]  # mind the i,j ordering
            vf[:, i] = self.integrate_diffusion(u0)

        # # Note that the vector diffusion finds the correct direction but changes
        # # the magnitude of the vectors. Diffuse the magnitudes seperately to
        # # recover the correct magnitudes after diffusion
        # u0 = np.zeros(len(self.pcloud.vertices))
        # for i, vertex in enumerate(self.source_vertices):
        #     u0[vertex] = np.linalg.norm(self.source_pure_quaternions[i, :])
        # uf = self.integrate_diffusion(u0)

        # phi_0 = np.zeros(len(self.pcloud.vertices))
        # for i, vertex in enumerate(self.source_vertices):
        #     phi_0[vertex] = 1
        # # Solve the linear system
        # phi_f = self.integrate_diffusion(phi_0)

        # # Normalize the vector field
        # vf = vf / np.linalg.norm(vf, axis=1)[:, None]
        # vf = vf * uf[:, None] / phi_f[:, None]
        self.diffused_pure_quaternions = vf

    def set_random_source_vertices(self):
        self.source_vertices = random.sample(
            range(0, self.pcloud.vertices.shape[0]), self.num_sources
        )

    def set_random_directions(self):
        if not hasattr(self, "num_source"):
            self.set_random_source_vertices()
        # sample directions from the unit sphere
        direction_vectors = 2 * np.random.rand(3 * self.num_sources) - 1
        direction_vectors = direction_vectors.reshape(self.num_sources, 3)
        direction_vectors = (
            direction_vectors / np.linalg.norm(direction_vectors, axis=1)[:, None]
        )
        self.directions = direction_vectors

    def set_pure_quaternions_from_directions(self):
        if not hasattr(self, "directions"):
            self.set_random_directions()
        x_vector = np.array([1, 0, 0])  # global x axis
        source_pure_quaternions = np.zeros_like(self.directions)
        for i in range(len(self.directions)):
            # get the rotation which rotates the x_vector to the desired vector
            quaternion = get_quat_between_vectors(x_vector, self.directions[i, :])
            source_pure_quaternions[i, :] = quat_log(
                quaternion
            )  # work on the Lie algebra
        self.source_pure_quaternions = source_pure_quaternions

    def set_random_sources(self, num_sources=5):
        self.num_sources = num_sources
        self.set_random_source_vertices()
        self.set_random_directions()
        self.set_pure_quaternions_from_directions()

    def set_random_planar_sources(
        self, source_vertices=None, z_angle=None, num_sources=5
    ):
        if source_vertices is None:
            self.num_sources = num_sources
            self.set_random_source_vertices()
        else:
            self.source_vertices = source_vertices
            self.num_sources = len(source_vertices)
        if z_angle is None:
            z_angle = 2 * 180 * np.random.rand(self.num_sources)
            z_angle = z_angle.reshape(self.num_sources, 1)

        Q = quats_from_z_angles(z_angle)
        self.source_pure_quaternions, _ = pure_quaternions_for_dirichlet(Q)
        self.source_pure_quaternions = np.array(self.source_pure_quaternions)

    def set_demo_sources(self):
        self.num_sources = 3
        self.source_vertices = [223, 520, 701]
        base_rot = R.from_matrix(np.eye(3))
        rot_2 = R.from_euler("z", np.pi / 2) * base_rot
        rot_3 = R.from_euler("y", np.pi / 2) * base_rot
        self.source_pure_quaternions = np.row_stack(
            [
                base_rot.as_rotvec(),
                rot_2.as_rotvec(),
                rot_3.as_rotvec(),
            ]
        )

    def visualize_diffused_quaternions(self):
        vector_length = 0.05 / 2
        vector_radius = 0.035 / 2
        point_radius = 0.003

        ps.init()

        self.pcloud.local_bases = expquat_2_rotated_frame(
            self.diffused_pure_quaternions
        )

        # rots = reconstruct_from_relative_logs(
        #     self.R_mean, self.diffused_pure_quaternions
        # )
        # self.pcloud.local_bases = np.array([r.as_matrix() for r in rots])

        # Point Cloud
        # ==============================================================================
        ps_object = plot_orientation_field(
            self.pcloud.vertices,
            self.pcloud.local_bases,
            name="object point cloud",
            vector_length=vector_length,
            vector_radius=vector_radius,
            point_radius=point_radius,
        )

        # Sources
        # ==============================================================================
        # Longer diffusion times might result in over-smoothed fields where the sources
        # change direction. In order to check it we compare the diffused/undiffused
        # sources

        # Debugging
        # ==============================================================================
        # ps_sources_diffused = plot_orientation_field(
        #     self.pcloud.vertices[self.source_vertices],
        #     self.pcloud.local_bases[self.source_vertices],
        #     name="sources diffused",
        #     vector_length=4 * vector_length,
        #     vector_radius=2 * vector_radius,
        #     point_radius=point_radius,
        #     enable_vector=False,
        #     enable_x=True,
        # )
        # ==============================================================================

        source_bases_original = expquat_2_rotated_frame(self.source_pure_quaternions)
        # rots = reconstruct_from_relative_logs(self.R_mean, self.source_pure_quaternions)
        # source_bases_original = np.array([r.as_matrix() for r in rots])

        ps_sources_undiffused = plot_orientation_field(
            self.pcloud.vertices[self.source_vertices],
            source_bases_original,
            name="sources original",
            vector_length=4 * vector_length,
            vector_radius=2 * vector_radius,
            point_radius=point_radius,
        )

        ps.show()
