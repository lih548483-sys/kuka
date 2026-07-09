"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import copy
import time

# import imgui
import numpy as np
import polyscope as ps
from scipy.spatial.transform import Rotation as R

from ..manifold import Mesh


def set_camera_and_plane():
    ps.set_up_dir("y_up")
    ps.set_front_dir("neg_x_front")

    ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction


def plot_tool_trajectory(
    position_trajectory, orientation_trajectory, mesh, name="tool"
):

    for i, position in enumerate(position_trajectory):
        mesh_c = copy.deepcopy(mesh.mesh)
        target_pos = position + mesh.center_offset
        mesh_c.translate(target_pos, relative=False)
        rot = (
            orientation_trajectory[i]
            @ R.from_euler("xyz", [0, 0, 0], degrees=False).as_matrix()
        )
        mesh_c.rotate(rot, center=mesh_c.get_center() - mesh.center_offset)

        vertices = np.asarray(mesh_c.vertices)
        faces = np.asarray(mesh_c.triangles)
        ps_mesh_source = ps.register_surface_mesh(
            f"{name} {i}", vertices, faces, transparency=0.5, color=[125, 125, 125]
        )
        # colors = np.asarray(mesh_c.vertex_colors)
        # print(f'Shape of the colors is : {colors.shape}')

        # ps_mesh.add_color_quantity("test_vals", colors,
        #                        defined_on='vertices',
        #                        enabled=True)


def animate_tool_trajectory(position_trajectory, orientation_trajectory, mesh):
    imgui = ps.imgui
    for i, position in enumerate(position_trajectory):
        mesh_c = copy.deepcopy(mesh.mesh)
        target_pos = position + mesh.center_offset
        mesh_c.translate(target_pos, relative=False)
        rot = (
            orientation_trajectory[i]
            @ R.from_euler("xyz", [0, 0, 0], degrees=False).as_matrix()
        )
        mesh_c.rotate(rot, center=mesh_c.get_center() - mesh.center_offset)

        vertices = np.asarray(mesh_c.vertices)
        faces = np.asarray(mesh_c.triangles)

        # Track frame index globally
        frame = {"i": 0}
        # Initialize Polyscope
        ps.init()
        ps.set_ground_plane_mode("none")
        ps_mesh = ps.register_surface_mesh(
            "moving mesh", vertices, faces, transparency=1.0, color=[125, 125, 125]
        )
        # Animation state
        state = {
            "frame": 0,
            "last_time": time.time(),
            "playing": True,
            "speed": 0.1,  # seconds per frame
        }

        # User input handler
        def my_callback():
            # Set the window position on the right side
            io = imgui.GetIO()
            # screen_width = io.DisplaySize.x
            # screen_height = io.DisplaySize.y
            # screen_width, screen_height = (
            #     io.DisplaySize
            # )  # Unpacking the ImVec2 into tuple (width, height)

            window_width = 300  # Adjust this width as necessary
            window_height = 150  # Adjust this height as necessary

            # Set window position to the right side (just offset from the edge)
            imgui.SetWindowPos(
                (
                    1000 - window_width - 10,
                    1000 // 2 - window_height // 2,
                )
            )

            imgui.Begin("Animation Control", False)
            changed, state["speed"] = imgui.SliderFloat(
                "Speed (s/frame)", state["speed"], 0.01, 1.0
            )
            if imgui.Button("Play/Pause [Space]"):
                state["playing"] = not state["playing"]
            imgui.Text("Press SPACE to toggle play/pause")
            imgui.End()

            # Listen for spacebar
            # io = imgui.GetIO()
            if io.KeysDown[32]:  # 32 is the keycode for space
                if not state.get("space_pressed", False):
                    state["playing"] = not state["playing"]
                    state["space_pressed"] = True
            else:
                state["space_pressed"] = False

            # Time-based update
            current_time = time.time()
            if state["playing"] and (
                current_time - state["last_time"] > state["speed"]
            ):
                state["last_time"] = current_time
                state["frame"] = (state["frame"] + 1) % len(position_trajectory)

                i = state["frame"]
                mesh_c = copy.deepcopy(mesh.mesh)
                target_pos = position_trajectory[i] + mesh.center_offset
                mesh_c.translate(target_pos, relative=False)

                rot = (
                    orientation_trajectory[i]
                    @ R.from_euler("xyz", [0, 0, 0], degrees=False).as_matrix()
                )
                mesh_c.rotate(rot, center=mesh_c.get_center() - mesh.center_offset)

                vertices = np.asarray(mesh_c.vertices)
                ps_mesh.update_vertex_positions(vertices)

        # Set the callback and show the animation
        ps.set_user_callback(my_callback)
        ps.show()
        ps.clear_user_callback()


def import_tool_mesh(tool_params):
    if tool_params.name == "knife":
        mesh = import_knife(tool_params)

    elif tool_params.name == "spoon":
        mesh = import_spoon(tool_params)
    return mesh


def import_knife(tool_params):
    mesh_filename = "knife.obj"  # vertical stripe
    mesh = Mesh(
        filename=mesh_filename,
        scale=tool_params.scale,
        rotation=R.from_euler("xyz", tool_params.orientation, degrees=True),
        center_vertex=tool_params.center_vertex,
    )
    return mesh


def import_spoon(tool_params):
    mesh_filename = "spoon.stl"  # vertical stripe
    mesh = Mesh(
        filename=mesh_filename,
        scale=tool_params.scale,
        rotation=R.from_euler("xyz", tool_params.orientation, degrees=True),
        center_vertex=tool_params.center_vertex,
    )
    return mesh


def plot_orientation_field(
    vertices,
    bases=None,
    name="",
    vector_length=None,
    vector_radius=None,
    point_radius=0.01,
    enable=True,
    enable_vector=False,
    enable_x=True,
    enable_z=False,
    color=None,
):
    if (len(vertices.shape) != 2) or (vertices.shape[1] not in (2, 3)):
        vertices = [vertices]

    ps_field = ps.register_point_cloud(
        name,
        vertices,
        radius=point_radius,
        transparency=1,
        color=[0.0, 0.0, 0.0],
        enabled=enable,
    )
    ps.set_ground_plane_mode("none")
    # if color is not None:
    #     ps_field.add_color_quantity("colors", color)
    if bases is None:
        return ps_field
    ps_field.add_vector_quantity(
        f"x_{name}",
        bases[:, :, 0],
        color=[1, 0, 0],
        length=vector_length,
        radius=vector_radius,
        enabled=enable_x,
    )
    ps_field.add_vector_quantity(
        f"y_{name}",
        bases[:, :, 1],
        color=[0, 1, 0],
        length=vector_length,
        radius=vector_radius,
        enabled=enable_vector,
    )
    ps_field.add_vector_quantity(
        f"z_{name}",
        bases[:, :, 2],
        color=[0, 0, 1],
        length=vector_length,
        radius=vector_radius,
        enabled=enable_z,
    )

    return ps_field


def plot_world_frame():
    plot_orientation_field(
        np.array([[0, 0, 0]]),
        np.array([np.eye(3)]),
        name="world",
        vector_length=0.1,
        vector_radius=0.005,
        enable_vector=True,
    )


def plot_point_cloud(
    vertices,
    name="",
    point_radius=None,
):

    ps_field = ps.register_point_cloud(
        name, vertices, radius=point_radius, color=[0, 0, 1], transparency=1
    )

    return ps_field


def plot_sources(vertices):
    ps.register_point_cloud(
        "sources",
        vertices,
        color=[0, 0, 1],
        radius=0.03,
    )


def animate_orientation_trajectory(vertices, bases_arr):

    # Initialize Polyscope
    ps.init()
    ps.set_ground_plane_mode("none")
    idx = 0
    ps_field = ps.register_point_cloud("o", vertices, radius=1e-5)

    bases = bases_arr[idx]
    ps_field.add_vector_quantity(f"x_", bases[:, :, 0], color=[1, 0, 0], enabled=True)
    ps_field.add_vector_quantity(f"y_", bases[:, :, 1], color=[0, 1, 0], enabled=False)
    ps_field.add_vector_quantity(f"z_", bases[:, :, 2], color=[0, 0, 1], enabled=False)

    # User input handler
    def my_callback():
        nonlocal idx
        bases = bases_arr[idx]

        ps_field.update_vector_quantity(
            f"x_", bases[:, :, 0], color=[1, 0, 0], enabled=True
        )
        ps_field.add_vector_quantity(
            f"y_", bases[:, :, 1], color=[0, 1, 0], enabled=False
        )
        ps_field.add_vector_quantity(
            f"z_", bases[:, :, 2], color=[0, 0, 1], enabled=False
        )
        idx += 1
        print(idx)
        if idx == 99:
            idx = 0

        # time.sleep(0.01)
        # Time-based update

    # Set the callback and show the animation
    ps.set_user_callback(my_callback)
    ps.show()
    ps.clear_user_callback()
