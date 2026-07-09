"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Creates a trajectory by following a simple action sequence on local reference frames
"""

import numpy as np

from diffused_fields.diffusion import PointcloudScalarDiffusion, WalkOnSpheresDiffusion
from diffused_fields.manifold import Pointcloud
from diffused_fields.visualization.plotting_ps import *

# Configuration
# ==========================================
# 输入点云文件名。Pointcloud 会按项目默认路径/当前运行路径查找该文件。
filename = "spot.ply"


# 创建点云对象，并估计/读取每个顶点的法向量。后续初始位置会沿法向量偏移，
# 局部坐标系和扩散场计算也依赖点云几何信息。
pcloud = Pointcloud(filename=filename)
pcloud.get_normals()


# Load teleoperation parameters and create scalar diffusion
# initial_vertex: 轨迹起点参考的点云顶点编号。
pcloud.initial_vertex = 350
# source_vertices: 标量扩散场的源点顶点编号列表，通常用来定义吸引/参考区域。
pcloud.source_vertices = [338]
# distance_to_surface: 初始位置相对点云表面的法向偏移距离。
# 负值表示沿法向反方向偏移，正值表示沿法向方向偏移。
pcloud.distance_to_surface = -0.06
# trajectory_rollout_steps: 轨迹在每个指定轴向上前进的步数。
pcloud.trajectory_rollout_steps = 8
# step_size: 每一步沿当前局部轴向移动的距离。
pcloud.step_size = 0.007

# 构造点云上的标量扩散对象，并计算每个点的局部基(local bases)。
# local_bases 后面会作为局部参考坐标系，用于解释 "+x"、"-y" 等动作指令。
scalar_diffusion = PointcloudScalarDiffusion(pcloud, diffusion_scalar=1000)
scalar_diffusion.get_local_bases()

# Calculate initial position for trajectory
# 起始位置不是直接放在 initial_vertex 顶点上，而是沿该点法向量偏移一定距离。
# 这样可以让轨迹从点云表面附近的空间位置开始，而不是严格贴在表面顶点上。
initial_position = (
    pcloud.vertices[pcloud.initial_vertex]
    + pcloud.normals[pcloud.initial_vertex] * pcloud.distance_to_surface
)

# Set up boundaries and WoS diffusion solver
# Walk-on-Spheres 扩散求解器需要边界对象列表；这里点云本身就是唯一边界。
boundaries = [pcloud]
# convergence_threshold 使用点云平均边长的 2 倍，作为 WoS 迭代接近边界/收敛的判定尺度。
wos_diffusion = WalkOnSpheresDiffusion(
    boundaries=boundaries,
    convergence_threshold=pcloud.get_mean_edge_length() * 2,
)

# Define trajectory information explicitly in the script
# ====================================================


# Define the sequence of axes to follow during trajectory rollout
# 轨迹动作序列。每个字符串表示在当前局部坐标系下沿某个轴向前进。
# 例如 "+z" 表示沿局部 z 轴正方向移动，之后再按 "+x"、"-y" 继续 rollout。
trajectory_axis_sequence = ["+z", "+x", "-y"]
# Alternative examples:

# Define custom direction mappings (axis_index, sign)
# axis_index: 0=x-axis, 1=y-axis, 2=z-axis
# sign: +1=positive direction, -1=negative direction
# direction_mappings 将人类可读的动作字符串映射到具体的局部基向量索引和方向符号。
# trajectory_rollout 会根据这个字典把 "+x" 等指令转换成实际移动方向。
trajectory_direction_mappings = {
    "+x": [0, 1],  # Move along positive x-axis
    "-x": [0, -1],  # Move along negative x-axis
    "+y": [1, 1],  # Move along positive y-axis
    "-y": [1, -1],  # Move along negative y-axis
    "+z": [2, 1],  # Move along positive z-axis
    "-z": [2, -1],  # Move along negative z-axis
}


print(f"Trajectory configuration:")
print(f"  - Axis sequence: {trajectory_axis_sequence}")
print(f"  - Steps per axis: {pcloud.trajectory_rollout_steps}")
print(f"  - Step size: {pcloud.step_size}")
print(f"  - Direction mappings: {trajectory_direction_mappings}")

# Execute trajectory rollout
# trajectory_rollout 返回：
# positions: rollout 得到的轨迹点序列，形状通常为 (轨迹点数量, 3)。
# local_bases: 每个轨迹点处估计/传播得到的局部坐标系，用于可视化姿态场。
positions, local_bases = wos_diffusion.trajectory_rollout(
    initial_position=initial_position,
    steps=pcloud.trajectory_rollout_steps,
    step_size=pcloud.step_size,
    axis_sequence=trajectory_axis_sequence,
    direction_mappings=trajectory_direction_mappings,
)

# Calculate and display trajectory statistics
# 相邻位置做差得到每一步的位移向量；这里命名为 velocities，用于统计轨迹步长大小。
velocities = np.diff(positions, axis=0)
print(f"\nTrajectory results:")
print(f"  - Total positions: {len(positions)}")
print(f"  - Velocity shape: {velocities.shape}")
print(
    f"  - Average velocity magnitude: {np.mean(np.linalg.norm(velocities, axis=1)):.6f}"
)
# Visualization
# 初始化 polyscope 可视化窗口。
ps.init()

# 设置默认相机视角和地面/参考平面，方便观察轨迹和点云。
set_camera_and_plane()
# Plot trajectory with orientation field
# 每隔 4 个轨迹点绘制一次局部坐标系，避免箭头过密影响观察。
# enable_vector 和 enable_z 控制是否额外显示方向向量/z 轴信息。
plot_orientation_field(
    positions[::4],
    local_bases[::4],
    "trajectory",
    enable_vector=True,
    enable_z=True,
    vector_length=0.05,
    vector_radius=0.01,
)

# Plot point cloud with scalar diffusion field
# 绘制点云顶点上的局部坐标系，并把标量扩散结果 ut 作为 scalar quantity 附加到点云显示对象上。
# 这样在 polyscope 中可以通过颜色查看扩散场分布。
ps_field = plot_orientation_field(pcloud.vertices, pcloud.local_bases, "pcloud")
ps_field.add_scalar_quantity(values=scalar_diffusion.ut, name="diffusion_field")

# Plot trajectory path
# 用红色曲线网络连接所有轨迹点，显示 rollout 得到的完整路径。
ps.register_curve_network(
    "trajectory_path",
    positions,
    radius=0.005,
    edges="line",
    color=[1, 0, 0],
)

# 打开 polyscope 交互式窗口。
ps.show()
