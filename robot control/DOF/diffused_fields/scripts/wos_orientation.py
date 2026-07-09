"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Computes an orientation field on a pointcloud conditioned on
keypoints.
"""

from diffused_fields.diffusion import PointcloudScalarDiffusion, WalkOnSpheresDiffusion
from diffused_fields.manifold import *
from diffused_fields.manifold.manifold import extract_plane
from diffused_fields.visualization.plotting_ps import *

# Select the object
# ==========================================
# 选择要处理的点云模型文件。
# 该文件会被 Pointcloud 类读取，并作为后续扩散与方向场计算的几何基础。
filename = "spot.ply"


# 创建点云对象。
# 点云包含顶点、法向量、邻接关系等几何信息，后续局部坐标系和扩散计算都依赖它。
pcloud = Pointcloud(filename=filename)
# 在点云表面上构造标量扩散对象。
# diffusion_scalar 是扩散强度/尺度参数，用于生成平滑的标量场并辅助建立局部参考系。
scalar_diffusion = PointcloudScalarDiffusion(pcloud, diffusion_scalar=1000)
# 计算点云各顶点的局部基(local bases)，通常由局部法向与切向方向组成。
# 后续可视化点云方向场时会直接使用这些局部基。
scalar_diffusion.get_local_bases()
# the object itself is the boundary condition for the diffusion
# on the ambient space (robot's workspace)
# 在环境空间中做扩散时，边界条件由物体本身提供。
# 这里将点云对象作为唯一边界，供 Walk-on-Spheres 求解器使用。
boundaries = [pcloud]


# Monte Carlo diffusion solver for the ambient space
# 创建 Walk-on-Spheres (WoS) 蒙特卡洛扩散求解器。
# convergence_threshold 使用点云平均边长的 2 倍，作为随机游走接近边界时的停止尺度。
wos_diffusion = WalkOnSpheresDiffusion(
    boundaries=boundaries,
    convergence_threshold=pcloud.get_mean_edge_length() * 2,
)

# We will compute the diffused field at the grid points for visualizing the result
# 在点云包围盒内生成规则网格，用于在环境空间中采样并可视化扩散后的方向场。
# bounding_box_scalar=1 表示使用原始包围盒尺度，nb_points=11 表示每个轴上采样 11 个点。
grid = pcloud.get_bounding_box_grid(bounding_box_scalar=1, nb_points=11)

# Get the center coordinates of the grid
# 计算网格中心点坐标，后面会以中心位置为截面提取一个平面。
grid.get_center()

# Extract cross-sections at mid-points using grid.center
# 只保留位于网格中心 x 坐标处的一个截面平面，便于在 3D 空间中观察方向场切片。
# extract_plane 会筛出满足 x = grid.center[0] 的网格点。
grid.vertices = extract_plane(grid.vertices, axis="x", value=grid.center[0])
# 在该截面网格点上求解环境空间中的扩散方向场。
# 结果通常会写回 grid 对象，例如填充 grid.local_bases 以供可视化使用。
wos_diffusion.diffuse_orientations_on_grid(grid)


# 初始化 polyscope 可视化环境。
ps.init()
# 设置相机和参考平面，方便查看点云与截面上的方向场。
set_camera_and_plane()
# 可视化原始点云上的局部方向场。
# point_radius=0 表示不强调点本身，只显示方向基/向量信息。
ps_field = plot_orientation_field(
    pcloud.vertices, pcloud.local_bases, name="pcloud", point_radius=0
)
# 将标量扩散值附加到点云显示对象上，便于通过颜色查看标量场分布。
ps_field.add_scalar_quantity(name="u0", values=scalar_diffusion.ut)
# 高亮显示扩散源点/关键点位置。
plot_sources(pcloud.vertices[scalar_diffusion.source_vertices])
# 可视化截面网格上的扩散方向场。
# enable_x 与 enable_z 控制显示局部基中的 x、z 方向，用于更清楚地观察姿态传播结果。
plot_orientation_field(
    grid.vertices,
    grid.local_bases,
    name="grid",
    enable_x=True,
    enable_z=True,
    point_radius=0,
)

# 打开交互式可视化窗口。
ps.show()
