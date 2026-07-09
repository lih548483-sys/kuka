# diffused_fields Core Library

本文件夹包含 **Diffused Orientation Fields (DOF)** 核心算法库的源代码。DOF 是一种基于扩散的方向场计算方法，能够在几何流形上求解方向值扩散偏微分方程，生成平滑、连续的局部坐标系场。

---

## 📁 目录结构

```
diffused_fields/src/
└── diffused_fields/
    ├── __init__.py                    # 包入口，导出核心类和函数
    ├── core/                          # 核心配置模块
    │   ├── __init__.py
    │   └── config.py                  # 集中式配置管理
    ├── manifold/                      # 几何流形模块
    │   ├── __init__.py
    │   ├── manifold.py                # 基础流形类（Plane, Line, Sphere）
    │   ├── pointcloud.py              # 点云流形类（核心数据结构）
    │   ├── mesh.py                    # 网格流形类
    │   └── grid.py                    # 规则网格流形类
    ├── diffusion/                     # 扩散算法模块
    │   ├── __init__.py
    │   ├── pointcloud_scalar_diffusion.py      # 标量扩散求解器
    │   ├── pointcloud_quaternion_diffusion.py  # 四元数扩散求解器
    │   └── walk_on_spheres.py         # Walk-on-Spheres蒙特卡洛扩散求解器
    ├── baselines/                     # 基准方法模块
    │   ├── __init__.py
    │   └── nearest_frame_baseline.py  # 最近帧基线方法
    ├── utils/                         # 工具函数模块
    │   ├── __init__.py
    │   └── keypoint_detection.py      # 自动关键点检测
    └── visualization/                 # 可视化模块
        ├── __init__.py
        └── plotting_ps.py             # Polyscope可视化工具
```

---

## 🏗️ 架构设计

### 核心设计理念

DOF 库采用 **分层架构**，将几何表示与扩散算法分离：

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Application)                      │
│    机器人操作原语、轨迹生成、方向场可视化                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    扩散算法层 (Diffusion)                    │
│    标量扩散、四元数扩散、Walk-on-Spheres                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    几何流形层 (Manifold)                     │
│    Pointcloud、Mesh、Grid、Plane、Line、Sphere               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    基础配置层 (Core)                         │
│    路径管理、配置加载                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 模块详解

### 1. Core 模块 (`core/`)

#### config.py

**功能**：集中式配置管理，提供项目路径解析和配置文件加载功能。

**核心类**：`Config`（静态类）

**主要方法**：

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `get_project_root()` | 获取项目根目录 | `Path` 对象 |
| `get_data_dir()` | 获取数据目录 | `Path` 对象 |
| `get_pointclouds_dir()` | 获取点云数据目录 | `Path` 对象 |
| `get_meshes_dir()` | 获取网格数据目录 | `Path` 对象 |
| `get_config_dir()` | 获取配置文件目录 | `Path` 对象 |
| `get_pointclouds_config_path()` | 获取点云配置文件路径 | `Path` 对象 |
| `load_pointcloud_config(object_name)` | 加载特定点云的配置 | `Dict` |
| `resolve_pointcloud_path(filename)` | 将文件名解析为完整路径 | `Path` 对象 |

**设计特点**：
- 路径自动检测，无需硬编码
- 配置文件支持 YAML 格式
- 支持缓存机制，避免重复计算

---

### 2. Manifold 模块 (`manifold/`)

#### manifold.py

**功能**：定义基础几何流形类及其操作。

**核心类**：

| 类名 | 说明 | 关键属性 |
|------|------|----------|
| `Manifold` | 所有流形的基类 | `type`, `scale`, `translation`, `rotation` |
| `Plane` | 平面流形 | `normal`, `point`, `angle` |
| `Line` | 直线流形 | `direction`, `point` |
| `Sphere` | 球流形 | `radius`, `center` |

**关键方法**：

- `get_closest_points(points)`：计算查询点到流形的最近点
- `get_local_bases(points_on_manifold)`：计算流形上点的局部坐标系
- `sample_points()`：采样流形上的点用于可视化

#### pointcloud.py

**功能**：点云流形类，是 DOF 库的**核心数据结构**。

**核心类**：`Pointcloud(Manifold)`

**初始化参数**：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `vertices` | `np.ndarray` | 顶点坐标 | `None` |
| `colors` | `np.ndarray` | 顶点颜色 | `None` |
| `filename` | `str` | PLY文件名 | `None` |
| `voxel_size` | `float` | 体素下采样尺寸 | `None` |
| `scale` | `float` | 缩放因子 | `None` |
| `translation` | `np.ndarray` | 平移向量 | `None` |
| `rotation` | `Rotation` | 旋转 | `None` |
| `normal_orientation` | `int` | 法线方向（±1） | `None` |

**核心功能**：

| 方法 | 说明 |
|------|------|
| `get_normals(num_neighbors=20)` | 计算点云法线 |
| `get_kd_tree()` | 构建 KD-Tree 用于最近邻查询 |
| `get_mean_edge_length()` | 计算平均边长（用于扩散步长） |
| `get_boundary()` | 检测边界点 |
| `get_bases_from_tangent_vector_and_normal(tangent)` | 从切向量和法线构建局部坐标系 |
| `correct_distance_smooth(position, target)` | 平滑校正点到表面的距离 |
| `add_gaussian_noise(noise_std)` | 添加高斯噪声 |
| `apply_bend(bend_axis, curvature)` | 弯曲变换 |
| `apply_twist(axis, twist_strength)` | 扭曲变换 |
| `create_holes(num_holes, hole_radius)` | 创建孔洞（拓扑噪声） |

**设计特点**：
- 支持从文件或内存数组初始化
- 自动加载对象特定参数（从 `pointclouds.yaml`）
- 提供完整的点云处理工具链
- 支持多种形变操作（用于鲁棒性测试）

#### mesh.py

**功能**：网格流形类，支持三角网格的加载和操作。

**核心类**：`Mesh(Manifold)`

**特点**：
- 继承自 `Manifold`
- 支持 OBJ/STL 格式网格
- 提供与 `Pointcloud` 类似的接口
- 主要用于工具模型（刀具、勺子等）

#### grid.py

**功能**：规则网格流形类，用于环境空间中的方向场插值。

**核心类**：`Grid(Manifold)`

**初始化参数**：

| 参数 | 说明 |
|------|------|
| `Nx`, `Ny`, `Nz` | 各维度的网格点数 |
| `x_min`, `x_max` 等 | 各维度的边界 |

**核心方法**：
- `laplacian_3d_matrix()`：构建 3D 拉普拉斯矩阵
- `solve_scalar_diffusion(u0)`：求解标量扩散方程
- `diffuse_vector_directions(source_vectors)`：扩散向量方向
- `visualize_angular_deviations()`：可视化方向场平滑度

---

### 3. Diffusion 模块 (`diffusion/`)

#### pointcloud_scalar_diffusion.py

**功能**：点云上的标量扩散求解器，是 DOF 的**核心算法**。

**核心类**：`PointcloudScalarDiffusion(DiffusionSolver)`

**初始化参数**：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `pcloud` | `Pointcloud` | 点云对象 | 必填 |
| `diffusion_scalar` | `float` | 扩散标量（越大越平滑） | `1` |
| `method` | `str` | 求解方法 | `"LU"` |
| `num_integration_steps` | `int` | 积分步数 | `1` |

**求解方法选项**：

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `"invert"` | 直接矩阵求逆 | 小规模点云 |
| `"LU"` | LU 分解 | 中等规模点云（推荐） |
| `"eigen"` | 特征值分解 | 需要频谱分析 |
| `"laplace"` | 拉普拉斯方程求解 | 稳态问题 |

**核心流程**：

```
1. get_laplacian()        → 计算离散拉普拉斯算子
2. prefactor_matrices()   → 预因子化系统矩阵
3. set_sources()          → 设置源顶点（Dirichlet边界条件）
4. integrate_diffusion()  → 积分扩散方程
5. get_gradient()         → 计算梯度得到方向场
6. get_local_bases()      → 构建局部坐标系
```

**关键方法**：

| 方法 | 说明 |
|------|------|
| `get_local_bases()` | 完整计算扩散方向场和局部坐标系 |
| `solve_heat_method(sources)` | 使用热方法计算测地线距离 |
| `get_endpoints()` | 自动检测点云端点 |
| `rotate_local_bases_around_z(angle)` | 旋转局部坐标系 |

#### pointcloud_quaternion_diffusion.py

**功能**：点云上的四元数扩散求解器，用于**直接扩散旋转**（避免万向锁问题）。

**核心类**：`PointcloudQuaternionDiffusion(PointcloudScalarDiffusion)`

**继承关系**：继承自 `PointcloudScalarDiffusion`，复用其扩散求解基础设施

**四元数扩散原理**：

传统的向量扩散方法在处理旋转时存在以下问题：
1. 三个向量分量独立扩散后，无法保证正交性
2. 归一化操作会破坏扩散的连续性

四元数扩散方法的解决方案：
1. 将旋转表示为四元数（单位四元数位于 S³ 上）
2. 使用对数映射将四元数映射到李代数（纯四元数，3D 向量）
3. 在欧几里得空间中扩散纯四元数的三个分量
4. 使用指数映射将扩散结果映射回四元数
5. 通过分别扩散幅度来恢复正确的旋转角度

**四元数工具函数**：

| 函数 | 说明 |
|------|------|
| `quat_normalize(q)` | 归一化四元数 |
| `quat_from_axis_angle(axis, angle)` | 从轴角表示创建四元数 |
| `quat_log(q)` | 四元数对数映射（映射到李代数） |
| `quat_exp(v)` | 四元数指数映射（从李代数映射回四元数） |
| `get_quat_between_vectors(v1, v2)` | 获取将 v1 旋转到 v2 的四元数 |
| `best_sign_assignment(quats)` | S³ 上的符号搜索，最大化最小点积 |
| `pure_quaternions_for_dirichlet(Q)` | 将四元数转换为 Dirichlet 边界条件 |
| `expquat_2_rotated_frame(pure_quats)` | 将纯四元数转换为旋转矩阵 |

**核心方法**：

| 方法 | 说明 |
|------|------|
| `diffuse_quaternions()` | 扩散四元数（含幅度恢复） |
| `steady_state_diffuse_quaternions()` | 稳态四元数扩散（Dirichlet边界） |
| `set_random_sources(num_sources)` | 设置随机源顶点和方向 |
| `set_random_planar_sources()` | 设置平面旋转源（绕 z 轴旋转） |
| `set_pure_quaternions_from_directions()` | 从方向向量构建纯四元数 |
| `visualize_diffused_quaternions()` | 可视化扩散后的方向场 |

**使用示例**：

```python
from diffused_fields.manifold import Pointcloud
from diffused_fields.diffusion import PointcloudQuaternionDiffusion

# 1. 加载点云
pcloud = Pointcloud(filename="rectangular_grid.ply")

# 2. 创建四元数扩散求解器
quaternion_diffusion = PointcloudQuaternionDiffusion(
    pcloud,
    diffusion_scalar=20,
    method="LU"
)

# 3. 设置源顶点和旋转角度（平面情况）
quaternion_diffusion.set_random_planar_sources(
    source_vertices=[100, 200],
    z_angle=[0, 90]  # 两个源顶点分别旋转 0° 和 90°
)

# 4. 执行四元数扩散
quaternion_diffusion.diffuse_quaternions()

# 5. 可视化结果
quaternion_diffusion.visualize_diffused_quaternions()
```

#### walk_on_spheres.py

**功能**：Walk-on-Spheres (WoS) 蒙特卡洛扩散求解器，用于**环境空间中的方向场插值**。

**核心类**：`WalkOnSpheresDiffusion`

**初始化参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `boundaries` | 边界流形列表 | `[]` |
| `batch_size` | 批处理大小 | `512` |
| `max_iterations` | 最大迭代次数 | `24` |
| `convergence_threshold` | 收敛阈值 | `1e-3` |

**核心算法原理**：

WoS 是一种蒙特卡洛方法，用于求解拉普拉斯方程的狄利克雷问题：
1. 从查询点出发，在边界之间随机游走
2. 每次游走距离为到最近边界的距离
3. 收敛到边界后，根据边界条件计算值
4. 多次采样取平均得到扩散结果

**核心方法**：

| 方法 | 说明 |
|------|------|
| `diffuse_rotations(points)` | 扩散旋转（四元数平均） |
| `diffuse_vectors(points, vector_type)` | 扩散向量（法线/切线） |
| `diffuse_scalars(points)` | 扩散标量值 |
| `diffuse_orientations_on_grid(grid)` | 在网格上扩散方向场 |
| `trajectory_rollout(initial_position, steps)` | 沿方向场展开轨迹 |
| `move_multistep(num_steps, x0, direction)` | 多步移动 |

**设计特点**：
- 支持批处理，提高计算效率
- 支持多种扩散类型（标量、向量、旋转）
- 支持轨迹生成和点云投影

---

### 4. Baselines 模块 (`baselines/`)

#### nearest_frame_baseline.py

**功能**：最近帧基线方法，用于与 DOF 方法进行对比。

**核心类**：`NearestFrameBaseline`

**原理**：
1. 找到查询点到点云表面的最近点
2. 使用最近点的法线作为局部坐标系的 z 轴
3. 将源向量投影到切线平面得到 x 轴
4. 通过叉积计算 y 轴

**核心方法**：

| 方法 | 说明 |
|------|------|
| `compute_local_frame(query_point)` | 计算单个查询点的局部坐标系 |
| `compute_local_frame_wos(query_point)` | 使用 WoS 扩散法线 |
| `compute_orientation_field(query_points)` | 计算多个查询点的方向场 |

**对比优势**：
- 计算简单，速度快
- 但方向场不连续，在边界处会跳变

---

### 5. Utils 模块 (`utils/`)

#### keypoint_detection.py

**功能**：自动关键点检测工具。

**核心函数**：

| 函数 | 说明 |
|------|------|
| `find_endpoints_via_diffusion(pcloud)` | 使用两阶段扩散检测端点 |
| `find_endpoints_via_extremal_projection(pcloud)` | 使用 PCA 投影检测端点 |
| `visualize_detected_endpoints(pcloud, endpoints)` | 可视化检测结果 |

**端点检测流程**（扩散法）：
1. 从中心点开始扩散，找到最"冷"的点作为第一个端点
2. 从第一个端点开始扩散，找到最"冷"的点作为第二个端点
3. 验证两个端点之间的距离是否合理

---

### 6. Visualization 模块 (`visualization/`)

#### plotting_ps.py

**功能**：基于 Polyscope 的可视化工具。

**核心函数**：

| 函数 | 说明 |
|------|------|
| `plot_point_cloud(vertices, name)` | 绘制点云 |
| `plot_orientation_field(vertices, bases)` | 绘制方向场（三个坐标轴） |
| `plot_tool_trajectory(positions, orientations, mesh)` | 绘制工具轨迹 |
| `animate_tool_trajectory(positions, orientations, mesh)` | 动画播放工具轨迹 |
| `import_tool_mesh(tool_params)` | 加载工具网格模型（刀具/勺子） |
| `plot_world_frame()` | 绘制世界坐标系 |

**可视化颜色约定**：
- **红色**：局部坐标系 x 轴（纵向方向）
- **绿色**：局部坐标系 y 轴（切向方向）
- **蓝色**：局部坐标系 z 轴（法向方向）

---

## 🔗 模块依赖关系

```
config.py
    ├── pointcloud.py
    ├── mesh.py
    └── grid.py

manifold.py
    ├── pointcloud.py (继承)
    ├── mesh.py (继承)
    ├── grid.py (继承)
    └── walk_on_spheres.py (使用)

pointcloud_scalar_diffusion.py
    ├── pointcloud.py (输入)
    └── plotting_ps.py (可视化)

pointcloud_quaternion_diffusion.py
    ├── pointcloud_scalar_diffusion.py (继承)
    ├── pointcloud.py (输入)
    └── plotting_ps.py (可视化)

walk_on_spheres.py
    ├── pointcloud.py (边界)
    ├── mesh.py (边界)
    ├── manifold.py (边界)
    └── plotting_ps.py (可视化)

nearest_frame_baseline.py
    ├── pointcloud.py (输入)
    └── walk_on_spheres.py (可选)

keypoint_detection.py
    ├── pointcloud.py (输入)
    └── pointcloud_scalar_diffusion.py (扩散)

plotting_ps.py
    ├── mesh.py (工具模型)
    └── pointcloud.py (点云)
```

---

## 🚀 使用示例

### 基本用法：计算点云方向场

```python
from diffused_fields.manifold import Pointcloud
from diffused_fields.diffusion import PointcloudScalarDiffusion

# 1. 加载点云
pcloud = Pointcloud(filename="pear.ply")

# 2. 创建标量扩散求解器
scalar_diffusion = PointcloudScalarDiffusion(
    pcloud,
    diffusion_scalar=1000,
    method="LU"
)

# 3. 设置源顶点（关键点）
scalar_diffusion.source_vertices = [1271, 2190]

# 4. 计算方向场和局部坐标系
scalar_diffusion.get_local_bases()

# 5. 获取结果
# pcloud.local_bases: N x 3 x 3 数组，存储每个顶点的局部坐标系
# scalar_diffusion.ut: N x 1 数组，存储扩散标量场
# scalar_diffusion.diffused_vectors: N x 3 数组，存储扩散向量场
```

### 使用 WoS 在环境空间中扩散方向场

```python
from diffused_fields.manifold import Pointcloud, Grid
from diffused_fields.diffusion import WalkOnSpheresDiffusion

# 1. 加载点云并计算局部坐标系
pcloud = Pointcloud(filename="spot.ply")
scalar_diffusion = PointcloudScalarDiffusion(pcloud)
scalar_diffusion.source_vertices = [338]
scalar_diffusion.get_local_bases()

# 2. 创建包围盒网格
grid = pcloud.get_bounding_box_grid(bounding_box_scalar=2.0, nb_points=10)

# 3. 创建 WoS 求解器
wos = WalkOnSpheresDiffusion(boundaries=[pcloud])

# 4. 在网格上扩散方向场
wos.diffuse_orientations_on_grid(grid)

# 5. grid.local_bases 包含网格上每个点的局部坐标系
```

### 轨迹生成

```python
# 使用 WoS 沿方向场展开轨迹
initial_position = pcloud.vertices[350] + pcloud.normals[350] * 0.06
positions, local_bases = wos.trajectory_rollout(
    initial_position=initial_position,
    steps=8,
    step_size=0.007,
    axis_sequence=["+z", "+x", "-y"]
)
```

---

## 📊 核心数据结构

### 局部坐标系表示

DOF 使用 **3x3 旋转矩阵**表示局部坐标系，列向量分别为：

| 列 | 含义 | 方向 |
|----|------|------|
| `[:, :, 0]` | x 轴（纵向） | 沿扩散梯度方向 |
| `[:, :, 1]` | y 轴（切向） | 垂直于纵向和法向 |
| `[:, :, 2]` | z 轴（法向） | 指向物体外部 |

### 扩散流程的数据流动

```
输入: 点云 + 源顶点
       │
       ▼
┌──────────────────────┐
│  拉普拉斯算子计算     │  → C (拉普拉斯矩阵), M (质量矩阵)
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  系统矩阵预因子化     │  → A_factorized (LU分解)
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  扩散方程积分        │  → ut (标量场)
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  梯度计算           │  → diffused_vectors (方向场)
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  局部坐标系构建      │  → local_bases (Nx3x3)
└──────────────────────┘
       │
       ▼
输出: 局部坐标系场 + 轨迹
```

---

## 📚 技术依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| `numpy` | >= 1.20 | 数值计算 |
| `scipy` | >= 1.7 | 稀疏矩阵、线性代数 |
| `open3d` | >= 0.15 | 点云处理 |
| `polyscope` | >= 1.2 | 3D可视化 |
| `potpourri3d` | >= 1.0 | 热方法求解器 |
| `robust_laplacian` | >= 0.1 | 离散拉普拉斯算子 |
| `pcdiff` | >= 0.1 | 梯度算子构建 |
| `PyYAML` | >= 6.0 | 配置文件解析 |

---

## 📝 引用

本代码基于以下论文：

```bibtex
@online{bilalogluTactileErgodicControl2024,
  title = {Tactile {{Ergodic Control Using Diffusion}} and {{Geometric Algebra}}},
  author = {Bilaloglu, Cem and Löw, Tobias and Calinon, Sylvain},
  date = {2024-02-07},
  eprint = {2402.04862},
  eprinttype = {arxiv},
  eprintclass = {cs},
  url = {http://arxiv.org/abs/2402.04862}
}
```

---

## 📮 联系信息

维护者：Cem Bilaloglu (cem.bilaloglu@idiap.ch)

版权所有 (c) 2024-2025 Idiap Research Institute
