# Diffused Fields Scripts

本文件夹包含 **Diffused Orientation Fields (DOF)** 核心算法包的演示脚本和工具。这些脚本展示了如何使用扩散方向场进行几何计算、方向场求解、轨迹生成以及与基准方法的比较。

---

## 📁 目录结构

```
scripts/
├── pointcloud_quaternion_diffusion.py    # 四元数扩散求解器演示
├── wos_orientation.py                    # Walk-on-Spheres方向场计算
├── wos_vector.py                         # Walk-on-Spheres向量扩散
├── local_frames_trajectory_rollout.py    # 基于局部坐标系的轨迹展开
├── comparisons/                          # 与基准方法的比较脚本
│   ├── nearest_frame_comparison.py       # 最近帧基线比较
│   ├── vector_diffusion_comparison.py    # 独立向量扩散与正交化比较
│   └── vector_projection_comparison.py   # 向量投影基线比较
└── utils/                                # 工具脚本
    ├── visualize_pointcloud.py           # 点云可视化工具
    ├── test_endpoint_detection.py        # 自动端点检测测试
    └── test_pointcloud_methods.py        # 点云变换方法测试
```

---

## 🚀 运行方式

所有脚本都需要在激活虚拟环境后运行：

```powershell
cd "c:\GitHub\kuka\robot control\DOF"
.venv\Scripts\activate
cd diffused_fields
```

---

## 🔬 核心演示脚本

### 1. pointcloud_quaternion_diffusion.py

**功能**：在点云上计算扩散旋转场（四元数扩散）

**技术原理**：
- 使用四元数表示旋转，避免万向锁问题
- 在点云上求解方向值扩散偏微分方程
- 通过Dirichlet边界条件指定关键点的旋转

**使用方法**：
```powershell
python scripts/pointcloud_quaternion_diffusion.py
```

**主要步骤**：
1. 加载点云（默认 `rectangular_grid.ply`）
2. 创建四元数扩散求解器
3. 设置源顶点和旋转角度（平面或非平面关键点）
4. 执行四元数扩散计算
5. 可视化扩散后的方向场

**可配置参数**：
- `diffusion_scalar`: 扩散标量（越大越平滑，默认20）
- `source_vertices`: 源顶点索引列表
- `z_angle`: 每个源顶点的旋转角度（平面情况）
- `quats`: 四元数列表（非平面情况）

**输出**：Polyscope 3D可视化窗口，显示点云及其扩散方向场

---

### 2. wos_orientation.py

**功能**：使用 Walk-on-Spheres (WoS) 蒙特卡洛方法在环境空间中计算方向场

**技术原理**：
- 将点云作为边界条件
- 在包围盒网格上计算扩散方向场
- 使用蒙特卡洛方法求解环境空间中的扩散方程
- 提取横截面进行可视化

**使用方法**：
```powershell
python scripts/wos_orientation.py
```

**主要步骤**：
1. 加载点云（默认 `spot.ply`）
2. 创建标量扩散求解器，计算点云表面的局部坐标系
3. 创建 Walk-on-Spheres 扩散求解器（环境空间）
4. 生成包围盒网格并提取横截面
5. 在网格上计算扩散方向场
6. 可视化点云表面和环境空间中的方向场

**输出**：
- 点云表面的局部坐标系（红色箭头）
- 横截面网格上的扩散方向场
- 扩散标量场的标量值

---

### 3. wos_vector.py

**功能**：使用 Walk-on-Spheres 方法将点云法向量扩散到环境空间

**技术原理**：
- 将点云法向量作为边界条件
- 在环境空间中扩散向量场
- 保持向量的方向连续性

**使用方法**：
```powershell
python scripts/wos_vector.py
```

**主要步骤**：
1. 加载点云并计算法向量
2. 设置点云为扩散边界
3. 创建 WoS 扩散求解器
4. 生成网格并提取横截面
5. 在网格上扩散向量场
6. 可视化点云法向量和扩散后的向量场

**输出**：
- 点云（黑色）和其法向量（紫色）
- 横截面网格上的扩散向量（蓝色）

---

### 4. local_frames_trajectory_rollout.py

**功能**：基于扩散方向场生成机器人轨迹

**技术原理**：
- 使用局部坐标系定义轨迹方向
- 通过 Walk-on-Spheres 在环境空间中跟踪方向场
- 支持自定义轴序列（如 +z → +x → -y）

**使用方法**：
```powershell
python scripts/local_frames_trajectory_rollout.py
```

**主要步骤**：
1. 加载点云并设置参数
2. 计算标量扩散和局部坐标系
3. 定义轨迹轴序列（如 `["+z", "+x", "-y"]`）
4. 执行轨迹展开（trajectory_rollout）
5. 计算速度统计信息
6. 可视化轨迹和方向场

**可配置参数**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `initial_vertex` | 初始顶点索引 | 350 |
| `source_vertices` | 源顶点列表 | [338] |
| `distance_to_surface` | 距表面距离 | -0.06 |
| `trajectory_rollout_steps` | 每轴步数 | 8 |
| `step_size` | 步长 | 0.007 |

**输出**：
- 轨迹路径（红色曲线）
- 轨迹上的局部坐标系
- 速度统计信息

---

## 📊 比较脚本（comparisons/）

### 1. nearest_frame_comparison.py

**功能**：比较 DOF 方法与最近帧基线方法

**比较内容**：
- **DOF**：基于扩散的方向场（平滑、连续）
- **Nearest Frame**：最近点云顶点的局部坐标系（不连续）

**使用方法**：
```powershell
python scripts/comparisons/nearest_frame_comparison.py
```

**输出**：
- 点云表面方向场
- 横截面网格上两种方法的方向场对比
- 可选：角度偏差分析

---

### 2. vector_diffusion_comparison.py

**功能**：比较 DOF 方法与独立向量扩散+正交化方法

**比较内容**：
| 方法 | 原理 | 特点 |
|------|------|------|
| **DOF** | 直接扩散旋转（四元数） | 保持正交性，平滑连续 |
| **Normalized Vectors** | 独立扩散三个向量后归一化 | 不保证正交 |
| **Orthonormal (z-fixed)** | 归一化后固定z轴进行正交化 | z轴固定，可能引入扭曲 |
| **Orthonormal (SVD)** | SVD分解进行正交化 | 最优正交逼近 |

**使用方法**：
```powershell
python scripts/comparisons/vector_diffusion_comparison.py
```

**输出**：
- 四种方法在横截面网格上的局部x轴对比（不同颜色）
- 处理时间统计
- 可选：角度偏差分析

---

### 3. vector_projection_comparison.py

**功能**：比较 DOF 方法与向量投影基线方法

**比较内容**：
- **DOF**：基于标量扩散的方向场
- **Vector Projection**：最近帧的向量投影

**使用方法**：
```powershell
python scripts/comparisons/vector_projection_comparison.py
```

**输出**：
- 点云上两种方法的方向场对比
- 角度偏差分析可视化

---

## 🛠️ 工具脚本（utils/）

### 1. visualize_pointcloud.py

**功能**：可视化点云及其原始颜色

**使用方法**：
```powershell
python scripts/utils/visualize_pointcloud.py
```

**特性**：
- 加载点云并显示原始颜色
- 如果没有颜色信息，使用高度梯度
- 显示法向量和法向量幅值
- 输出点云统计信息（顶点数、包围盒等）

---

### 2. test_endpoint_detection.py

**功能**：测试自动端点检测算法的鲁棒性

**测试场景**：
- 生成50个随机形变的香蕉点云
- 每个点云应用随机缩放、扭曲、弯曲
- 使用两阶段扩散自动检测端点
- 在网格布局中可视化所有结果

**使用方法**：
```powershell
python scripts/utils/test_endpoint_detection.py
```

**输出**：
- 50个形变香蕉的网格可视化
- 每个香蕉标记检测到的端点（红色）
- 形变参数统计信息

---

### 3. test_pointcloud_methods.py

**功能**：测试点云变换方法

**测试内容**：
| 方法 | 功能 | 参数示例 |
|------|------|----------|
| `add_gaussian_noise()` | 添加高斯噪声 | `noise_std=0.002` |
| `apply_scaling()` | 各向异性缩放 | `[1.2, 0.8, 1.1]` |
| `create_holes()` | 创建孔洞 | `num_holes=5, hole_radius=0.003` |
| `apply_bend()` | 弯曲变换 | `bend_axis=2, curvature=0.02` |
| `apply_bulge()` | 膨胀变换 | `amount=0.01` |
| `apply_twist()` | 扭曲变换 | `axis=2, twist_strength=1.0` |

**使用方法**：
```powershell
python scripts/utils/test_pointcloud_methods.py
```

**输出**：
- 所有变换后的点云在 Polyscope 中可视化
- 每种方法的成功/失败状态
- 变换参数摘要

---

## 📁 支持的点云文件

所有脚本默认使用以下点云文件（位于 `data/pointclouds/`）：

| 文件 | 描述 | 适用脚本 |
|------|------|----------|
| `rectangular_grid.ply` | 矩形网格点云 | 四元数扩散 |
| `spot.ply` | Spot模型点云 | WoS方向场、比较脚本 |
| `pear.ply` | 梨的点云 | 可视化工具 |
| `banana_half.ply` | 半根香蕉点云 | 端点检测、变换测试 |

---

## 🎨 可视化说明

所有脚本使用 **Polyscope** 进行3D可视化：

- **窗口操作**：
  - 左键拖动：旋转视角
  - 右键拖动：平移视角
  - 滚轮：缩放
  - `Esc`：关闭窗口

- **图例**：
  - 红色箭头：局部坐标系x轴（纵向方向）
  - 绿色箭头：局部坐标系y轴（切向方向）
  - 蓝色箭头：局部坐标系z轴（法向方向）
  - 黑色曲线：轨迹路径

---

## 📚 技术依赖

| 依赖 | 用途 |
|------|------|
| `numpy` | 数值计算 |
| `scipy` | 稀疏矩阵与线性代数 |
| `open3d` | 点云处理 |
| `polyscope` | 3D可视化 |
| `potpourri3d` | 几何处理 |
| `robust_laplacian` | 离散拉普拉斯算子 |
| `tqdm` | 进度条（比较脚本） |

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

版权所有 (c) 2025 Idiap Research Institute
