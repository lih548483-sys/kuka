# Diffused Orientation Fields (DOF) 项目总览

本项目包含两个核心包，用于基于扩散方向场的物体中心任务表示与迁移：

- **diffused_fields** - 核心算法包：在几何流形上求解方向值扩散偏微分方程
- **diffused_fields_robotics** - 机器人应用包：基于DOF的物体中心机器人操作应用

---

## 📁 项目结构

```
DOF/
├── diffused_fields/              # 核心扩散算法包
│   ├── src/diffused_fields/      # 源代码目录
│   │   ├── core/                 # 核心算法与数据结构
│   │   ├── diffusion/            # 扩散求解器（标量、向量、四元数）
│   │   ├── manifold/             # 流形操作与几何计算
│   │   ├── utils/                # 工具函数（关键点检测等）
│   │   ├── visualization/        # 可视化工具（Polyscope）
│   │   └── baselines/            # 基准方法比较
│   ├── scripts/                  # 示例脚本
│   │   ├── comparisons/          # 与基准方法的对比脚本
│   │   └── utils/                # 测试与可视化工具
│   ├── data/                     # 示例数据
│   │   ├── pointclouds/          # 点云文件 (.ply)
│   │   └── meshes/               # 网格模型 (.obj)
│   └── config/                   # 配置文件
├── diffused_fields_robotics/     # 机器人操作应用包
│   ├── src/diffused_fields_robotics/
│   │   ├── core/                 # 核心配置
│   │   ├── local_action_primitives/  # 局部动作原语
│   │   └── utils/                # 工具函数
│   ├── scripts/                  # 机器人应用脚本
│   │   ├── analysis/             # 数据分析与可视化
│   │   └── batch_experiments/    # 批量实验脚本
│   ├── data/                     # 策略模型数据
│   └── results/                  # 实验结果与截图
└── .venv/                        # Python虚拟环境
```

---

## 🚀 快速开始

### 1. 安装要求

- Python 3.12+
- Git LFS（用于存储大型数据文件）

### 2. 环境配置

```powershell
# 进入项目目录
cd "c:\GitHub\kuka\robot control\DOF"

# 激活虚拟环境
.venv\Scripts\activate

# 验证安装
python -c "import diffused_fields; import diffused_fields_robotics; print('安装成功!')"
```

如果虚拟环境未配置或需要重新安装：

```powershell
# 创建虚拟环境
python3.12 -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate

# 安装两个包（editable模式）
pip install -e diffused_fields -e diffused_fields_robotics
```

---

## 🎯 核心功能

### 1. 扩散方向场计算 (diffused_fields)

| 模块 | 功能 |
|------|------|
| `manifold.Pointcloud` | 点云数据结构与操作 |
| `diffusion.pointcloud_quaternion_diffusion` | 四元数扩散求解 |
| `diffusion.walk_on_spheres` | Walk-on-Spheres蒙特卡洛方法 |
| `visualization.plotting_ps` | Polyscope 3D可视化 |
| `baselines.nearest_frame_baseline` | 最近帧基准方法 |

### 2. 机器人操作原语 (diffused_fields_robotics)

| 脚本 | 功能 | 命令 |
|------|------|------|
| `slicing.py` | 切片操作（香蕉等物体） | `python scripts/slicing.py` |
| `peeling.py` | 削皮操作（梨等物体） | `python scripts/peeling.py` |
| `coverage.py` | 触觉覆盖操作 | `python scripts/coverage.py` |
| `policy_transfer.py` | 策略迁移 | `python scripts/policy_transfer.py` |

---

## 📊 示例数据

### 点云数据 (`diffused_fields/data/pointclouds/`)

| 文件 | 描述 |
|------|------|
| `banana_half.ply` | 半根香蕉的点云 |
| `pear.ply` | 梨的点云 |
| `spot.ply` | Spot模型点云 |
| `rectangular_grid.ply` | 矩形网格点云 |
| `stiffness_sample.ply` | 刚度采样点云 |

### 网格模型 (`diffused_fields/data/meshes/`)

| 文件 | 描述 |
|------|------|
| `knife.obj` | 刀具网格模型 |

---

## 🧪 实验脚本

### 批量实验 (`scripts/batch_experiments/`)

| 脚本 | 功能 |
|------|------|
| `batch_peeling.py` | 跨物体削皮迁移实验 |
| `batch_slicing_geometric_noise.py` | 几何噪声鲁棒性测试 |
| `batch_slicing_keypoint_noise.py` | 关键点噪声鲁棒性测试 |
| `batch_slicing_topological_noise.py` | 拓扑噪声鲁棒性测试 |

### 数据分析 (`scripts/analysis/`)

| 脚本 | 功能 |
|------|------|
| `visualize_results.py` | 结果可视化 |
| `batch_peeling_stats_primitives.py` | 削皮统计分析 |
| `robustness.py` | 鲁棒性分析 |
| `visualize_ft_data.py` | 力/力矩数据可视化 |

---

## 🔧 配置文件

| 配置文件 | 用途 |
|----------|------|
| `diffused_fields/config/pointclouds.yaml` | 点云数据路径配置 |
| `diffused_fields/config/meshes.yaml` | 网格模型配置 |
| `diffused_fields_robotics/config/action_primitives.yaml` | 动作原语参数 |

---

## 📝 使用示例

### 示例1：运行切片操作

```powershell
cd diffused_fields_robotics
python scripts/slicing.py
```

### 示例2：运行削皮操作

```powershell
cd diffused_fields_robotics
python scripts/peeling.py
```

### 示例3：可视化点云

```powershell
cd diffused_fields
python scripts/utils/visualize_pointcloud.py
```

### 示例4：运行批量实验

```powershell
cd diffused_fields_robotics
python scripts/batch_experiments/batch_peeling.py
python scripts/analysis/batch_peeling_stats_primitives.py
```

---

## 📚 依赖说明

| 依赖 | 用途 |
|------|------|
| `numpy` | 数值计算 |
| `scipy` | 稀疏矩阵与线性代数 |
| `open3d` | 点云处理 |
| `polyscope` | 3D可视化 |
| `potpourri3d` | 几何处理 |
| `robust_laplacian` | 离散拉普拉斯算子 |
| `stable-baselines3` | 强化学习 |
| `PyYAML` | 配置文件解析 |

---

## 📄 版权信息

本项目由Idiap Research Institute开发，基于MIT许可证。

**论文引用**：
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