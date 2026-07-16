# DOF 骨骼弧面轨迹生成 - 开发复盘

## 项目概述

本项目基于 **DOF（Diffused Orientation Fields）** 算法，实现了在骨骼点云弧面上生成贴合表面的轨迹，轨迹上每个点附带局部坐标系，可直接用于机械臂控制。

---

## 对话记录

### 对话 1：项目梳理与功能需求

**时间**: 2026-07-15  
**用户需求**:
1. 梳理 DOF 文件夹中不同程序的关系
2. 创建 `bone_cuting` 子文件夹
3. 实现骨头上弧面轨迹生成功能：在点云上选择起点和终点，自动生成弧面连线轨迹，每个点附带局部坐标系

**实现内容**:
- 创建了 `arc_trajectory.py` - 核心轨迹生成类，基于标量扩散和 Walk-on-Spheres 算法
- 创建了 `point_selector.py` - 点选择器，支持交互式选点、自动选点、索引选点
- 创建了 `run_bone_arc.py` - 运行脚本，提供命令行接口

**核心算法**:
1. **标量扩散**: 在起点和终点设置热源，求解热传导方程，梯度方向给出表面切线方向场
2. **Walk-on-Spheres**: 将表面局部坐标系扩散到自由空间
3. **轨迹展开**: 从起点沿方向场逐步前进，每步投影回弧面

---

### 对话 2：Polyscope 中文乱码问题

**时间**: 2026-07-15  
**问题描述**:
Polyscope 使用的 Dear ImGui 默认只加载 ASCII 字符集，中文名称（如"骨骼"、"轨迹"）渲染为问号 `?`

**解决方案**:
采用双重保障策略：

**方案一：UI 名称英文化**（核心保障）
| 原中文名称 | 新英文名称 |
|-----------|-----------|
| 骨骼点云 | Bone Point Cloud |
| 弧面轨迹 | Arc Trajectory |
| 轨迹坐标系 | Trajectory Frames |
| 起终点 | Endpoints |
| 颜色 | colors |

**方案二：中文字体加载**（增强体验）
- 创建 `font_utils.py` 模块
- 自动检测系统中文字体（支持 Windows/macOS/Linux）
- 使用 Polyscope 的 `set_prepare_imgui_fonts_callback` API 加载字体
- 容错设计：字体加载失败不影响程序运行

**修改文件**:
- `arc_trajectory.py` - 修改 `_visualize_polyscope` 方法
- 新增 `font_utils.py` - 中文字体工具模块

---

### 对话 3：轨迹终点提前终止问题

**时间**: 2026-07-15  
**问题描述**:
轨迹总长度 (12.00 mm) < 起终点直线距离 (12.68 mm)，出现几何悖论。正常情况下弧面轨迹长度应 ≥ 直线距离。

**问题根因**:
1. **终点位置未偏移**: 起点位置沿法线偏移了 `distance_to_surface`，但终点位置仍是原始表面位置，两者不在同一比较基准上
2. **终止条件逻辑错误**: 原代码中试图通过 `abs(dist_to_end - abs(self.distance_to_surface))` 来补偿偏移，但这个逻辑不正确

**解决方案**:
1. 将终点位置也沿法线偏移，与起点保持一致的偏移平面
2. 简化终止条件，直接比较当前位置到终点位置的欧氏距离

**修改文件**:
- `arc_trajectory.py` - 修改 `_rollout_trajectory` 和 `_check_endpoint_reached` 方法

**修改内容**:
```python
# 修改前
self.end_position = self.pcloud.vertices[self.end_idx]

# 修改后
self.end_position = (
    self.pcloud.vertices[self.end_idx]
    + self.pcloud.normals[self.end_idx] * self.distance_to_surface
)
```

---

### 对话 4：算法参数优化与硬编码修复

**时间**: 2026-07-15  
**问题描述**:
1. 默认步长 2mm + 容差倍数 30.0 导致轨迹过粗且终点提前终止
2. `resolve_ply_path` 存在硬编码 `Bone_m_later.ply`，无法通过 `--ply` 自由指定
3. Polyscope 名称中残留空格/混用

**解决方案**:

**1. 默认参数调整**（run_bone_arc.py）:
- `--step`: 0.002 → **0.0005** (步长减小 4 倍)
- `--tolerance`: 30.0 → **1.5** (容差倍数降低 20 倍)

**2. 硬编码修复**（run_bone_arc.py resolve_ply_path）:
- 修复前：`local_path = os.path.join(script_dir, "Bone_m_later.ply")`
- 修复后：`local_path = os.path.join(script_dir, ply_name)`

**3. Polyscope 名称规范**（arc_trajectory.py）:
| 旧名称 | 新名称 |
|-------|-------|
| Bone Point Cloud | **Bone_Cloud** |
| Arc Trajectory | **Trajectory** |
| Trajectory Frames | **Trajectory_Frames** |

向量数量名称由 `plot_orientation_field` 自动生成（如 `x_Bone_Cloud`、`y_Bone_Cloud`、`z_Bone_Cloud`），符合 `x_tangent` / `y_bitangent` / `z_normal` 的语义映射。

**高精度运行命令**:
```powershell
python run_bone_arc.py --interactive --init-steps 0 --step 0.0005 --tolerance 1.0 --use-o3d
```

**修改文件**:
- `run_bone_arc.py` - 修改 `resolve_ply_path` 函数和默认参数
- `arc_trajectory.py` - 规范 Polyscope 注册名称

---

## 项目文件结构

```
bone_cuting/
├── arc_trajectory.py      # 弧面轨迹生成器（核心类）
├── point_selector.py      # 点选择器（交互式/自动/索引选点）
├── run_bone_arc.py        # 运行脚本（命令行接口）
├── font_utils.py          # 字体工具（解决中文乱码）
├── Bone_m_later.ply       # 骨骼点云示例数据
├── .polyscope.ini         # Polyscope 配置
└── imgui.ini              # ImGui 配置
```

---

## 使用方法

### 自动选点模式
```bash
python run_bone_arc.py --auto --axis 0
```

### 交互式选点模式
```bash
python run_bone_arc.py --interactive
```

### 指定起终点索引
```bash
python run_bone_arc.py --start 100 --end 800
```

### 自定义参数
```bash
python run_bone_arc.py --auto --diffusion 3000 --step 0.0015 --offset -0.001
```

---

## 核心参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--diffusion` | 2000 | 扩散标量，越大越平滑，骨骼有孔洞用 3000-5000 |
| `--step` | 0.0005 | 步长（米），0.5mm，更小步长获得更密集路点（30~50 个） |
| `--offset` | -0.001 | 距表面偏移（米），负值=表面外侧 |
| `--axis` | 0 | 自动选点的轴方向（0=X, 1=Y, 2=Z） |
| `--tolerance` | 1.5 | 终点到达容差倍数（相对步长），越小越精准 |

---

## 输出数据

轨迹生成后，可通过以下方法获取机械臂控制数据：
- `get_trajectory_poses()` - 位置 + 3x3 旋转矩阵
- `get_trajectory_quaternions()` - 位置 + 四元数
- `get_trajectory_euler()` - 位置 + 欧拉角
- `save_results()` - 保存到 pickle 文件

旋转矩阵列向量含义：
- 第 0 列 = X 轴 = 沿轨迹前进方向（切向）
- 第 1 列 = Y 轴 = 侧向（表面内垂直于轨迹）
- 第 2 列 = Z 轴 = 法向（表面法线方向）

---

## 注意事项

1. **中文乱码**: Polyscope 窗口中的名称已改为英文，确保无乱码
2. **点云格式**: 支持 `.ply` 格式，建议使用 Open3D 加载检查点云质量
3. **尺度单位**: 程序会自动检测并转换毫米为米制单位
4. **扩散参数**: 骨骼有孔洞时需增大 `--diffusion` 参数（3000-5000）
