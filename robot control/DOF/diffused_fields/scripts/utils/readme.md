# Diffused Fields - Pointcloud Transformations & Feature Detection
# 几何扩散场 - 点云形变与特征检测验证

This subdirectory contains robust validation suites designed to test non-linear geometry processing utilities, data augmentation pipelines, and automated geodesic feature extraction within the `diffused_fields` framework.

本目录包含了一系列高鲁棒性的验证程序，旨在测试 `diffused_fields` 框架内的非线性几何处理工具、数据增强流水线以及自动化测地线特征提取算法。

---

## Scripts Architecture & Core Logic / 脚本架构与核心逻辑

### 1. Batch Automated Endpoint Detection (`test_endpoint_detection.py`)
* **English**: Validates the topological invariance and robustness of the **Two-Stage Diffusion Endpoint Detector**. The script synthesizes a batch of 50 highly deformed specimens (subjected to stochastic scaling, bending, and twisting). It runs an intrinsic diffusion-based solver to automatically extract the two geometric poles (endpoints) of the underlying manifold. Results are programmatically arranged in a 3D grid layout ($10 \times 5$) for parallel empirical evaluation.
* **中文**: 验证**双阶段扩散端点检测器**的拓扑不变性与鲁棒性。该脚本随机合成了 50 个经受高度形变（涵盖随机缩放、弯曲、扭曲）的样本。它运行一个基于本征扩散的求解器，自动提取流形表面的两个几何极点（端点）。为了便于并行经验评估，计算结果在 3D 空间中以网格阵列（$10 \times 5$）的形式自动平移排列。

### 2. Pointcloud Transformation Operators (`test_pointcloud_methods.py`)
* **English**: A comprehensive validation suite dedicated to testing geometric and topological mutation operators implemented within the `Pointcloud` class. It evaluates individual and combined effects of:
  * Sensor artifact emulation: Gaussian noise insertion and anisotropic scaling.
  * Topological occlusion simulation: Stochastic hole creation via local neighborhood pruning.
  * Non-linear physical deformation: Bending, twisting, and volumetric bulging.
* **中文**: 专用于测试 `Pointcloud` 类中实现的几何与拓扑突变算子的综合验证套件。它评估了以下独立及组合算子的效果：
  * 传感器误差模拟：加入高斯噪声和各向异性缩放。
  * 拓扑遮挡模拟：通过局部邻域修剪随机制造空洞。
  * 非线性物理形变：弯曲、扭曲和体积鼓包。

### 3. Multimodal Attribute Inspector (`visualize_pointcloud.py`)
* **English**: A general-purpose diagnostic tool for pointcloud visualization. It adaptively parses raw mesh metadata, resolving RGB properties if present. For uncolored/monochromatic scans, it automatically projects an alternate $z$-axis height field scalar representation and renders embedded vector fields such as surface normals.
* **中文**: 通用点云数据诊断与检查工具。它自适应地解析原始网格元数据，若存在 RGB 颜色则直接渲染。对于无颜色或单色的扫描数据，它会自动投影基于 $z$ 轴的高度场标量表示，并支持直接渲染表面法向量等内置向量场。

---

## Technical Concept: Two-Stage Diffusion / 技术概念：双阶段扩散

The automated feature extraction pipeline avoids brittle curvature heuristics by utilizing intrinsic diffusion operations:

自动化特征提取流水线通过利用本征扩散操作，避免了脆弱的曲率启发式算法：


$$\Delta \phi(x) = 0, \quad x \in \mathcal{M}$$

1. **Stage 1 (Global Anchoring)**: Diffuses a harmonic field from the barycenter of the mesh to isolate the most geometrically remote boundary pole ($\text{Endpoint}_1$).
2. **Stage 2 (Geodesic Conjugation)**: Uses $\text{Endpoint}_1$ as the new Dirac delta source condition. The subsequent diffusion maximum robustly defines the conjugate pole ($\text{Endpoint}_2$) along the manifold's longest intrinsic path.

1. **第一阶段（全局锚定）**: 从网格的重心扩散调和场，以分离出几何结构上最遥远的边界极点（$\text{端点}_1$）。
2. **第二阶段（测地共轭）**: 将 $\text{端点}_1$ 作为新的狄拉克 $\delta$ 函数源条件。随后的扩散极大值点将稳健地定义沿着流形最长本征路径的共轭极点（$\text{端点}_2$）。

---

## Pipeline Execution / 运行方式

Execute the testing procedures using your local Python interpreter:
请使用本地 Python 解释器运行以下测试程序：

```bash
# Run batch endpoint detection benchmark (50 deformed shapes)
python test_endpoint_detection.py

# Verify all geometric transformation and data augmentation operators
python test_pointcloud_methods.py

# Launch the pointcloud diagnostic inspector
python visualize_pointcloud.py