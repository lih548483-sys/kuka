"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
扩散场机器人动作原语模块

这个模块实现了基于扩散场的机器人动作原语（Action Primitives）系统。
通过在点云表示的物体表面上执行预定义的运动模式，实现各种操作任务。

核心概念：
1. 动作原语（Action Primitive）：预定义的运动模式，如切割、切片、覆盖、去皮等
2. 扩散场（Diffusion Field）：在点云上计算的标量/旋转场，用于引导运动方向
3. Walk-on-Spheres 算法：用于在点云表面进行扩散计算的高效方法
4. 局部坐标系（Local Basis）：在每个位置定义的局部参考框架

支持的原语类型：
- Cutting（切割）：沿一个方向直线切割
- Slicing（切片）：反复切割形成薄片
- Coverage（覆盖）：螺旋向内移动覆盖整个表面
- Peeling（去皮）：分层去除材料

设计模式：
- 使用基类继承，子类实现特定原语的具体行为
- 配置驱动的参数管理
- 支持结果的保存和加载
"""

# 导入操作系统模块
import os
# 导入 pickle 用于序列化和反序列化 Python 对象
import pickle
# 导入日期时间模块用于生成时间戳
from datetime import datetime

# 导入 NumPy 用于数值计算
import numpy as np
# 导入 YAML 用于解析配置文件
import yaml
# 从 diffused_fields 包导入扩散相关类
from diffused_fields import PointcloudScalarDiffusion, WalkOnSpheresDiffusion
# 从可视化模块导入所有绘图函数
from diffused_fields.visualization.plotting_ps import *


class pcloudActionPrimitives(object):
    """
    动作原语基类。

    所有具体动作原语类都继承自这个基类。
    提供了通用的初始化流程、轨迹管理、扩散系统设置和结果保存功能。

    子类需要实现的方法：
    - _setup_primitive_specific_attributes: 设置原语特定的属性
    - _post_initialization_setup: 扩散系统初始化后的设置
    - run: 执行原语的核心逻辑

    属性：
    - pcloud: 点云对象
    - primitive_type: 原语类型名称
    - step_size: 运动步长
    - trajectory: 计算出的轨迹点
    - trajectory_local_bases: 轨迹点的局部坐标系
    - scalar_diffusion: 标量扩散系统
    - wos: Walk-on-Spheres 扩散系统
    """

    def __init__(self, pcloud, primitive_type, **kwargs):
        """
        初始化动作原语基类。

        Args:
            pcloud: 点云对象，表示要操作的物体表面
            primitive_type: 原语类型字符串（如 "cutting", "slicing"）
            **kwargs: 可选参数
                - start_vertex: 起始顶点索引
                - end_vertex: 终止顶点索引
                - source_vertices: 源顶点列表
                - diffusion_scalar: 扩散标量参数
                - 其他参数由具体子类定义
        """
        # 存储点云引用
        self.pcloud = pcloud
        # 存储原语类型
        self.primitive_type = primitive_type

        # 从配置文件加载参数
        self.load_parameters()

        # 如果配置文件没有设置 step_size，使用默认值
        if not hasattr(self, "step_size"):
            self.step_size = 0.001  # 默认步长 1mm

        # 从 kwargs 设置通用属性（仅当提供时才设置）
        if "start_vertex" in kwargs:
            self.start_vertex = kwargs["start_vertex"]
        if "end_vertex" in kwargs:
            self.end_vertex = kwargs["end_vertex"]
        if "source_vertices" in kwargs:
            self.source_vertices = kwargs["source_vertices"]

        # 获取扩散标量，默认为 None
        diffusion_scalar = kwargs.get("diffusion_scalar", None)

        # 允许子类设置额外的属性
        self._setup_primitive_specific_attributes(**kwargs)

        # 初始化扩散系统
        self._initialize_diffusion_systems(diffusion_scalar)

        # 后期初始化设置（在扩散系统准备好之后）
        self._post_initialization_setup()

        # 初始化轨迹
        self.init_trajectory()

    def _setup_primitive_specific_attributes(self, **kwargs):
        """
        在子类中重写，设置原语特定的属性。

        Args:
            **kwargs: 可选参数
        """
        pass

    def _post_initialization_setup(self):
        """
        在子类中重写，用于需要扩散系统已初始化的设置。

        例如 Cutting 类在这里设置 end_point。
        """
        pass

    def _initialize_diffusion_systems(self, diffusion_scalar):
        """
        初始化标量扩散和 Walk-on-Spheres 扩散系统。

        标量扩散用于计算物体表面的热传导/扩散场。
        Walk-on-Spheres 用于在自由空间中计算旋转场。

        Args:
            diffusion_scalar: 扩散标量参数，如果为 None 则使用配置中的值
        """
        # 如果没有提供扩散标量，使用配置中的值
        if diffusion_scalar is None:
            diffusion_scalar = self.diffusion_scalar

        # 创建标量扩散系统
        self.scalar_diffusion = PointcloudScalarDiffusion(
            self.pcloud, diffusion_scalar=diffusion_scalar
        )

        # 设置源顶点
        if hasattr(self, "source_vertices") and self.source_vertices is not None:
            if type(self.source_vertices) == str:
                # 如果是字符串 "endpoints"，使用端点作为源
                self.scalar_diffusion.get_endpoints()
                self.scalar_diffusion.source_vertices = self.scalar_diffusion.endpoints
            else:
                # 直接使用提供的源顶点
                self.scalar_diffusion.source_vertices = self.source_vertices
        else:
            # 使用配置文件中的默认 source_vertices
            self.scalar_diffusion.source_vertices = self.source_vertices

        # 计算局部坐标系
        self.scalar_diffusion.get_local_bases()

        # 设置 Walk-on-Spheres 扩散用于环境空间
        # 边界是点云表面
        boundaries = [self.pcloud]
        self.wos = WalkOnSpheresDiffusion(
            boundaries=boundaries,
            # 收敛阈值基于平均边长的两倍
            convergence_threshold=self.pcloud.get_mean_edge_length() * 2,
        )

    def init_trajectory(self):
        """
        初始化轨迹，包含可选的安全偏移。

        安全偏移在初始顶点沿局部 x 方向（纵向/径向）移动 num_init_steps 步，
        然后从这个安全位置开始主轨迹。这有助于避免边界/源顶点问题。

        初始化过程：
        1. 获取初始点
        2. 如果设置了 num_init_steps，沿 x 方向偏移安全距离
        3. 添加初始点到轨迹
        4. 计算初始点的局部坐标系
        """
        # 初始化轨迹存储
        self.trajectory_local_bases = []
        self.x_arr = []
        self.actions = []

        # 获取起始点
        x0 = self._get_initial_point()

        # 获取初始化步数（默认为 0）
        num_init_steps = getattr(self, "num_init_steps", 0)

        if num_init_steps > 0:
            # 沿 x 方向（纵向/径向）移动 num_init_steps 步以确保安全
            # 这个移动不添加到轨迹，只是找到安全的起始位置
            print(f"Moving {num_init_steps} steps from starting vertex for safety...")
            x_current = x0
            for _ in range(num_init_steps):
                x_current, local_basis = self.local_step(
                    x_current, direction=0, sign=1  # x 方向（纵向/径向）
                )
                # 修正到距离表面正确的位置
                x_current, _, _ = self.pcloud.correct_distance_smooth(
                    x_current, self.distance_to_surface
                )

            # 使用偏移后的位置作为实际起始点
            x0 = x_current
            print(f"Safety offset complete. Starting from interior position.")

        # 设置起始点（可能是原始的或经过安全偏移的）
        self.x_arr.append(x0)
        # 从点获取批处理点用于扩散
        batch_points = self.wos.get_batch_from_point(x0)
        # 使用扩散计算局部坐标系
        local_basis, _, _ = self.wos.diffuse_rotations(batch_points)
        self.trajectory_local_bases.append(local_basis)

    def _get_initial_point(self):
        """
        获取轨迹的初始点。

        如果设置了 start_vertex，从该顶点开始。
        否则使用 source_vertices 中的第一个顶点。

        Returns:
            np.ndarray: 初始点位置（在表面上偏移 distance_to_surface）
        """
        if hasattr(self, "start_vertex"):
            # 从 start_vertex 顶点在法线方向偏移
            return (
                self.pcloud.vertices[self.start_vertex]
                + self.pcloud.normals[self.start_vertex] * self.distance_to_surface
            )
        else:
            # 从 source_vertices[0] 顶点在法线方向偏移
            return (
                self.pcloud.vertices[self.source_vertices[0]]
                + self.pcloud.normals[self.source_vertices[0]]
                * self.distance_to_surface
            )

    def load_parameters(self):
        """
        使用新的层次化配置系统加载参数。

        配置来源：
        1. 全局默认值
        2. 原语特定配置
        3. 对象特定覆盖
        """
        from ..core.config import get_action_primitive_config

        # 加载合并后的配置
        merged_config = get_action_primitive_config(
            self.primitive_type, self.pcloud.object_name
        )

        # 将所有参数设置为对象的属性
        self._set_parameters(merged_config)

    def _set_parameters(self, parameters: dict):
        """
        递归地将参数字典设置为对象属性。

        支持嵌套字典，嵌套的字典会被转换为子对象。

        Args:
            parameters: 参数字典
        """
        def set_attributes(obj, dictionary):
            """内部函数，递归设置属性"""
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    # 创建一个新的子对象（或使用已存在的）
                    # 这创建了一个简单的类来存储嵌套参数
                    sub_obj = getattr(obj, key, type("SubParams", (), {})())
                    # 递归设置子属性
                    set_attributes(sub_obj, value)
                    setattr(obj, key, sub_obj)
                else:
                    # 直接设置属性值
                    setattr(obj, key, value)

        set_attributes(self, parameters)

    def local_step(self, x, direction, sign):
        """
        在给定位置执行单步局部移动。

        使用 Walk-on-Spheres 扩散计算局部坐标系，
        然后沿指定方向移动一步。

        Args:
            x: 当前位置
            direction: 移动方向（0=x/纵向, 1=y/切向, 2=z/法向）
            sign: 移动符号（+1 或 -1）

        Returns:
            tuple: (下一位置, 局部坐标系)
        """
        # 从当前点获取批处理点用于扩散
        batch_points = self.wos.get_batch_from_point(x)

        # 使用扩散计算局部坐标系
        local_basis, _, _ = self.wos.diffuse_rotations(batch_points)

        # 计算下一位置：沿指定方向移动 step_size * sign
        next_x = x + (local_basis[:, direction] * self.step_size * sign)
        return next_x, local_basis

    def check_endpoint_reached(self, x_next, local_basis=None):
        """
        检查是否到达终点。

        考虑表面距离的情况下，判断当前位置是否在终点的容差范围内。

        Args:
            x_next: 下一位置
            local_basis: 局部坐标系（可选）

        Returns:
            bool: 是否到达终点
        """
        # 如果没有设置 end_point，永远返回 False
        if not hasattr(self, "end_point"):
            return False

        # 获取容差乘数
        tolerance = getattr(self, "endpoint_tolerance_multiplier", 30.0)
        # 计算到终点的距离
        dist2end = np.linalg.norm(x_next - self.end_point)

        # 到达条件：距离减去表面距离的绝对值小于容差
        # 这确保我们在表面上而不是在表面内部或外部
        reached = (
            np.abs(dist2end - np.abs(self.distance_to_surface))
            < self.step_size * tolerance
        )

        if reached:
            self.reached_end_point = True

        return reached

    def move_multistep(
        self,
        num_steps,
        x0,
        direction,
        sign,
        project=False,
        distance_to_surface=0.0,
        terminal_condition=None,
    ):
        """
        执行多步移动。

        循环执行 local_step，可以选择性地将点投影到表面。

        Args:
            num_steps: 移动步数
            x0: 起始位置
            direction: 移动方向
            sign: 移动符号
            project: 是否投影到表面
            distance_to_surface: 投影时距离表面的距离
            terminal_condition: 终止条件回调函数
        """
        x_next = x0
        for _ in range(num_steps):
            # 如果 direction 是列表，同时处理多个方向
            if type(direction) == list:
                for dir, sgn in zip(direction, sign):
                    x_next, local_basis = self.local_step(x_next, dir, sgn)
                    self.x_arr.append(x_next)
                    self.trajectory_local_bases.append(local_basis)
                    print(f"Step {_ + 1} of {num_steps}: {x_next}")
            else:
                x_next, local_basis = self.local_step(x_next, direction, sign)

            # 如果需要，投影到表面
            if project:
                x_next, _, _ = self.pcloud.correct_distance_smooth(
                    x_next, distance_to_surface
                )

            # 检查终止条件
            if terminal_condition is not None:
                if terminal_condition(x_next, local_basis):
                    break

            # 添加到轨迹
            self.x_arr.append(x_next)
            self.trajectory_local_bases.append(local_basis)

    def visualize_trajectory(self, show_tool=False, num_samples=None):
        """
        可视化轨迹和点云。

        Args:
            show_tool: 是否显示工具可视化
            num_samples: 轨迹采样点数
        """
        # 保存可视化参数以供后续使用
        self.visualization_num_samples = num_samples

        # 初始化 PyScene 绘图
        ps.init()

        # 轨迹曲线
        ps.register_curve_network(
            f"Trajectory",
            self.trajectory,
            edges="line",  # 使用线条作为边
            radius=curve_radius,  # 曲线半径
            transparency=1.0,  # 不透明
            color=[0, 0, 0],  # 黑色
        )

        # 点云可视化
        vector_length = 0.05 / 2  # 向量长度
        vector_radius = 0.035 / 2  # 向量半径
        point_radius = 0.003  # 点半径
        curve_radius = 0.003  # 曲线半径

        ps_cloud = plot_orientation_field(
            self.pcloud.vertices,
            self.pcloud.local_bases,
            name="object point cloud",
            vector_length=vector_length,
            vector_radius=vector_radius,
            point_radius=point_radius,
            enable_vector=True,
            enable_z=True,  # 显示法线
            color=self.pcloud.colors,
        )

        # 手动添加颜色，因为 plot_orientation_field 可能忽略它们
        if hasattr(self.pcloud, "colors") and self.pcloud.colors is not None:
            ps_cloud.add_color_quantity("colors", self.pcloud.colors, enabled=True)
        ps_cloud.add_vector_quantity("z", self.pcloud.normals)

        # 轨迹坐标系
        plot_orientation_field(
            self.trajectory,
            self.trajectory_local_bases,
            name="trajectory frames",
            vector_length=vector_length,
            vector_radius=vector_radius,
            point_radius=point_radius,
            enable_x=False,
        )

        # 关键点（源顶点）
        if len(self.source_vertices) == 1:
            # 单个关键点
            ps_keypoints = plot_orientation_field(
                self.pcloud.vertices[self.source_vertices],
                name="keypoints",
                vector_length=vector_length,
                vector_radius=vector_radius,
            )
            # 手动添加蓝色
            blue_color = np.array([[0, 0, 1]])
            ps_keypoints.add_color_quantity("colors", blue_color, enabled=True)
        else:
            # 多个关键点
            ps_keypoints = plot_orientation_field(
                self.pcloud.vertices[self.source_vertices],
                name="keypoints",
                point_radius=0.0326,
                vector_length=vector_length,
                vector_radius=vector_radius,
            )
            # 手动添加蓝色
            blue_colors = np.tile([0, 0, 1], (len(self.source_vertices), 1))
            ps_keypoints.add_color_quantity("colors", blue_colors, enabled=True)

        # 工具可视化
        if show_tool:
            tool_mesh = import_tool_mesh(self.tool)
            if num_samples == None:
                indices = np.linspace(
                    0, len(self.trajectory) - 1, len(self.trajectory) - 1, dtype=int
                )
            else:
                # 创建均匀分布的索引，但排除首尾
                all_indices = np.linspace(
                    0, len(self.trajectory) - 1, num_samples + 2, dtype=int
                )
                indices = all_indices[1:-1]
            downsampled_trajectory = self.trajectory[indices]
            downsampled_trajectory_local_bases = self.trajectory_local_bases[indices]

            if num_samples is None:
                animate_tool_trajectory(
                    downsampled_trajectory,
                    -downsampled_trajectory_local_bases,
                    tool_mesh,
                )
            else:
                plot_orientation_field(
                    downsampled_trajectory,
                    -downsampled_trajectory_local_bases,
                    name="local frames",
                    vector_length=vector_length,
                    vector_radius=vector_radius,
                    point_radius=point_radius,
                    enable_vector=True,
                )

                plot_tool_trajectory(
                    downsampled_trajectory,
                    -downsampled_trajectory_local_bases,
                    tool_mesh,
                )

        ps.show()

    def save_results(self, filepath=None, include_pointcloud=True):
        """
        将实验结果保存到 pickle 文件。

        保存的数据包括：
        - 轨迹点和局部坐标系
        - 使用的参数
        - 时间戳
        - 可选：点云数据

        Args:
            filepath: 保存路径，如果为 None 则自动生成
            include_pointcloud: 是否包含完整点云数据

        Returns:
            str: 保存的文件路径
        """
        if not hasattr(self, "trajectory") or self.trajectory is None:
            raise RuntimeError("No trajectory data found. Run the experiment first.")

        # 如果没有提供路径，自动生成
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{self.primitive_type}_{self.pcloud.object_name}_{timestamp}.pkl"
            )
            filepath = os.path.join("results", filename)

        # 创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 准备保存的数据
        results_data = {
            "primitive_type": self.primitive_type,
            "object_name": self.pcloud.object_name,
            "trajectory": self.trajectory,
            "trajectory_local_bases": self.trajectory_local_bases,
            "source_vertices": getattr(self, "source_vertices", None),
            "parameters": self._get_experiment_parameters(),
            "timestamp": datetime.now().isoformat(),
        }

        # 添加原语特定数据
        if hasattr(self, "end_point"):
            results_data["end_point"] = self.end_point
        if hasattr(self, "reached_end_point"):
            results_data["reached_end_point"] = self.reached_end_point

        # 可选：包含点云数据
        if include_pointcloud:
            results_data["pointcloud"] = {
                "vertices": self.pcloud.vertices,
                "normals": self.pcloud.normals,
                "faces": getattr(self.pcloud, "faces", None),
                "colors": getattr(self.pcloud, "colors", None),
                "local_bases": getattr(self.pcloud, "local_bases", None),
            }
        else:
            results_data["pointcloud_filename"] = self.pcloud.filename

        # 保存到 pickle 文件
        with open(filepath, "wb") as f:
            pickle.dump(results_data, f)

        print(f"Results saved to: {filepath}")
        return filepath

    @classmethod
    def load_results(cls, filepath):
        """
        从 pickle 文件加载实验结果。

        Args:
            filepath: 结果文件路径

        Returns:
            dict: 加载的实验数据
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results file not found: {filepath}")

        with open(filepath, "rb") as f:
            results_data = pickle.load(f)

        print(f"Results loaded from: {filepath}")
        print(
            f"Experiment: {results_data['primitive_type']} on {results_data['object_name']}"
        )
        print(f"Trajectory points: {len(results_data['trajectory'])}")

        return results_data

    def _convert_to_dict(self, obj):
        """
        将对象（包括动态 SubParams）转换为字典，用于 pickle 序列化。

        Args:
            obj: 要转换的对象

        Returns:
            dict 或原始对象（如果不可转换）
        """
        if hasattr(obj, "__dict__"):
            result = {}
            for key, value in obj.__dict__.items():
                result[key] = self._convert_to_dict(value)  # 递归转换
            return result
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_dict(item) for item in obj]
        else:
            return obj

    def _get_experiment_parameters(self):
        """
        提取实验参数用于保存。

        Returns:
            dict: 实验参数字典
        """
        params = {}

        # 通用参数列表
        param_names = [
            "diffusion_scalar",
            "step_size",
            "distance_to_surface",
            "num_init_steps",
            "num_slices",
            "num_slicing_steps",
            "num_slide_steps",
            "num_peels",
            "num_peeling_steps",
            "num_loops",
            "num_tangential_steps",
            "num_radial_steps",
            "num_cut_steps",
            "tool",
            "visualization_num_samples",
        ]

        for param in param_names:
            if hasattr(self, param):
                param_value = getattr(self, param)
                params[param] = self._convert_to_dict(param_value)

        return params

    @staticmethod
    def visualize_from_results(results_data, show_tool=False, num_samples=None):
        """
        可视化从文件加载的实验结果。

        Args:
            results_data: 从 load_results() 加载的数据
            show_tool: 是否显示工具可视化
            num_samples: 轨迹采样点数
        """
        # 使用存储的 num_samples（如果没有提供）
        if num_samples is None and "parameters" in results_data:
            num_samples = results_data["parameters"].get(
                "visualization_num_samples", None
            )
        if "pointcloud" not in results_data:
            raise ValueError(
                "Pointcloud data not found in results. Re-save with include_pointcloud=True"
            )

        trajectory = results_data["trajectory"]
        trajectory_local_bases = results_data["trajectory_local_bases"]
        pcloud_data = results_data["pointcloud"]

        # 可视化参数
        vector_length = 0.05 / 2
        vector_radius = 0.035 / 2
        point_radius = 0.003
        curve_radius = 0.003

        ps.init()

        # 轨迹曲线
        ps.register_curve_network(
            f"Trajectory",
            trajectory,
            edges="line",
            radius=curve_radius,
            transparency=1.0,
            color=[0, 0, 0],
        )

        # 点云
        ps_cloud = plot_orientation_field(
            pcloud_data["vertices"],
            pcloud_data["local_bases"],
            name="object point cloud",
            vector_length=vector_length,
            vector_radius=vector_radius,
            point_radius=point_radius,
            enable_vector=False,
            enable_x=False,
            color=pcloud_data["colors"],
        )

        # 添加颜色和法线
        if pcloud_data["colors"] is not None:
            ps_cloud.add_color_quantity("colors", pcloud_data["colors"], enabled=True)
        if pcloud_data["normals"] is not None:
            ps_cloud.add_vector_quantity("z", pcloud_data["normals"])

        # 轨迹坐标系
        plot_orientation_field(
            trajectory,
            trajectory_local_bases,
            name="trajectory frames",
            vector_length=vector_length,
            vector_radius=vector_radius,
            point_radius=point_radius,
            enable_vector=False,
            enable_x=True,
        )

        # 源点
        if (
            "source_vertices" in results_data
            and results_data["source_vertices"] is not None
        ):
            source_vertices = results_data["source_vertices"]
            if len(source_vertices) == 1:
                plot_orientation_field(
                    pcloud_data["vertices"][source_vertices],
                    name="source",
                    vector_length=vector_length,
                    vector_radius=vector_radius,
                )
            else:
                plot_orientation_field(
                    pcloud_data["vertices"][source_vertices],
                    name="source",
                    point_radius=0.0326,
                    vector_length=vector_length,
                    vector_radius=vector_radius,
                )

        # 工具可视化（如果请求）
        if show_tool:
            tool_data = None
            if "parameters" in results_data and "tool" in results_data["parameters"]:
                tool_data = results_data["parameters"]["tool"]
            elif "tool" in results_data:
                tool_data = results_data["tool"]

            if tool_data is None:
                print(
                    "Warning: Tool information not found in results. Re-save results to include tool data."
                )
            else:
                try:
                    # 如果需要，将字典转换回对象
                    if isinstance(tool_data, dict):

                        class ToolData:
                            def __init__(self, data_dict):
                                for key, value in data_dict.items():
                                    setattr(self, key, value)

                        tool_obj = ToolData(tool_data)
                    else:
                        tool_obj = tool_data

                    tool_mesh = import_tool_mesh(tool_obj)

                    if num_samples is None:
                        indices = np.linspace(
                            0, len(trajectory) - 1, len(trajectory) - 1, dtype=int
                        )
                    else:
                        indices = np.linspace(
                            0, len(trajectory) - 1, num_samples, dtype=int
                        )
                    downsampled_trajectory = trajectory[indices]
                    downsampled_trajectory_local_bases = trajectory_local_bases[indices]

                    if num_samples is None:
                        animate_tool_trajectory(
                            downsampled_trajectory,
                            -downsampled_trajectory_local_bases,
                            tool_mesh,
                        )
                    else:
                        plot_orientation_field(
                            downsampled_trajectory,
                            -downsampled_trajectory_local_bases,
                            name="local frames",
                            vector_length=vector_length,
                            vector_radius=vector_radius,
                            point_radius=point_radius,
                            enable_vector=True,
                        )

                        plot_tool_trajectory(
                            downsampled_trajectory,
                            -downsampled_trajectory_local_bases,
                            tool_mesh,
                        )
                except Exception as e:
                    print(f"Warning: Could not load or visualize tool: {e}")

        ps.show()


class Cutting(pcloudActionPrimitives):
    """
    切割原语。

    沿纵向（x）方向从起点移动到终点。
    使用 endpoint_tolerance_multiplier 控制到达终点的判定精度。
    """

    def __init__(self, pcloud, primitive_type="cutting", **kwargs):
        """
        初始化切割原语。

        Args:
            pcloud: 点云对象
            primitive_type: 原语类型（默认为 "cutting"）
            **kwargs: 其他参数
        """
        super().__init__(pcloud, primitive_type, **kwargs)

    def _post_initialization_setup(self):
        """
        在扩散系统初始化后设置切割特定的属性。

        从 source_vertices[1] 获取终点位置。
        """
        end_vertex = self.scalar_diffusion.source_vertices[1]
        self.end_point = self.pcloud.vertices[end_vertex]
        self.endpoint_tolerance_multiplier = 20.0

    def run(self):
        """
        执行切割运动。

        沿方向 0（纵向）从当前位置移动到终点。
        """
        self.move_multistep(
            self.num_cut_steps,
            self.x_arr[-1],
            direction=0,  # 纵向方向
            sign=1,
            project=True,  # 投影到表面
            terminal_condition=self.check_endpoint_reached,  # 到达终点时停止
        )

        # 将轨迹转换为 NumPy 数组
        self.trajectory = np.array(self.x_arr)
        self.trajectory_local_bases = np.array(self.trajectory_local_bases)


class Slicing(pcloudActionPrimitives):
    """
    切片原语。

    反复执行切下和退回的动作，形成一系列平行的薄片。
    类似于用刀切面包片的动作。
    """

    def __init__(self, pcloud, primitive_type="slicing", **kwargs):
        """
        初始化切片原语。

        Args:
            pcloud: 点云对象
            primitive_type: 原语类型（默认为 "slicing"）
            **kwargs: 其他参数
        """
        super().__init__(pcloud, primitive_type, **kwargs)

        # 如果没有设置 num_slices，使用默认值 30
        if not hasattr(self, "num_slices"):
            self.num_slices = 30

    def _post_initialization_setup(self):
        """
        设置切片特定的属性。
        """
        end_vertex = self.scalar_diffusion.source_vertices[1]
        self.end_point = self.pcloud.vertices[end_vertex]
        self.reached_end_point = False
        self.endpoint_tolerance_multiplier = 20.0  # 较大的值意味着较早终止

    def run(self):
        """
        执行切片运动。

        每个切片的动作序列：
        1. 向下移动（direction=2, sign=1）进行切割
        2. 向上退回（direction=2, sign=-1）
        3. 沿纵向滑动（direction=0）到下一个位置
        4. 检查是否到达终点，如果是则停止
        """
        for slice_idx in range(self.num_slices):
            print(f"Performing slice {slice_idx + 1} of {self.num_slices}")

            # 步骤1：向下移动进行切割
            self.move_multistep(
                self.num_slicing_steps,
                self.x_arr[-1],
                direction=2,  # 向内部
                sign=1,
                project=False,  # 不投影
            )

            # 步骤2：向上退回
            self.move_multistep(
                self.num_slicing_steps,
                self.x_arr[-1],
                direction=2,  # 向内部
                sign=-1,
                project=False,
            )

            # 步骤3：滑动到下一个切片位置
            self.move_multistep(
                self.num_slide_steps,
                self.x_arr[-1],
                direction=0,  # 纵向轴
                sign=1,
                project=True,  # 投影到表面
                distance_to_surface=self.distance_to_surface,
                terminal_condition=self.check_endpoint_reached,
            )

            # 检查是否在滑动过程中到达终点
            if self.reached_end_point:
                print(
                    f"Endpoint reached after upward movement in slice {slice_idx + 1}. Stopping slicing."
                )
                print(f"Slicing completed after {slice_idx + 1} slices")
                break

        self.trajectory = np.array(self.x_arr)
        self.trajectory_local_bases = np.array(self.trajectory_local_bases)


class Coverage(pcloudActionPrimitives):
    """
    覆盖原语。

    从边界开始，螺旋向内移动覆盖整个表面。
    类似于给圆形表面涂漆或抛光。

    运动模式：
    1. 沿切向（y）方向绕圈
    2. 向内移动一步（径向，x 方向）
    3. 重复，直到覆盖完整个表面
    """

    def __init__(self, pcloud, primitive_type="coverage", **kwargs):
        """
        初始化覆盖原语。

        Args:
            pcloud: 点云对象
            primitive_type: 原语类型（默认为 "coverage"）
            **kwargs: 其他参数
        """
        super().__init__(pcloud, primitive_type, **kwargs)

        # 设置默认的环路距离阈值
        if not hasattr(self, "loop_distance_threshold"):
            self.loop_distance_threshold = 0.01

    def _get_initial_point(self):
        """
        重写以支持边界检测。

        如果设置了 start_vertex，使用它。
        否则从边界点中随机选择一个作为起始点。
        """
        if hasattr(self, "start_vertex"):
            return (
                self.pcloud.vertices[self.start_vertex]
                + self.pcloud.normals[self.start_vertex] * self.distance_to_surface
            )
        else:
            # 获取边界点并选择其中一个
            self.pcloud.get_boundary()
            boundary_vertices = np.where(self.pcloud.is_boundary_arr)[0]
            # 使用第一个边界点
            start_vertex = boundary_vertices[0]
            return (
                self.pcloud.vertices[start_vertex]
                + self.pcloud.normals[start_vertex] * self.distance_to_surface
            )

    def visualize_trajectory(self, show_tool=False, num_samples=None):
        """
        重写以可视化所有边界点作为关键点。
        """
        # 获取边界点
        if not hasattr(self.pcloud, "is_boundary_arr"):
            self.pcloud.get_boundary()
        boundary_vertices = np.where(self.pcloud.is_boundary_arr)[0]

        # 临时将边界顶点存储为 source_vertices
        original_source_vertices = self.source_vertices
        self.source_vertices = boundary_vertices

        # 调用父类的可视化
        super().visualize_trajectory(show_tool=show_tool, num_samples=num_samples)

        # 恢复原始 source_vertices
        self.source_vertices = original_source_vertices

    def check_terminal_condition(self, x_next, local_basis):
        """
        使用欧几里得距离检查切向环路是否完成。

        当运动点返回到环路起点附近时，环路完成。
        使用距离变化的方向（增加还是减少）来判定。

        Returns:
            bool: 环路是否完成
        """
        # 初始化步数计数器
        if not hasattr(self, "_loop_step_count"):
            self._loop_step_count = 0
        self._loop_step_count += 1

        # 安全限制：防止无限循环
        if self._loop_step_count > 1000:
            print(
                f"Loop force-completed after {self._loop_step_count} steps (safety limit)"
            )
            self._loop_step_count = 0
            return True

        # 计算到环路原点的欧几里得距离
        euclidean_distance = np.linalg.norm(x_next - self.x_origin)

        # 检测距离开始减少（意味着我们正在返回）
        if euclidean_distance - self.euclidean_distance_prev < 0:
            self.triggered = True

        # 当距离小于阈值时，认为环路完成
        if self.triggered and euclidean_distance < self.loop_distance_threshold:
            print(f"Completed a loop (in {self._loop_step_count} steps)")
            self._loop_step_count = 0  # 为下一个环路重置
            return True

        self.euclidean_distance_prev = euclidean_distance
        return False

    def check_coverage_complete(self):
        """
        检查覆盖是否完成。

        通过测量当前环路的周长来判断。
        当环路变得非常小（接近零周长）时，说明我们已经到达中心，覆盖完成。

        使用多个标准确保鲁棒性：
        1. 绝对最小环路周长阈值
        2. 环路大小下降速率过慢（可能的无限循环）
        3. 最大环路数量限制
        """
        # 需要至少有一个完成的环路才能测量
        if not hasattr(self, "loop_path_lengths") or len(self.loop_path_lengths) < 1:
            return False

        current_loop_length = self.loop_path_lengths[-1]

        # 标准1：绝对最小环路周长阈值
        # 使用第一个环路的 5% 作为阈值，避免过早终止
        if len(self.loop_path_lengths) >= 2:
            first_loop_length = self.loop_path_lengths[0]
            min_loop_threshold = max(first_loop_length * 0.05, self.step_size * 5)

            if current_loop_length < min_loop_threshold:
                print(
                    f"Coverage complete: loop circumference too small ({current_loop_length:.6f} < {min_loop_threshold:.6f}, {current_loop_length/first_loop_length*100:.1f}% of first loop)"
                )
                return True

        # 标准2：检查环路大小是否下降太慢
        if len(self.loop_path_lengths) >= 5:
            recent_loops = self.loop_path_lengths[-4:]
            # 检查最近4个环路是否没有缩小（3%容差内）
            if all(
                abs(recent_loops[i] - recent_loops[i + 1]) / recent_loops[i] < 0.03
                for i in range(len(recent_loops) - 1)
            ):
                print(
                    f"Coverage complete: loop size not decreasing (last 4 loops: {[f'{l:.6f}' for l in recent_loops]})"
                )
                return True

        # 标准3：检查是否达到最大环路数
        max_loops = getattr(self, "num_loops", 20)
        if self.loop_count >= max_loops:
            print(f"Coverage complete: reached maximum number of loops ({max_loops})")
            return True

        return False

    def run(self):
        """
        执行覆盖运动。

        运动模式：
        1. 切向移动完成一个环路（绕圈）
        2. 径向向内移动一步
        3. 重复直到覆盖完成
        """
        sign_y = 1  # 切向方向符号

        # 跟踪环路路径长度
        self.loop_path_lengths = []
        self.loop_count = 0

        # 最大环路数
        max_loops = getattr(self, "num_loops", 30)

        for loop_idx in range(max_loops):
            self.loop_count = loop_idx + 1

            # 存储环路起点
            self.x_origin = self.x_arr[-1]
            loop_start_idx = len(self.x_arr) - 1

            # 获取参考局部坐标系用于角度跟踪
            batch_points = self.wos.get_batch_from_point(self.x_origin)
            reference_basis, _, _ = self.wos.diffuse_rotations(batch_points)

            # 存储参考方向
            self.reference_x = reference_basis[:, 0]  # 参考径向方向
            self.reference_z = reference_basis[:, 2]  # 参考法线方向

            # 初始化角度跟踪变量
            self.total_angular_change = 0.0
            self.prev_signed_angle = None

            # 初始化欧几里得距离跟踪
            self.euclidean_distance_prev = 0
            self.triggered = False

            # 步骤1：完成切向环路
            self.move_multistep(
                1000,  # 大数字确保完成，但会被终端条件中断
                self.x_arr[-1],
                direction=1,  # 切向方向
                sign=sign_y,  # 翻转方向
                distance_to_surface=self.distance_to_surface,
                project=True,
                terminal_condition=self.check_terminal_condition,
            )
            # 完成后翻转方向
            sign_y *= -1

            # 计算这个环路的路径长度
            loop_end_idx = len(self.x_arr) - 1
            loop_path_length = 0.0
            for i in range(loop_start_idx, loop_end_idx):
                loop_path_length += np.linalg.norm(self.x_arr[i + 1] - self.x_arr[i])
            self.loop_path_lengths.append(loop_path_length)

            print(f"Loop {self.loop_count}: circumference = {loop_path_length:.6f}")

            # 检查覆盖是否完成
            if self.check_coverage_complete():
                break

            # 步骤2：径向向内移动
            self.move_multistep(
                self.num_radial_steps,
                self.x_arr[-1],
                direction=0,  # 径向方向
                sign=1,
                distance_to_surface=self.distance_to_surface,
                project=True,
            )

        print(f"Coverage completed after {self.loop_count} loops")
        self.trajectory = np.array(self.x_arr)
        self.trajectory_local_bases = np.array(self.trajectory_local_bases)


class Peeling(pcloudActionPrimitives):
    """
    去皮原语。

    分层去除材料，类似于削苹果皮。
    每次去皮动作：
    1. 沿纵向切割
    2. 抬起工具
    3. 返回起点
    4. 侧向移动
    5. 放下工具
    6. 重复
    """

    def __init__(self, pcloud, primitive_type="peeling", **kwargs):
        """
        初始化去皮原语。

        Args:
            pcloud: 点云对象
            primitive_type: 原语类型（默认为 "peeling"）
            **kwargs: 其他参数
        """
        super().__init__(pcloud, primitive_type, **kwargs)

    def _post_initialization_setup(self):
        """
        设置去皮特定的属性。
        """
        self.end_point = self.pcloud.vertices[self.source_vertices[1]]
        self.force_list = []
        self.endpoint_tolerance_multiplier = 10.0

    def run(self):
        """
        执行去皮运动。
        """
        self.x_home = np.copy(self.x_arr[0])

        # 跟踪转换索引
        self.transition_indices = []

        for _ in range(self.num_peels):
            print(f"Performing peel {_ + 1} of {self.num_peels}")

            # 去皮动作：沿纵向移动
            self.move_multistep(
                500,  # 大数字确保完成
                self.x_arr[-1],
                direction=0,  # 纵向方向
                sign=1,
                project=True,
                distance_to_surface=self.distance_to_surface,
                terminal_condition=self.check_endpoint_reached,
            )
            self.transition_indices.append(len(self.x_arr) - 1)

            # 抬起工具（沿法线方向远离表面）
            lift_steps = int(-self.retract_distance_to_surface / self.step_size)
            self.move_multistep(
                lift_steps,
                self.x_arr[-1],
                direction=2,  # 法线方向
                sign=-1,  # 远离表面
            )

            self.transition_indices.append(len(self.x_arr) - 1)

            # 安全返回起点
            self.return_home_safe(distance_to_surface=self.retract_distance_to_surface)
            self.transition_indices.append(len(self.x_arr) - 1)
            print(f"Peeling period completed")

            # 侧向移动到下一个去皮位置
            self.move_multistep(
                self.num_slide_steps,
                self.x_arr[-1],
                direction=1,  # 切向方向
                sign=-1,
                project=True,
                distance_to_surface=self.retract_distance_to_surface,
            )
            self.transition_indices.append(len(self.x_arr) - 1)

            # 放下工具回到表面
            self.move_multistep(
                lift_steps,
                self.x_arr[-1],
                direction=2,  # 法线方向
                sign=1,  # 接近表面
            )
            self.transition_indices.append(len(self.x_arr) - 1)

            self.x_home = np.copy(self.x_arr[-1])

        self.trajectory = np.array(self.x_arr)
        self.trajectory_local_bases = np.array(self.trajectory_local_bases)

    def return_home_safe(self, distance_to_surface):
        """
        安全返回到起始点。

        使用测地线距离引导运动，确保沿表面移动。
        当测地线距离小于阈值时停止。
        """
        # 保存原始局部坐标系
        local_basis_real = np.copy(self.pcloud.local_bases)

        # 创建初始条件：在源顶点设置值为 1
        u0 = np.zeros(len(self.pcloud.vertices))
        _, target_vertex = self.pcloud.get_closest_points(self.x_home)
        u0[self.source_vertices[0]] = 1

        # 预计算测地线和梯度
        geodesic_arr, geodesic_gradient_arr = (
            self.scalar_diffusion.precompute_geodesics_and_gradients(
                [self.source_vertices[0]]
            )
        )

        # 积分扩散
        self.ut = self.scalar_diffusion.integrate_diffusion(u0)

        x_start = np.copy(self.x_arr[-1])
        for _ in range(500):
            # 向后走一步（负方向）
            x_next, local_basis = self.local_step(x_start, direction=0, sign=-1)

            # 投影到指定距离表面
            x_next, _, projected_point = self.pcloud.correct_distance_smooth(
                x_next, distance_to_surface
            )

            # 计算到家的测地线距离
            geodesic_distance2home = geodesic_arr[0, projected_point]

            # 如果距离足够近，停止
            if geodesic_distance2home < 1e-2:
                break

            x_start = np.copy(x_next)

            # 添加到轨迹
            self.x_arr.append(x_next)

            # 恢复局部坐标系并计算
            self.pcloud.local_bases = local_basis_real
            batch_points = self.wos.get_batch_from_point(x_next)
            local_basis, _, _ = self.wos.diffuse_rotations(batch_points)

            self.trajectory_local_bases.append(local_basis)

        # 恢复原始局部坐标系
        self.pcloud.local_bases = local_basis_real
