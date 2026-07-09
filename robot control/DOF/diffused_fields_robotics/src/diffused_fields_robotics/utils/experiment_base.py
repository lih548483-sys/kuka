"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
实验基类模块

本模块提供实验的基类，消除重复的样板代码。

设计模式：
- 基类继承：提供通用的实验框架
- 特化子类：为特定原语类型定制实验
- 动态导入：优雅处理可选依赖

主要类：
1. BaseBatchExperiment: 批量实验的基类
2. BatchSlicingBase: 切片实验特化基类
3. BatchPeelingBase: 去皮实验特化基类
4. BatchCoverageBase: 覆盖实验特化基类
"""

# 导入 pickle 用于序列化和反序列化
import pickle
# 导入 NumPy 用于数值计算
import numpy as np
# 导入类型注解
from typing import Callable, Optional, Dict, Any, List

# 从 diffused_fields 包导入点云类
from diffused_fields.manifold import Pointcloud
# 从核心配置模块导入批量结果路径获取函数
from ..core.config import get_batch_results_path

# 导入动作原语类 - 这些类可能存在但可能有导入问题
# 使用 None 初始化，允许动态导入
Cutting = Slicing = Peeling = Coverage = None


def _import_action_primitives():
    """
    动态导入动作原语，带错误处理。

    尝试导入四个动作原语类。
    如果导入失败，返回 False 但不抛出异常。

    Returns:
        bool: 导入是否成功
    """
    global Cutting, Slicing, Peeling, Coverage
    try:
        from ..local_action_primitives.action_primitives import (
            Cutting, Slicing, Peeling, Coverage
        )
        return True
    except ImportError as e:
        print(f"Warning: Could not import action primitives: {e}")
        return False


# 模块级别尝试导入动作原语
_import_action_primitives()


class BaseBatchExperiment:
    """
    批量实验的基类，消除常见的样板代码。

    这个类处理：
    - 通用实验参数设置
    - 点云初始化
    - 嵌套实验循环
    - 数据收集和存储
    - 结果保存

    使用方式：
    1. 继承此类
    2. 重写 experiment_func 方法定义具体实验
    3. 调用 run_experiment_loop 执行实验

    Attributes:
        filename: 点云文件名
        num_experiments: 实验数量
        num_samples: 每个实验的样本数量
        diffusion_scalar_arr: 扩散标量数组
        pcloud: 点云对象
        all_data: 所有实验结果列表
    """

    def __init__(
        self,
        filename: str = "banana_half.ply",
        num_experiments: int = 10,
        num_samples: int = 50,
        diffusion_scalar: float = 1000,
        diffusion_range: tuple = None,
        random_seed: int = 42
    ):
        """
        初始化批量实验基类。

        Args:
            filename: 要加载的点云文件名
            num_experiments: 实验数量（变化参数的数量）
            num_samples: 每个实验的随机噪声样本数
            diffusion_scalar: 固定扩散标量值（默认：1000）
            diffusion_range: 可选的 (最小, 最大) 扩散标量对数范围
                            如果指定，会覆盖固定的 diffusion_scalar
            random_seed: 随机种子，确保可重复性（默认：42）
        """
        self.filename = filename
        self.num_experiments = num_experiments
        self.num_samples = num_samples
        self.diffusion_range = diffusion_range
        self.random_seed = random_seed

        # 设置实验参数
        if diffusion_range is not None:
            # 如果指定了范围，使用对数空间变化的扩散标量
            # 这样可以在对数尺度上均匀采样
            self.diffusion_scalar_arr = np.logspace(
                np.log10(diffusion_range[0]),
                np.log10(diffusion_range[1]),
                num_experiments
            )
            print(f"Using varying diffusion_scalar from {diffusion_range[0]} to {diffusion_range[1]}")
        else:
            # 对所有实验使用固定的扩散标量
            self.diffusion_scalar_arr = np.full(num_experiments, diffusion_scalar)
            print(f"Using fixed diffusion_scalar: {diffusion_scalar} for all experiments")

        # 初始化数据存储
        self.all_data = []

        # 初始化点云
        self.pcloud = Pointcloud(filename=filename)

        print(f"Initialized batch experiment with {filename}")
        print(f"Experiments: {num_experiments}, Samples per experiment: {num_samples}")

    def run_experiment_loop(
        self,
        experiment_func: Callable,
        save_filename: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[Any, Any]]:
        """
        运行标准的嵌套实验循环。

        实验结构：
        - 外层循环：不同的扩散标量
        - 内层循环：不同的随机噪声样本

        Args:
            experiment_func: 每个 (exp_idx, sample_idx) 组合运行的函数
            save_filename: 可选的结果保存文件名
            progress_callback: 可选的进度跟踪回调函数

        Returns:
            实验结果列表

        Note:
            每个实验开始时设置随机种子，使得同一实验的样本可重复
            但不同实验之间有变化
        """
        total_experiments = self.num_experiments * self.num_samples
        current_exp = 0

        # 外层循环：不同实验配置
        for exp_idx in range(self.num_experiments):
            # 在每个实验开始时设置随机种子
            # 每个实验获得不同的基础种子
            np.random.seed(self.random_seed + exp_idx)

            # 获取当前实验的扩散标量
            diffusion_scalar = self.diffusion_scalar_arr[exp_idx]

            # 内层循环：不同样本
            for sample_idx in range(self.num_samples):
                # 不在这里重置种子，让 RNG 状态自然推进
                # 这样可以获得不同的噪声样本，但同一实验中可重复

                # 运行具体的实验
                result = experiment_func(exp_idx, sample_idx)

                # 添加通用元数据
                result.update({
                    "exp_idx": exp_idx,        # 实验索引
                    "sample_idx": sample_idx,   # 样本索引
                    "diffusion_scalar": diffusion_scalar,  # 当前扩散标量
                })

                self.all_data.append(result)

                # 进度跟踪
                current_exp += 1
                if progress_callback:
                    progress_callback(current_exp, total_experiments)
                else:
                    # 每10个实验或最后一个实验时打印进度
                    if current_exp % 10 == 0 or current_exp == total_experiments:
                        print(f"Progress: {current_exp}/{total_experiments} experiments completed")

        # 如果提供了文件名，保存结果
        if save_filename:
            self.save_results(save_filename)

        return self.all_data

    def save_results(self, filename: str):
        """
        将实验结果保存到 pickle 文件。

        Args:
            filename: 保存的文件名
        """
        filepath = get_batch_results_path(filename)
        print(f"Saving {len(self.all_data)} results to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self.all_data, f)
        print(f"Results saved successfully")

    def load_results(self, filename: str) -> List[Dict]:
        """
        从 pickle 文件加载实验结果。

        Args:
            filename: 要加载的文件名

        Returns:
            加载的结果列表
        """
        filepath = get_batch_results_path(filename)
        with open(filepath, "rb") as f:
            self.all_data = pickle.load(f)
        print(f"Loaded {len(self.all_data)} results from {filepath}")
        return self.all_data

    def compute_rmse_analysis(self) -> Dict[str, float]:
        """
        计算轨迹数据的基本 RMSE 分析。

        计算轨迹点之间差值的均方根。

        Returns:
            包含 RMSE 统计的字典

        Note:
            这是一个通用实现，子类可以重写以进行特定分析
        """
        if not self.all_data:
            return {}

        rmse_values = []
        for result in self.all_data:
            if 'trajectory' in result:
                # 简单 RMSE 计算 - 子类可重写
                trajectory = np.array(result['trajectory'])
                if len(trajectory) > 1:
                    # 计算相邻点之间的差值
                    diffs = np.diff(trajectory, axis=0)
                    # 计算欧几里得距离的平方
                    squared_distances = np.sum(diffs**2, axis=1)
                    # 计算均方根
                    rmse = np.sqrt(np.mean(squared_distances))
                    rmse_values.append(rmse)

        if rmse_values:
            return {
                'mean_rmse': np.mean(rmse_values),   # 平均 RMSE
                'std_rmse': np.std(rmse_values),     # RMSE 标准差
                'min_rmse': np.min(rmse_values),      # 最小 RMSE
                'max_rmse': np.max(rmse_values)       # 最大 RMSE
            }
        return {}



class BatchSlicingBase(BaseBatchExperiment):
    """
    切片实验的特化基类。

    包含切片实验的通用设置。
    自动从切片控制器获取参考关键点。

    Attributes:
        source_vertices: 源顶点索引
        original_keypoints: 原始关键点位置
    """

    def __init__(self, **kwargs):
        """
        初始化切片实验基类。

        尝试从切片控制器获取关键点。
        如果失败，使用备用关键点。
        """
        super().__init__(**kwargs)

        # 尝试从切片控制器获取参考关键点
        if Slicing is not None:
            try:
                controller = Slicing(self.pcloud)
                self.original_keypoints = self.pcloud.vertices[controller.source_vertices]
                self.source_vertices = controller.source_vertices
                print(f"✓ Slicing base initialized with {len(self.source_vertices)} keypoints")
            except Exception as e:
                print(f"Warning: Could not initialize Slicing controller: {e}")
                self._use_fallback_keypoints(kwargs)
        else:
            self._use_fallback_keypoints(kwargs)

    def _use_fallback_keypoints(self, kwargs):
        """
        当切片控制器不可用时使用备用关键点。

        Args:
            kwargs: 构造函数参数
        """
        # 使用提供的 source_vertices 或默认值
        self.source_vertices = kwargs.get('source_vertices', [0, len(self.pcloud.vertices)//2])
        self.original_keypoints = self.pcloud.vertices[self.source_vertices]
        print(f"✓ Using fallback keypoints: {len(self.source_vertices)} vertices")


class BatchPeelingBase(BaseBatchExperiment):
    """
    去皮实验的特化基类。

    包含去皮实验的通用设置。
    自动从去皮控制器获取参考关键点。
    """

    def __init__(self, **kwargs):
        """
        初始化去皮实验基类。
        """
        super().__init__(**kwargs)

        # 尝试从去皮控制器获取参考关键点
        if Peeling is not None:
            try:
                controller = Peeling(self.pcloud)
                self.source_vertices = controller.source_vertices
                print(f"✓ Peeling base initialized with {len(self.source_vertices)} keypoints")
            except Exception as e:
                print(f"Warning: Could not initialize Peeling controller: {e}")
                self._use_fallback_keypoints(kwargs)
        else:
            self._use_fallback_keypoints(kwargs)

    def _use_fallback_keypoints(self, kwargs):
        """
        当去皮控制器不可用时使用备用关键点。
        """
        # 使用提供的 source_vertices 或默认值（使用四分之一处）
        self.source_vertices = kwargs.get('source_vertices', [0, len(self.pcloud.vertices)//4])
        print(f"✓ Using fallback keypoints: {len(self.source_vertices)} vertices")


class BatchCoverageBase(BaseBatchExperiment):
    """
    覆盖实验的特化基类。

    覆盖实验使用边界顶点作为源。
    与其他原语不同，覆盖不需要预先指定的源顶点。
    """

    def __init__(self, **kwargs):
        """
        初始化覆盖实验基类。
        """
        super().__init__(**kwargs)

        # 尝试从覆盖控制器获取参考关键点
        if Coverage is not None:
            try:
                controller = Coverage(self.pcloud)
                # 覆盖使用边界顶点，可能初始为空
                self.source_vertices = controller.source_vertices if hasattr(controller, 'source_vertices') else []
                print(f"✓ Coverage base initialized with {len(self.source_vertices)} boundary keypoints")
            except Exception as e:
                print(f"Warning: Could not initialize Coverage controller: {e}")
                self._use_fallback_keypoints(kwargs)
        else:
            self._use_fallback_keypoints(kwargs)

    def _use_fallback_keypoints(self, kwargs):
        """
        当覆盖控制器不可用时使用备用关键点。

        Note:
            覆盖不需要源顶点，因为它使用边界。
            所以即使这里是空的也是正常的。
        """
        # 覆盖不需要源顶点 - 它使用边界
        self.source_vertices = kwargs.get('source_vertices', [])
        print(f"✓ Using fallback keypoints: {len(self.source_vertices)} vertices (empty for boundary-based coverage)")
