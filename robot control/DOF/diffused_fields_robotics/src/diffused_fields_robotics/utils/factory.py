"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
工厂函数模块 - 动作原语控制器创建

本模块提供工厂模式（Factory Pattern）来标准化控制器的创建过程，
消除重复的初始化代码。

工厂模式的优势：
1. 集中管理对象创建逻辑
2. 减少代码重复
3. 更容易扩展新的原语类型
4. 统一错误处理和验证

主要功能：
1. create_primitive_controller: 创建单个动作原语控制器
2. create_batch_controllers: 创建批量控制器用于批量处理
3. create_controller_from_config: 从配置字典创建控制器
4. create_experiment_suite: 创建多种原语的实验套件
5. get_primitive_defaults: 获取原语的默认参数
6. validate_primitive_config: 验证配置字典的有效性
"""

# 导入 NumPy 用于数值计算
import numpy as np
# 导入类型注解
from typing import Optional, Any, Dict, Union

# 导入核心类 - 点云类
try:
    from diffused_fields import Pointcloud
except ImportError:
    print("Warning: diffused_fields not available. Some functionality may be limited.")
    Pointcloud = None


def create_primitive_controller(
    primitive_type: str,
    pcloud: Union[Pointcloud, str],
    diffusion_scalar: Optional[float] = None,
    source_vertices: Optional[np.ndarray] = None,
    start_vertex: Optional[int] = None,
    end_vertex: Optional[int] = None,
    **kwargs
) -> Any:
    """
    工厂函数：创建动作原语控制器。

    这是最常用的工厂函数，用于创建单个动作原语控制器。
    支持多种输入格式，自动处理点云的加载。

    参数:
        primitive_type: 原语类型 ('cutting', 'slicing', 'peeling', 'coverage')
        pcloud: 点云对象或文件名
        diffusion_scalar: 扩散标量参数覆盖
        source_vertices: 源顶点数组
        start_vertex: 起始顶点索引
        end_vertex: 终止顶点索引
        **kwargs: 传递给控制器构造函数的其他参数

    返回:
        初始化后的动作原语控制器对象

    异常:
        ValueError: 原语类型不支持
        TypeError: pcloud 既不是 Pointcloud 也不是字符串
        ImportError: 无法导入动作原语依赖
    """
    # 动态导入动作原语，检查依赖
    from ..utils.experiment_base import _import_action_primitives
    if not _import_action_primitives():
        raise ImportError("Action primitives could not be imported due to missing dependencies")

    # 导入具体的动作原语类
    from ..local_action_primitives.action_primitives import (
        Cutting, Slicing, Peeling, Coverage
    )

    # 处理点云输入
    # 支持字符串文件名，自动加载为 Pointcloud 对象
    if isinstance(pcloud, str):
        pcloud = Pointcloud(filename=pcloud)
    elif not hasattr(pcloud, 'vertices'):
        # 确保是有效的点云对象
        raise TypeError("pcloud must be a Pointcloud object or filename string")

    # 定义原语类型到类的映射
    # 使用字典而不是 if-elif 链，便于扩展新的原语类型
    PRIMITIVE_CLASSES = {
        'cutting': Cutting,    # 切割原语
        'slicing': Slicing,    # 切片原语
        'peeling': Peeling,    # 去皮原语
        'coverage': Coverage,  # 覆盖原语
    }

    # 验证原语类型
    if primitive_type not in PRIMITIVE_CLASSES:
        raise ValueError(f"Unknown primitive type: {primitive_type}. "
                        f"Supported types: {list(PRIMITIVE_CLASSES.keys())}")

    # 获取对应的类
    controller_class = PRIMITIVE_CLASSES[primitive_type]

    # 准备构造函数参数
    init_args = {
        'pcloud': pcloud,
        'primitive_type': primitive_type,
    }

    # 添加可选参数（仅当提供时才添加）
    if diffusion_scalar is not None:
        init_args['diffusion_scalar'] = diffusion_scalar
    if source_vertices is not None:
        init_args['source_vertices'] = source_vertices
    if start_vertex is not None:
        init_args['start_vertex'] = start_vertex
    if end_vertex is not None:
        init_args['end_vertex'] = end_vertex

    # 添加其他关键字参数
    init_args.update(kwargs)

    # 创建控制器实例
    controller = controller_class(**init_args)

    print(f"✓ Created {primitive_type} controller with {len(pcloud.vertices)} vertices")

    return controller


def create_batch_controllers(
    primitive_type: str,
    pcloud: Union[Pointcloud, str],
    num_controllers: int,
    diffusion_scalars: Optional[np.ndarray] = None,
    **common_kwargs
) -> list:
    """
    创建批量控制器用于批量处理。

    批量创建多个控制器，每个使用不同的参数。
    用于参数搜索或敏感性分析。

    参数:
        primitive_type: 原语类型
        pcloud: 点云对象或文件名
        num_controllers: 要创建的控制器数量
        diffusion_scalars: 扩散标量数组
        **common_kwargs: 所有控制器共用的参数

    返回:
        控制器实例列表

    示例:
        # 创建10个切割控制器，扩散标量从0.1到10000对数分布
        controllers = create_batch_controllers('cutting', pcloud, 10)
    """
    controllers = []

    # 处理点云输入
    if isinstance(pcloud, str):
        pcloud = Pointcloud(filename=pcloud)

    # 如果没有提供扩散标量数组，生成对数分布的默认值
    if diffusion_scalars is None:
        # 对数空间均匀分布：10^(-1) 到 10^4
        diffusion_scalars = np.logspace(np.log10(0.1), np.log10(10000), num_controllers)

    # 逐个创建控制器
    for i in range(num_controllers):
        # 如果提供了扩散标量数组，使用对应的值
        if i < len(diffusion_scalars):
            diffusion_scalar = diffusion_scalars[i]
        else:
            diffusion_scalar = None

        # 创建控制器
        controller = create_primitive_controller(
            primitive_type=primitive_type,
            pcloud=pcloud,
            diffusion_scalar=diffusion_scalar,
            **common_kwargs
        )

        controllers.append(controller)

    print(f"✓ Created {len(controllers)} {primitive_type} controllers")

    return controllers


def create_controller_from_config(
    config: Dict[str, Any],
    pcloud: Optional[Union[Pointcloud, str]] = None
) -> Any:
    """
    从配置字典创建控制器。

    简化配置文件或字典驱动的控制器创建流程。
    支持从 JSON/YAML 配置文件或 API 参数创建控制器。

    参数:
        config: 配置字典，包含控制器参数
        pcloud: 可选的点云对象（可被 config 覆盖）

    返回:
        初始化后的控制器对象

    配置字典示例:
        {
            'primitive_type': 'cutting',
            'filename': 'banana_half.ply',  # 或使用 pcloud 参数
            'diffusion_scalar': 1000.0,
            'source_vertices': [10, 20],
            'start_vertex': 10
        }
    """
    config = config.copy()  # 不修改原始配置字典

    # 提取必需的原语类型
    primitive_type = config.pop('primitive_type')

    # 处理点云来源
    if 'filename' in config:
        # 如果 config 中有 filename，使用它
        pcloud = config.pop('filename')
    elif pcloud is None:
        # 如果既没有提供 pcloud 也没有 filename，抛出错误
        raise ValueError("Either 'filename' must be in config or pcloud must be provided")

    # 使用剩余的 config 作为 kwargs 创建控制器
    return create_primitive_controller(
        primitive_type=primitive_type,
        pcloud=pcloud,
        **config
    )


def get_primitive_defaults(primitive_type: str) -> Dict[str, Any]:
    """
    获取特定原语类型的默认参数。

    用于查看或文档化各原语类型的默认配置。

    参数:
        primitive_type: 原语类型

    返回:
        默认参数字典
    """
    # 定义各原语的默认参数
    defaults = {
        'cutting': {
            'diffusion_scalar': 1000.0,  # 切割默认使用较大的扩散标量
        },
        'slicing': {
            'diffusion_scalar': 1000.0,  # 切片默认使用较大的扩散标量
        },
        'peeling': {
            'diffusion_scalar': 1000.0,  # 去皮默认使用较大的扩散标量
        },
        'coverage': {
            'diffusion_scalar': 1000.0,  # 覆盖默认使用较大的扩散标量
        },
    }

    # 返回对应类型的默认参数，如果不存在则返回空字典
    return defaults.get(primitive_type, {})


def validate_primitive_config(config: Dict[str, Any]) -> bool:
    """
    验证原语配置字典的有效性。

    在创建控制器之前验证配置，避免运行时错误。
    进行以下检查：
    1. 必需字段是否存在
    2. 原语类型是否有效
    3. 数值参数是否在有效范围内

    参数:
        config: 要验证的配置字典

    返回:
        验证通过返回 True

    异常:
        ValueError: 配置无效时抛出
    """
    # 必需的键
    required_keys = ['primitive_type']

    # 检查必需字段
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")

    # 验证原语类型
    valid_types = ['cutting', 'slicing', 'peeling', 'coverage']
    if config['primitive_type'] not in valid_types:
        raise ValueError(f"Invalid primitive_type. Must be one of: {valid_types}")

    # 验证可选的数值参数
    if 'diffusion_scalar' in config:
        value = config['diffusion_scalar']
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("diffusion_scalar must be a positive number")

    # 验证 source_vertices
    if 'source_vertices' in config:
        source_vertices = config['source_vertices']
        if not isinstance(source_vertices, (list, np.ndarray)) or len(source_vertices) == 0:
            raise ValueError("source_vertices must be a non-empty list or array")

    # 验证 start_vertex
    if 'start_vertex' in config:
        value = config['start_vertex']
        if not isinstance(value, int) or value < 0:
            raise ValueError("start_vertex must be a non-negative integer")

    return True


def create_experiment_suite(
    primitive_types: list,
    pcloud: Union[Pointcloud, str],
    **common_kwargs
) -> Dict[str, Any]:
    """
    创建多种原语的实验套件。

    用于同时测试多种动作原语。
    如果某个原语创建失败，会继续创建其他的。

    参数:
        primitive_types: 原语类型名称列表
        pcloud: 点云对象或文件名
        **common_kwargs: 所有控制器共用的参数

    返回:
        字典，键为原语类型，值为控制器实例

    示例:
        suite = create_experiment_suite(
            ['cutting', 'slicing', 'peeling'],
            pcloud
        )
        # suite = {'cutting': <Cutting>, 'slicing': <Slicing>, 'peeling': <Peeling>}
    """
    suite = {}

    # 处理点云输入
    if isinstance(pcloud, str):
        pcloud = Pointcloud(filename=pcloud)

    # 逐个创建控制器
    for primitive_type in primitive_types:
        try:
            controller = create_primitive_controller(
                primitive_type=primitive_type,
                pcloud=pcloud,
                **common_kwargs
            )
            suite[primitive_type] = controller
        except Exception as e:
            # 如果创建失败，打印警告但继续
            print(f"Warning: Failed to create {primitive_type} controller: {e}")

    print(f"✓ Created experiment suite with {len(suite)} controllers")

    return suite
