"""
骨骼弧面轨迹生成 - 运行脚本

使用方法：
    # 方式1：自动选点（沿X轴两端）
    python run_bone_arc.py --auto --axis 0

    # 方式2：交互式选点（在3D窗口中点击选择起终点）
    python run_bone_arc.py --interactive

    # 方式3：指定起终点索引
    python run_bone_arc.py --start 100 --end 800

    # 方式4：指定点云文件
    python run_bone_arc.py --ply Bone.ply --auto --axis 2

    # 方式5：自定义参数
    python run_bone_arc.py --auto --diffusion 3000 --step 0.0015 --offset -0.001
"""

import argparse
import os
import sys
import traceback

import numpy as np
import open3d as o3d

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from diffused_fields.manifold import Pointcloud

# 导入本模块
from point_selector import PointSelector
from arc_trajectory import ArcTrajectory


def resolve_ply_path(ply_name):
    """
    解析点云文件路径。

    优先查找顺序：
    1. 绝对路径
    2. bone-cuting 目录下
    3. diffused_fields/data/pointclouds/ 目录下
    """
    # 绝对路径
    if os.path.isabs(ply_name) and os.path.exists(ply_name):
        return ply_name

    # bone-cuting 目录下
    local_path = os.path.join(script_dir, ply_name)
    if os.path.exists(local_path):
        return local_path

    # diffused_fields/data/pointclouds/ 目录下
    data_path = os.path.join(
        project_root, "diffused_fields", "data", "pointclouds", ply_name
    )
    if os.path.exists(data_path):
        return data_path

    raise FileNotFoundError(f"找不到点云文件: {ply_name}")


def load_pcloud(ply_name, voxel_size=None):
    """
    加载点云并执行必要的预处理。

    Args:
        ply_name: 点云文件名或路径
        voxel_size: 体素下采样大小（None=不额外下采样）

    Returns:
        Pointcloud 对象
    """
    # 先用 Open3D 检查并归一化尺度
    ply_path = resolve_ply_path(ply_name)
    print(f"[加载] 点云文件: {ply_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise ValueError("点云为空")

    # 检查是否需要缩放（基于包围盒尺寸）
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_dim = max(extent)

    if max_dim > 10.0:
        # 假设大于10的数值是毫米单位，需要转为米
        print(f"  检测到最大跨度 {max_dim:.3f} (可能是mm)，缩放为米制...")
        pcd.scale(0.001, center=(0, 0, 0))
    else:
        print(f"  点云尺寸合理 (最大跨度: {max_dim:.3f})")

    # 保存到临时文件供 Pointcloud 类加载
    data_dir = os.path.join(
        project_root, "diffused_fields", "data", "pointclouds"
    )
    temp_filename = "_bone_arc_temp.ply"
    temp_path = os.path.join(data_dir, temp_filename)

    try:
        o3d.io.write_point_cloud(temp_path, pcd, write_ascii=True)
        pcloud = Pointcloud(
            filename=temp_filename,
            voxel_size=voxel_size,
        )
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    # 确保法线已估计
    if not hasattr(pcloud, 'normals'):
        pcloud.get_normals(num_neighbors=20)

    print(f"  加载完成: {len(pcloud.vertices)} 个顶点")
    return pcloud


def run_auto(pcloud, axis, args):
    """自动选点模式"""
    selector = PointSelector(pcloud)
    start_idx, end_idx = selector.select_auto(axis=axis)
    run_trajectory(pcloud, start_idx, end_idx, args)


def run_interactive(pcloud, args):
    """交互式选点模式"""
    selector = PointSelector(pcloud)
    start_idx, end_idx = selector.select_interactive()
    selector.visualize_selection()
    run_trajectory(pcloud, start_idx, end_idx, args)


def run_by_index(pcloud, start_idx, end_idx, args):
    """指定索引模式"""
    selector = PointSelector(pcloud)
    start_idx, end_idx = selector.select_by_index(start_idx, end_idx)
    run_trajectory(pcloud, start_idx, end_idx, args)


def run_trajectory(pcloud, start_idx, end_idx, args):
    """
    运行轨迹生成。

    Args:
        pcloud: 点云对象
        start_idx: 起点索引
        end_idx: 终点索引
        args: 命令行参数
    """
    print("\n" + "=" * 60)
    print("  开始生成骨骼弧面轨迹")
    print("=" * 60)

    # 创建轨迹生成器
    generator = ArcTrajectory(
        pcloud,
        start_idx=start_idx,
        end_idx=end_idx,
        diffusion_scalar=args.diffusion,
        step_size=args.step,
        distance_to_surface=args.offset,
        num_init_steps=args.init_steps,
        max_steps=args.max_steps,
        endpoint_tolerance=args.tolerance,
    )

    # 计算轨迹
    try:
        generator.compute()
    except Exception as e:
        print(f"\n[错误] 轨迹生成失败:")
        traceback.print_exc()
        sys.exit(1)

    # 打印摘要
    generator.print_summary()

    # 保存结果
    if args.save:
        generator.save_results()

    # 输出位姿信息（用于机械臂控制）
    positions, rotations = generator.get_trajectory_poses()
    print(f"\n[机械臂数据] 轨迹位姿序列已就绪:")
    print(f"  位置数组: {positions.shape}  (N个点 x 3D坐标)")
    print(f"  姿态数组: {rotations.shape}  (N个点 x 3x3旋转矩阵)")
    print(f"  每个旋转矩阵的列向量含义:")
    print(f"    第0列 = X轴 = 沿轨迹前进方向")
    print(f"    第1列 = Y轴 = 侧向（表面内垂直于轨迹）")
    print(f"    第2列 = Z轴 = 法向（表面法线方向）")

    # 可视化
    if args.no_viz:
        print("\n[跳过可视化]")
    else:
        print(f"\n[可视化] 启动渲染器...")
        generator.visualize(
            show_frames=not args.no_frames,
            use_polyscope=not args.use_o3d,
        )


def main():
    parser = argparse.ArgumentParser(
        description="骨骼弧面轨迹生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python run_bone_arc.py --auto --axis 0
  python run_bone_arc.py --interactive
  python run_bone_arc.py --start 100 --end 800
  python run_bone_arc.py --ply Bone.ply --auto --diffusion 3000
        """,
    )

    # 点云参数
    parser.add_argument(
        "--ply", type=str, default="Bone.ply",
        help="点云文件名（在 diffused_fields/data/pointclouds/ 下）"
    )
    parser.add_argument(
        "--voxel", type=float, default=None,
        help="体素下采样大小（米），None=不下采样"
    )

    # 选点模式
    point_group = parser.add_mutually_exclusive_group(required=True)
    point_group.add_argument(
        "--auto", action="store_true",
        help="自动选点（沿指定轴两端）"
    )
    point_group.add_argument(
        "--interactive", action="store_true",
        help="交互式选点（在3D窗口中点击）"
    )
    point_group.add_argument(
        "--start", type=int, default=None,
        help="起点索引（需同时指定 --end）"
    )

    # 选点参数
    parser.add_argument(
        "--end", type=int, default=None,
        help="终点索引"
    )
    parser.add_argument(
        "--axis", type=int, default=0, choices=[0, 1, 2],
        help="自动选点的轴方向 (0=X, 1=Y, 2=Z)"
    )

    # 轨迹参数
    parser.add_argument(
        "--diffusion", type=float, default=2000,
        help="扩散标量（越大越平滑，骨骼有孔洞用3000-5000）"
    )
    parser.add_argument(
        "--step", type=float, default=0.002,
        help="步长（米），建议0.001-0.003"
    )
    parser.add_argument(
        "--offset", type=float, default=-0.001,
        help="距表面偏移（米），负值=表面外侧"
    )
    parser.add_argument(
        "--init-steps", type=int, default=20,
        help="初始安全偏移步数"
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000,
        help="最大步数"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1.5,
        help="终点到达容差倍数（相对步长），越小越精准，建议1.0-2.0"
    )

    # 可视化与输出
    parser.add_argument(
        "--no-viz", action="store_true",
        help="不进行可视化"
    )
    parser.add_argument(
        "--no-frames", action="store_true",
        help="不显示轨迹局部坐标系"
    )
    parser.add_argument(
        "--use-o3d", action="store_true",
        help="使用 Open3D 而非 Polyscope 可视化"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="保存结果到文件"
    )

    args = parser.parse_args()

    # 验证参数
    if args.start is not None and args.end is None:
        parser.error("指定了 --start 但未指定 --end")
    if args.start is not None and args.interactive:
        parser.error("--start 和 --interactive 不能同时使用")
    if args.start is not None and args.auto:
        parser.error("--start 和 --auto 不能同时使用")

    # 加载点云
    try:
        pcloud = load_pcloud(args.ply, voxel_size=args.voxel)
    except Exception as e:
        print(f"[错误] 加载点云失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 根据选点模式运行
    if args.auto:
        run_auto(pcloud, args.axis, args)
    elif args.interactive:
        run_interactive(pcloud, args)
    elif args.start is not None:
        run_by_index(pcloud, args.start, args.end, args)


if __name__ == "__main__":
    main()
