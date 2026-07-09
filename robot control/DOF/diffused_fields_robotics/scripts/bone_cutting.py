"""
严谨版：基于扩散方向场的骨骼表面贴合滑动轨迹生成脚本
（修复路径拼接Bug、实现内存级即时擦除、引入流形端点自适应搜索与数值微调）
"""
import os
import sys
import traceback
import open3d as o3d
import numpy as np
from diffused_fields.manifold import Pointcloud
from diffused_fields_robotics.local_action_primitives.action_primitives import Cutting

# ==============================================================================
#                                实验参数配置区
# ==============================================================================
class Config:
    # 1. 点云数据路径配置（支持绝对路径或基于脚本的相对路径推导）
    # 若置为 None，则脚本会自动推导至项目默认的 ../diffused_fields/data/pointclouds 目录
    CUSTOM_DATA_DIR = None 
    
    SOURCE_PLY_NAME = "banana_half.ply"    # 输入的原始点云文件名

    # 2. 轨迹演化方向配置
    # 0 = X轴, 1 = Y轴, 2 = Z轴。算法将沿着该轴向寻找骨骼几何跨度最长的两端作为起止点。
    ALIGNMENT_AXIS = 0  
# ==============================================================================


def resolve_data_paths():
    """
    确定性路径解析函数。根据 Config 配置返回输入文件的绝对路径及标准数据目录。
    """
    if Config.CUSTOM_DATA_DIR is not None:
        data_dir = os.path.abspath(Config.CUSTOM_DATA_DIR)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.abspath(os.path.join(project_root, "..", "diffused_fields", "data", "pointclouds"))
    
    source_path = os.path.join(data_dir, Config.SOURCE_PLY_NAME)
    return source_path, data_dir


def run_verification():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "config", "pointclouds.yaml")
    
    if not os.path.exists(config_path):
        print(f"[错误] 系统中断：未找到配置文件 -> {config_path}")
        sys.exit(1)
    
    # 1. 解析路径并加载原始点云
    source_path, data_dir = resolve_data_paths()
    if not os.path.exists(source_path):
        print(f"[错误] 数据源不存在 -> {source_path}")
        sys.exit(1)
        
    print(f"[0/4] 正在加载原始点云并执行内存级几何尺度归一化 (mm -> m)...")
    pcd = o3d.io.read_point_cloud(source_path)
    
    # 验证是否需要缩放 (基于包围盒尺寸的启发式检查)
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_dimension = max(extent)
    
    if max_dimension > 10.0:  # 假设骨骼在米级单位下不可能大于 10 米
        print(f"      检测到模型最大跨度为 {max_dimension:.3f} mm，正在进行内存缩放...")
        pcd.scale(0.001, center=(0, 0, 0))
    else:
        print(f"      检测到模型尺寸已在合理范围 (最大跨度: {max_dimension:.3f} m)，维持原尺度。")

    # 2. 生成同级目录下的安全临时文件名，顺应底层库的硬编码相对路径拼接
    temp_filename = "_mem_temp_run_.ply"
    temp_absolute_path = os.path.join(data_dir, temp_filename)
    
    print("[1/4] 正在通过流形结构加载内存点云...")
    try:
        # 将内存中处理好的点云物理写入该安全路径
        success = o3d.io.write_point_cloud(temp_absolute_path, pcd, write_ascii=True)
        if not success:
            print(f"[致命错误] Open3D 写入安全临时文件失败 -> {temp_absolute_path}")
            sys.exit(1)
            
        # 顺应底层库逻辑：传入相对文件名，让其内部拼出正确路径
        pcloud = Pointcloud(filename=temp_filename)
        
    except Exception as e:
        print(f"[致命错误] 流形初始化过程中捕获到异常：{e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 严谨性保障：一旦初始化完毕，必须立即在退出前擦除该临时文件，确保磁盘不留痕迹
        if os.path.exists(temp_absolute_path):
            try:
                os.remove(temp_absolute_path)
            except Exception as e:
                print(f"[警告] 无法自动清理临时文件 {temp_absolute_path}: {e}")
    
    # 3. 检查点云有效性
    total_vertices = len(pcloud.vertices) if hasattr(pcloud, 'vertices') else 0
    if total_vertices == 0:
        print("[致命错误] 加载的点云顶点数为 0，流形结构构建失败。")
        sys.exit(1)
    print(f"      流形加载成功。当前点云有效总顶点数: {total_vertices}")

    # ==============================================================================
    # 严谨修正：自适应寻找点云在指定主轴两端跨度最大的两个顶点，确保势能场两极正确摆放
    # ==============================================================================
    vertices_matrix = np.array(pcloud.vertices)
    axis = Config.ALIGNMENT_AXIS
    
    # 查找在该轴向上的极小值和极大值点索引
    min_idx = int(np.argmin(vertices_matrix[:, axis]))
    max_idx = int(np.argmax(vertices_matrix[:, axis]))
    
    dynamic_source_vertices = [min_idx, max_idx]
    print(f"      [自适应索引解算] 已定位骨骼两端拓扑极点索引: {dynamic_source_vertices}")
    # ==============================================================================

    print("[2/4] 正在实例化 Cutting 动作基元控制器...")
    # 显式传入自适应解算出的两端关键点作为边界热源
    controller = Cutting(pcloud, source_vertices=dynamic_source_vertices)
    
   # ===================== 核心流形边界约束区 (多连通拓扑优化) =====================
    # 1. 强力提升全场扩散权重（原 1500 升级至 5000）
    # 物理意义：强行让势能场穿透中间的椎孔空洞，为轨迹提供跨越拓扑障碍的宏观全局引力场
    if hasattr(controller, 'diffusion_scalar'):
        controller.diffusion_scalar = 5000
        
    # 2. 适当放宽微观表面约束容差，赋予轨迹在曲率突变处的“数值动量”
    # 将贴合高度放宽至 1mm (-0.001)，步长放宽至 1.5mm (0.0015)，防止在孔洞边缘因精密投影失败而卡死
    controller.distance_to_surface = -0.001 
    controller.step_size = 0.0015
    
    # 3. 增加初始安全步数，保证起步时建立足够的方向矢量
    controller.num_init_steps = 30 
    
    # 4. 提供充裕的积分步数上限
    controller.num_cut_steps = 4000 
    # ==============================================================================
    
    print("[3/4] 正在求解偏微分方程，计算扩散方向场并生成步进轨迹...")
    try:
        # 显式激活底层库的扩散场松弛解算（若存在该显式接口）
        if hasattr(controller, 'initialize_diffusion'):
            controller.initialize_diffusion()
            
        # 执行 Walk-on-Spheres 算法与轨迹积分
        controller.run()
    except Exception as e:
        print(f"[致命错误] 扩散场求解或轨迹积分失败：")
        traceback.print_exc()
        sys.exit(1)
        
    print(f"[4/4] 轨迹计算完毕。实际生成的轨迹点数量: {len(controller.trajectory)}")
    
    # 4. 结果验证
    if len(controller.trajectory) == 0:
        print("[致命错误] 生成的轨迹点数量为 0。请进入 Polyscope 检查两极蓝球是否由于摆放轴不对导致重合。")
        sys.exit(1)
        
    print("      正在启动 Polyscope 渲染器...")
    controller.visualize_trajectory(show_tool=False)


if __name__ == "__main__":
    run_verification()