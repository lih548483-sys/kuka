import open3d as o3d
import os
import sys
import tkinter as tk
from tkinter import filedialog

def select_input_file() -> str:
    """
    调用系统图形界面选择输入文件。
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 强制窗口焦点置于顶层
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="选择要处理的点云模型文件",
        filetypes=[
            ("PLY Files", "*.ply"),
            ("PCD Files", "*.pcd"),
            ("All Files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def preprocess_bone_pointcloud(input_path: str, output_path: str):
    """
    执行严谨的点云去噪与法向量估算管线。
    """
    if not os.path.exists(input_path):
        print(f"[Error] 输入文件路径无效: {input_path}")
        sys.exit(1)

    print(f"\n[Info] 正在加载点云: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    
    if pcd.is_empty():
        print("[Error] 点云加载失败或文件为空，请检查文件编码格式。")
        sys.exit(1)
        
    original_size = len(pcd.points)
    print(f"[Info] 原始点云顶点数量: {original_size}")

    # ==========================================
    # 1: 物理尺度计算
    # ==========================================
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    print(f"[Info] 点云包围盒尺寸 (X, Y, Z): {extent}")
    print(f"[Info] 最大跨度: {max(extent):.4f}")
    
    volume = extent[0] * extent[1] * extent[2]
    if original_size > 0:
        avg_spacing = (volume / original_size) ** (1/3)
    else:
        avg_spacing = 0.001
    print(f"[Info] 估算平均点间距: {avg_spacing:.6f}")

    # ==========================================
    # 2: 统计滤波 (SOR)
    # ==========================================
    print("[Info] 执行统计滤波 (去除高斯离群点)...")
    pcd_clean, ind_sor = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    sor_size = len(ind_sor)
    print(f"[Info] 统计滤波后剩余点数: {sor_size} (剔除 {original_size - sor_size} 点)")

    if sor_size == 0:
        print("[Error] 统计滤波剔除了所有点，算法终止。原因：点云密度极度不均或输入数据异常。")
        sys.exit(1)

    # ==========================================
    # 3: 半径滤波 (ROR) - 动态自适应半径
    # ==========================================
    # 以平均间距的 5 倍作为搜索半径，保证尺度不变性
    dynamic_radius = avg_spacing * 5.0
    ror_points = 10
    print(f"[Info] 执行半径滤波 (动态半径={dynamic_radius:.4f}, 最少点数={ror_points})...")
    
    pcd_final, ind_ror = pcd_clean.remove_radius_outlier(nb_points=ror_points, radius=dynamic_radius)
    final_size = len(ind_ror)
    print(f"[Info] 半径滤波后剩余点数: {final_size} (剔除 {sor_size - final_size} 点)")

    if final_size == 0:
        print("[Error] 半径滤波剔除了所有点，算法终止。原因：物理尺度判定失效。")
        sys.exit(1)

    # ==========================================
    # 4: 拓扑结构更新与法向量估算
    # ==========================================
    normal_radius = max(dynamic_radius, avg_spacing * 10)
    print(f"[Info] 重新计算法向量 (KDTree 搜索半径: {normal_radius:.4f})...")
    pcd_final.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    pcd_final.orient_normals_consistent_tangent_plane(100)

    # ==========================================
    # 5: 数据落盘
    # ==========================================
    print(f"[Info] 正在持久化处理结果至: {output_path}")
    success = o3d.io.write_point_cloud(output_path, pcd_final, write_ascii=True)
    
    if success:
        print("[Success] 点云预处理管线执行完毕。")
    else:
        print("[Error] I/O 异常，文件保存失败。")

if __name__ == "__main__":
    # 交互式获取输入路径
    input_file_path = select_input_file()
    
    # 校验用户操作
    if not input_file_path:
        print("[Info] 用户取消了文件选择，程序正常退出。")
        sys.exit(0)
        
    # 自动推导输出路径
    # 例如: "C:/data/Bone.ply" -> "C:/data/Bone_denoised.ply"
    directory = os.path.dirname(input_file_path)
    base_name, extension = os.path.splitext(os.path.basename(input_file_path))
    output_file_name = f"{base_name}_denoised{extension}"
    output_file_path = os.path.join(directory, output_file_name)
    
    preprocess_bone_pointcloud(input_file_path, output_file_path)