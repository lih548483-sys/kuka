import open3d as o3d
import sys
import os
import tkinter as tk
from tkinter import filedialog

def select_input_file():
    """弹出文件选择对话框，让用户选择要处理的点云文件"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 支持的点云文件格式
    filetypes = [
        ("点云文件", "*.ply"),
        ("点云文件", "*.pcd"),
        ("点云文件", "*.xyz"),
        ("点云文件", "*.txt"),
        ("所有文件", "*.*")
    ]
    
    filepath = filedialog.askopenfilename(
        title="选择要处理的点云文件",
        filetypes=filetypes,
        initialdir=os.getcwd()
    )
    
    root.destroy()
    return filepath

def select_save_path(default_filename):
    """弹出保存对话框，让用户选择保存路径"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 默认文件名添加_sparse后缀
    base, ext = os.path.splitext(default_filename)
    suggested_name = f"{base}_sparse{ext}"
    
    save_path = filedialog.asksaveasfilename(
        title="选择保存路径",
        defaultextension=".ply",
        initialfile=suggested_name,
        filetypes=[
            ("PLY 文件", "*.ply"),
            ("PCD 文件", "*.pcd"),
            ("所有文件", "*.*")
        ],
        initialdir=os.getcwd()
    )
    
    root.destroy()
    return save_path

class InteractiveDownsampler:
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            print(f"[错误] I/O 中断：文件不存在 -> {filepath}")
            sys.exit(1)
            
        print("[1/3] 正在加载原始点云并在内存中构建流形副本...")
        self.pcd_raw = o3d.io.read_point_cloud(filepath)
        if not self.pcd_raw.has_normals():
            self.pcd_raw.estimate_normals()
            
        self.pcd_current = o3d.geometry.PointCloud(self.pcd_raw)
        self.original_filename = os.path.basename(filepath)
        
        # 物理尺度评估：动态计算分辨率步长
        bbox_extent = self.pcd_raw.get_max_bound() - self.pcd_raw.get_min_bound()
        self.max_extent = max(bbox_extent)
        # 设定基准步长为模型最大跨度的 1/500，保证调整粒度的连续性
        self.step_size = self.max_extent / 500.0
        self.voxel_size = 0.0  
        
        print(f"      -> 原始拓扑顶点数: {len(self.pcd_raw.points)}")
        print(f"      -> 模型空间最大跨度: {self.max_extent:.4f}m")
        print(f"      -> 体素栅格调节步长: {self.step_size:.6f}m")

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=1024, height=768, 
                               window_name="科研级点云密度标定 (关闭窗口以进入导出流程)")
        self.vis.add_geometry(self.pcd_current)
        
        # 消除 GLFW ASCII 映射的大小写盲区
        self.vis.register_key_callback(ord('='), self.increase_sparsity)
        self.vis.register_key_callback(ord('+'), self.increase_sparsity)
        self.vis.register_key_callback(ord('-'), self.decrease_sparsity)
        self.vis.register_key_callback(ord('_'), self.decrease_sparsity)

    def _execute_downsample(self):
        """执行确定性体素降采样并触发显存同步"""
        if self.voxel_size <= 1e-6:
            downsampled = self.pcd_raw
        else:
            downsampled = self.pcd_raw.voxel_down_sample(self.voxel_size)
            
        self.pcd_current.points = downsampled.points
        if downsampled.has_colors():
            self.pcd_current.colors = downsampled.colors
        if downsampled.has_normals():
            self.pcd_current.normals = downsampled.normals
            
        self.vis.update_geometry(self.pcd_current)
        print(f"[内存状态] 当前 Voxel Size: {self.voxel_size:.6f} | 剩余顶点数: {len(self.pcd_current.points)}")

    def increase_sparsity(self, vis):
        self.voxel_size += self.step_size
        self._execute_downsample()

    def decrease_sparsity(self, vis):
        self.voxel_size = max(0.0, self.voxel_size - self.step_size)
        self._execute_downsample()

    def run(self):
        print("\n[2/3] 渲染管线已启动。")
        print("      操作说明:")
        print("      - 按 '+' 或 '=' 键增加稀疏度 (增大体素尺寸)")
        print("      - 按 '-' 或 '_' 键减少稀疏度 (减小体素尺寸)")
        print("      - 关闭窗口以完成处理并进入保存流程\n")
        
        self.vis.run()
        self.vis.destroy_window()
        
        self.export_result()
    
    def export_result(self):
        """导出处理后的点云"""
        print("\n[3/3] 进入导出流程...")
        
        # 让用户选择保存路径
        save_path = select_save_path(self.original_filename)
        
        if not save_path:
            print("      用户取消保存操作")
            print("[完成] 程序已退出")
            return
        
        # 保存点云
        try:
            o3d.io.write_point_cloud(save_path, self.pcd_current)
            print(f"      -> 点云已成功保存到: {save_path}")
            print(f"      -> 保存的顶点数: {len(self.pcd_current.points)}")
            print(f"      -> 使用的体素尺寸: {self.voxel_size:.6f}m")
        except Exception as e:
            print(f"[错误] 保存失败: {str(e)}")
            sys.exit(1)
        
        print("[完成] 点云稀疏处理已完成")

if __name__ == "__main__":
    print("=" * 60)
    print("         科研级点云稀疏处理工具")
    print("=" * 60)
    
    # 让用户选择输入文件
    input_file = select_input_file()
    
    if not input_file:
        print("用户取消选择文件，程序退出")
        sys.exit(0)
    
    print(f"\n已选择文件: {input_file}")
    
    # 创建并运行处理程序
    try:
        processor = InteractiveDownsampler(input_file)
        processor.run()
    except Exception as e:
        print(f"\n[错误] 程序运行异常: {str(e)}")
        sys.exit(1)
