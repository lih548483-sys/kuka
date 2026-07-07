import polyscope as ps
import open3d as o3d
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def select_ply_file() -> str:
    """调用系统底层文件管理器选择 PLY 文件"""
    # 初始化 tkinter 并隐藏冗余的主窗口
    root = tk.Tk()
    root.withdraw()
    
    # 唤起系统文件选择对话框，限制文件后缀为 .ply
    file_path = filedialog.askopenfilename(
        title="请选择点云文件 (.ply)",
        filetypes=[("PLY Pointcloud", "*.ply"), ("All Files", "*.*")]
    )
    return file_path

def main():
    # 1. 动态获取文件路径
    print("正在唤起系统文件选择器...")
    ply_path = select_ply_file()
    
    # 验证输入合法性
    if not ply_path:
        print("操作中止：未选择任何文件。")
        return
        
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"文件路径不存在: {ply_path}")

    print(f"载入目标文件: {ply_path}")

    # 2. 读取并解析点云几何体数据
    pcd = o3d.io.read_point_cloud(ply_path)
    vertices = np.asarray(pcd.points)
    
    if len(vertices) == 0:
        raise ValueError("读取失败：文件中未包含有效的点云顶点数据。")
        
    print(f"模型加载完毕，顶点总数: {len(vertices)}")

    # 3. 初始化可视化上下文
    ps.init()
    
    # 提取文件名作为显示名称
    model_name = os.path.basename(ply_path).split('.')[0]
    ps_cloud = ps.register_point_cloud(model_name, vertices)
    ps_cloud.set_radius(0.005)

    print("系统提示：请在图形窗口中使用 Pick 工具进行源顶点采集，并记录对应 Point ID。")

    # 4. 挂起主线程，进入渲染循环
    ps.show()

if __name__ == "__main__":
    main()