"""
骨骼点云交互式选点工具

提供两种选点方式：
1. Open3D 交互式选点：在可视化窗口中用 Shift+左键点击选择起点和终点
2. 自动选点：沿指定轴自动找到两端极值点
3. 索引指定：直接传入顶点索引

使用方法：
    selector = PointSelector(pcloud)
    start_idx, end_idx = selector.select_interactive()   # 交互式
    start_idx, end_idx = selector.select_auto(axis=0)     # 自动沿X轴
    start_idx, end_idx = selector.select_by_index(100, 800)  # 指定索引
"""

import numpy as np
import open3d as o3d


class PointSelector:
    """骨骼点云上的起终点选择器"""

    def __init__(self, pcloud):
        """
        Args:
            pcloud: diffused_fields.manifold.Pointcloud 对象
        """
        self.pcloud = pcloud
        self.start_idx = None
        self.end_idx = None

    def select_interactive(self):
        """
        使用 Open3D 交互式选点。

        操作说明：
        - Shift + 左键点击：添加选中的点
        - 选好2个点后关闭窗口即可

        Returns:
            (start_idx, end_idx): 起点和终点的顶点索引
        """
        print("=" * 60)
        print("  交互式选点模式")
        print("  操作：Shift + 左键点击选择点")
        print("  请选择2个点：第1个为起点，第2个为终点")
        print("  选好后关闭窗口")
        print("=" * 60)

        # 创建可视化窗口
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="选择起点和终点 (Shift+左键点击)", width=1024, height=768)

        # 注册点云
        vis.add_geometry(self.pcloud.pcd)

        # 设置渲染选项
        opt = vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.array([0.1, 0.1, 0.1])

        # 运行交互
        vis.run()
        vis.destroy_window()

        # 获取选中的点索引
        picked_indices = vis.get_picked_points()

        if len(picked_indices) < 2:
            print(f"[警告] 只选了 {len(picked_indices)} 个点，需要2个。回退到自动模式。")
            return self.select_auto(axis=0)

        self.start_idx = int(picked_indices[0])
        self.end_idx = int(picked_indices[1])

        self._print_selection_info()

        return self.start_idx, self.end_idx

    def select_auto(self, axis=0):
        """
        自动选择：沿指定轴找到两端极值点。

        Args:
            axis: 主轴方向 (0=X, 1=Y, 2=Z)

        Returns:
            (start_idx, end_idx): 起点和终点的顶点索引
        """
        axis_names = {0: "X", 1: "Y", 2: "Z"}
        print(f"[自动选点] 沿 {axis_names[axis]} 轴寻找两端极值点...")

        vertices = self.pcloud.vertices

        # 沿指定轴找到最小和最大值的点
        min_idx = int(np.argmin(vertices[:, axis]))
        max_idx = int(np.argmax(vertices[:, axis]))

        self.start_idx = min_idx
        self.end_idx = max_idx

        self._print_selection_info()

        return self.start_idx, self.end_idx

    def select_by_index(self, start_idx, end_idx):
        """
        直接指定起点和终点的索引。

        Args:
            start_idx: 起点顶点索引
            end_idx: 终点顶点索引

        Returns:
            (start_idx, end_idx): 起点和终点的顶点索引
        """
        n = len(self.pcloud.vertices)
        if start_idx >= n or end_idx >= n:
            raise ValueError(f"索引超出范围: start={start_idx}, end={end_idx}, 总点数={n}")

        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)

        self._print_selection_info()

        return self.start_idx, self.end_idx

    def select_by_position(self, start_pos, end_pos):
        """
        通过3D坐标近似选择最近的点。

        Args:
            start_pos: 起点3D坐标 [x, y, z]
            end_pos: 终点3D坐标 [x, y, z]

        Returns:
            (start_idx, end_idx): 起点和终点的顶点索引
        """
        if not hasattr(self.pcloud, 'kd_tree'):
            self.pcloud.get_kd_tree()

        _, start_idx = self.pcloud.kd_tree.query(np.array(start_pos), k=1)
        _, end_idx = self.pcloud.kd_tree.query(np.array(end_pos), k=1)

        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)

        self._print_selection_info()

        return self.start_idx, self.end_idx

    def _print_selection_info(self):
        """打印选中点的信息"""
        start_pos = self.pcloud.vertices[self.start_idx]
        end_pos = self.pcloud.vertices[self.end_idx]
        dist = np.linalg.norm(start_pos - end_pos)

        print(f"  起点索引: {self.start_idx}, 坐标: [{start_pos[0]:.4f}, {start_pos[1]:.4f}, {start_pos[2]:.4f}]")
        print(f"  终点索引: {self.end_idx}, 坐标: [{end_pos[0]:.4f}, {end_pos[1]:.4f}, {end_pos[2]:.4f}]")
        print(f"  两点欧氏距离: {dist:.4f}")

    def visualize_selection(self):
        """可视化当前选中的起点和终点"""
        if self.start_idx is None or self.end_idx is None:
            print("[错误] 尚未选择起点和终点")
            return

        # 创建点云副本用于可视化
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pcloud.vertices)
        pcd.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色

        # 标记起点（红色）和终点（绿色）
        colors = np.asarray(pcd.colors)
        colors[self.start_idx] = [1.0, 0.0, 0.0]
        colors[self.end_idx] = [0.0, 1.0, 0.0]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 创建连线
        start_pos = self.pcloud.vertices[self.start_idx]
        end_pos = self.pcloud.vertices[self.end_idx]
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([start_pos, end_pos])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色连线

        # 起终点球体标记
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        start_sphere.translate(start_pos)
        start_sphere.paint_uniform_color([1.0, 0.0, 0.0])

        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        end_sphere.translate(end_pos)
        end_sphere.paint_uniform_color([0.0, 1.0, 0.0])

        print("  红色 = 起点, 绿色 = 终点, 关闭窗口继续...")
        o3d.visualization.draw_geometries(
            [pcd, start_sphere, end_sphere],
            window_name="起点(红) 终点(绿)",
            width=1024, height=768
        )
