"""
骨骼弧面轨迹生成器

基于 DOF（Diffused Orientation Fields）算法，在骨骼点云的弧面上
从起点到终点生成一条贴合表面的轨迹，轨迹上每个点附带局部坐标系。

核心原理：
1. 在起点和终点设置热源，求解热传导方程（标量扩散）
2. 标量场的梯度方向给出从起点到终点的表面切线方向场
3. Walk-on-Spheres 将表面的局部坐标系扩展到自由空间
4. 从起点出发，沿方向场逐步前进，每步投影回弧面，直到到达终点

输出：
- trajectory: (N, 3) 轨迹点位置
- trajectory_local_bases: (N, 3, 3) 每个点的局部坐标系
  - local_bases[i, :, 0] = 切向（沿轨迹前进方向）
  - local_bases[i, :, 1] = 侧向（表面内垂直于轨迹）
  - local_bases[i, :, 2] = 法向（表面法线方向）

使用示例：
    from diffused_fields.manifold import Pointcloud
    from arc_trajectory import ArcTrajectory

    pcloud = Pointcloud(filename="Bone.ply")
    generator = ArcTrajectory(pcloud, start_idx=100, end_idx=800)
    generator.compute()
    generator.visualize()
"""
import open3d as o3d
import os
import pickle
import time
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation as R

from diffused_fields import PointcloudScalarDiffusion, WalkOnSpheresDiffusion


class ArcTrajectory:
    """
    骨骼弧面轨迹生成器

    在骨骼点云的弧面上，从起点到终点生成一条贴合表面的轨迹。
    轨迹上每个点都有局部坐标系，可直接用于机械臂控制。
    """

    def __init__(
        self,
        pcloud,
        start_idx=None,
        end_idx=None,
        diffusion_scalar=2000,
        step_size=0.002,
        distance_to_surface=-0.001,
        num_init_steps=20,
        max_steps=5000,
        endpoint_tolerance=30.0,
        wos_batch_size=512,
        wos_max_iterations=24,
    ):
        """
        初始化弧面轨迹生成器。

        Args:
            pcloud: diffused_fields.manifold.Pointcloud 对象
            start_idx: 起点顶点索引（可通过 PointSelector 选择）
            end_idx: 终点顶点索引
            diffusion_scalar: 扩散标量，控制方向场的平滑程度
                - 值越大，方向场越平滑，能穿透更大的空洞
                - 骨骼有孔洞时建议 3000-5000
                - 简单弧面可用 1000-2000
            step_size: 每步移动距离（米），建议 0.001-0.003
            distance_to_surface: 轨迹距表面的偏移（米）
                - 负值：轨迹在表面外侧（工具在表面上方）
                - 正值：轨迹在表面内侧
                - 0：轨迹贴合表面
            num_init_steps: 初始安全偏移步数（避免从源顶点出发的不稳定性）
            max_steps: 最大步数（防止无限循环）
            endpoint_tolerance: 终点到达容差倍数（相对于 step_size）
            wos_batch_size: Walk-on-Spheres 采样批大小
            wos_max_iterations: Walk-on-Spheres 最大迭代次数
        """
        self.pcloud = pcloud
        self.start_idx = start_idx
        self.end_idx = end_idx

        # 扩散参数
        self.diffusion_scalar = diffusion_scalar
        self.step_size = step_size
        self.distance_to_surface = distance_to_surface
        self.num_init_steps = num_init_steps
        self.max_steps = max_steps
        self.endpoint_tolerance = endpoint_tolerance

        # WoS 参数
        self.wos_batch_size = wos_batch_size
        self.wos_max_iterations = wos_max_iterations

        # 结果存储
        self.trajectory = None
        self.trajectory_local_bases = None
        self.reached_end = False
        self.num_steps_taken = 0

    def compute(self):
        """
        计算弧面轨迹。

        完整流程：
        1. 验证输入
        2. 初始化标量扩散系统（计算方向场）
        3. 初始化 Walk-on-Spheres 系统（计算空间旋转场）
        4. 从起点逐步前进到终点
        5. 返回轨迹和局部坐标系

        Returns:
            self: 返回自身以支持链式调用
        """
        self._validate_inputs()
        self._initialize_diffusion()
        self._initialize_wos()
        self._rollout_trajectory()
        return self

    def _validate_inputs(self):
        """验证输入参数"""
        if self.start_idx is None or self.end_idx is None:
            raise ValueError(
                "必须指定起点和终点索引。请使用 PointSelector 选择，"
                "或在构造函数中传入 start_idx 和 end_idx。"
            )
        n = len(self.pcloud.vertices)
        if self.start_idx >= n or self.end_idx >= n:
            raise ValueError(
                f"索引超出范围: start={self.start_idx}, end={self.end_idx}, 总点数={n}"
            )
        if self.start_idx == self.end_idx:
            raise ValueError("起点和终点不能相同")

        # 确保点云有必要属性
        if not hasattr(self.pcloud, 'normals'):
            print("[预处理] 估计表面法线...")
            self.pcloud.get_normals(num_neighbors=20)

        print(f"[验证通过] 起点={self.start_idx}, 终点={self.end_idx}, 总点数={n}")

    def _initialize_diffusion(self):
        """
        初始化标量扩散系统。

        在起点设置 +1 热源，终点设置 -1 热源，
        求解热传导方程，得到从起点指向终点的方向场。
        """
        print(f"[扩散初始化] diffusion_scalar={self.diffusion_scalar}...")

        # 创建标量扩散求解器
        self.scalar_diffusion = PointcloudScalarDiffusion(
            self.pcloud,
            diffusion_scalar=self.diffusion_scalar,
            method="LU",
        )

        # 设置源顶点：起点 +1，终点 -1
        # 这样梯度方向从起点指向终点
        self.scalar_diffusion.source_vertices = [self.start_idx, self.end_idx]

        # 计算标量扩散场和局部坐标系
        # get_local_bases() 内部会：
        #   1. 设置边界条件 (set_sources)
        #   2. 预分解矩阵 (prefactor_matrices)
        #   3. 积分扩散方程 (integrate_diffusion)
        #   4. 计算梯度 (get_gradient)
        #   5. 从梯度方向构建局部坐标系 (get_bases_from_tangent_vector_and_normal)
        self.scalar_diffusion.get_local_bases()

        print(f"[扩散完成] 方向场已计算，局部坐标系已构建")

    def _initialize_wos(self):
        """
        初始化 Walk-on-Spheres 系统。

        WoS 将表面上的局部坐标系扩散到自由空间，
        使得在表面附近的任意点都能查询到一致的局部坐标系。
        """
        print(f"[WoS初始化] batch_size={self.wos_batch_size}...")

        # 收敛阈值：基于平均边长的2倍
        mean_edge_length = self.pcloud.get_mean_edge_length()
        convergence_threshold = mean_edge_length * 2

        # 边界就是点云表面
        boundaries = [self.pcloud]

        self.wos = WalkOnSpheresDiffusion(
            boundaries=boundaries,
            batch_size=self.wos_batch_size,
            max_iterations=self.wos_max_iterations,
            convergence_threshold=convergence_threshold,
        )

        print(f"[WoS完成] 收敛阈值={convergence_threshold:.6f}")

    def _rollout_trajectory(self):
        """
        轨迹展开：从起点逐步前进到终点。

        每一步：
        1. 在当前位置用 WoS 扩散得到局部坐标系
        2. 沿局部坐标系的 X 方向（方向场方向）前进一步
        3. 将位置投影回弧面（保持 distance_to_surface）
        4. 检查是否到达终点
        """
        print(f"[轨迹生成] 步长={self.step_size}, 最大步数={self.max_steps}...")

        # 获取起始位置：从起点顶点沿法线偏移
        x0 = self._get_start_position()

        # 安全偏移：从源顶点沿方向场前进几步，避免源顶点处方向场奇异
        x_current = self._apply_init_offset(x0)

        # 获取终点的3D位置（用于到达检测）
        # 注意：终点位置也需要沿法线偏移，与起点保持一致的偏移平面
        self.end_position = (
            self.pcloud.vertices[self.end_idx]
            + self.pcloud.normals[self.end_idx] * self.distance_to_surface
        )

        # 初始化轨迹存储
        positions = [x_current.copy()]
        local_bases_list = []

        # 获取初始局部坐标系
        batch_points = self.wos.get_batch_from_point(x_current)
        initial_basis, _, _ = self.wos.diffuse_rotations(batch_points)
        local_bases_list.append(initial_basis.copy())

        # 逐步前进
        start_time = time.time()
        self.reached_end = False

        for step in range(self.max_steps):
            # 1. 获取当前位置的局部坐标系
            batch_points = self.wos.get_batch_from_point(x_current)
            local_basis, _, _ = self.wos.diffuse_rotations(batch_points)

            # 2. 沿 X 方向（方向场方向）前进一步
            # local_basis[:, 0] 是沿轨迹前进的切向方向
            x_next = x_current + local_basis[:, 0] * self.step_size

            # 3. 投影回弧面，保持距离表面的偏移
            x_next = self._project_to_surface(x_next)

            # 4. 存储轨迹点和局部坐标系
            positions.append(x_next.copy())
            local_bases_list.append(local_basis.copy())

            x_current = x_next

            # 5. 检查是否到达终点
            if self._check_endpoint_reached(x_current):
                self.reached_end = True
                self.num_steps_taken = step + 1
                break

            # 进度输出
            if (step + 1) % 200 == 0:
                dist_to_end = np.linalg.norm(x_current - self.end_position)
                elapsed = time.time() - start_time
                print(
                    f"  步骤 {step + 1}/{self.max_steps}, "
                    f"距终点: {dist_to_end:.4f}, "
                    f"耗时: {elapsed:.1f}s"
                )
        else:
            self.num_steps_taken = self.max_steps
            print(f"[警告] 达到最大步数 {self.max_steps}，可能未到达终点")

        # 转换为 numpy 数组
        self.trajectory = np.array(positions)
        self.trajectory_local_bases = np.array(local_bases_list)

        # 确保轨迹方向的局部坐标系 X 轴对齐前进方向
        self._align_trajectory_frames()

        elapsed = time.time() - start_time
        print(
            f"[轨迹完成] "
            f"步数={self.num_steps_taken}, "
            f"轨迹点数={len(self.trajectory)}, "
            f"到达终点={'是' if self.reached_end else '否'}, "
            f"耗时={elapsed:.2f}s"
        )

    def _get_start_position(self):
        """获取起始位置：起点顶点沿法线偏移"""
        return (
            self.pcloud.vertices[self.start_idx]
            + self.pcloud.normals[self.start_idx] * self.distance_to_surface
        )

    def _apply_init_offset(self, x0):
        """
        初始安全偏移：沿方向场前进几步，避免源顶点处方向场的奇异性。

        在源顶点处，标量场的梯度可能不稳定（因为源顶点本身就是热源）。
        偏移几步后进入方向场稳定的区域。
        """
        if self.num_init_steps <= 0:
            return x0

        print(f"[安全偏移] 沿方向场前进 {self.num_init_steps} 步...")
        x_current = x0

        for _ in range(self.num_init_steps):
            batch_points = self.wos.get_batch_from_point(x_current)
            local_basis, _, _ = self.wos.diffuse_rotations(batch_points)
            x_current = x_current + local_basis[:, 0] * self.step_size
            x_current = self._project_to_surface(x_current)

        return x_current

    def _project_to_surface(self, position):
        """
        将位置投影到弧面，保持指定的表面距离偏移。

        使用 correct_distance_smooth 方法，通过迭代修正
        使点到表面的有符号距离等于 distance_to_surface。

        Args:
            position: 当前3D位置

        Returns:
            投影后的3D位置
        """
        projected_pos, _, _ = self.pcloud.correct_distance_smooth(
            position, self.distance_to_surface
        )
        return projected_pos

    def _check_endpoint_reached(self, x_current):
        """
        检查是否到达终点。

        条件：当前位置到终点的距离小于 step_size * endpoint_tolerance。

        Args:
            x_current: 当前3D位置

        Returns:
            bool: 是否到达终点
        """
        dist_to_end = np.linalg.norm(x_current - self.end_position)
        tolerance = self.step_size * self.endpoint_tolerance

        return dist_to_end < tolerance

    def _align_trajectory_frames(self):
        """
        对齐轨迹局部坐标系的 X 轴方向与前进方向。

        确保每个轨迹点的 local_bases[:, 0] 指向下一步的方向，
        这样局部坐标系的 X 轴始终沿轨迹前进方向。
        """
        if len(self.trajectory) < 2:
            return

        for i in range(len(self.trajectory) - 1):
            # 计算实际前进方向
            forward = self.trajectory[i + 1] - self.trajectory[i]
            forward_norm = np.linalg.norm(forward)

            if forward_norm < 1e-10:
                continue

            forward = forward / forward_norm

            # 获取当前的 X 方向
            current_x = self.trajectory_local_bases[i, :, 0]

            # 如果 X 方向与前进方向相反，翻转整个局部坐标系
            if np.dot(current_x, forward) < 0:
                self.trajectory_local_bases[i, :, 0] = -current_x
                self.trajectory_local_bases[i, :, 1] = -self.trajectory_local_bases[i, :, 1]
                # Z 轴（法线）保持不变

    def visualize(self, show_frames=True, frame_step=None, use_polyscope=True):
        """
        可视化轨迹结果。

        Args:
            show_frames: 是否显示局部坐标系
            frame_step: 坐标系显示间隔（None=自动）
            use_polyscope: 是否使用 Polyscope（否则用 Open3D）
        """
        if self.trajectory is None:
            print("[错误] 尚未计算轨迹，请先调用 compute()")
            return

        if use_polyscope:
            self._visualize_polyscope(show_frames, frame_step)
        else:
            self._visualize_open3d(show_frames, frame_step)

    def _visualize_polyscope(self, show_frames, frame_step):
        """使用 Polyscope 可视化"""
        try:
            import polyscope as ps
        except ImportError:
            print("Polyscope 未安装，回退到 Open3D")
            self._visualize_open3d(show_frames, frame_step)
            return

        try:
            from diffused_fields.visualization.plotting_ps import (
                plot_orientation_field,
                plot_point_cloud,
            )
        except ImportError:
            print("diffused_fields 可视化模块不可用，回退到 Open3D")
            self._visualize_open3d(show_frames, frame_step)
            return

        ps.init()

        # 尝试加载中文字体（解决中文显示乱码问题）
        # 注意：UI 名称已使用英文，确保即使字体加载失败也无乱码
        try:
            from font_utils import setup_chinese_font
            setup_chinese_font(font_size=15.0, verbose=False)
        except Exception:
            pass

        # 点云
        vector_length = 0.02
        vector_radius = 0.015
        point_radius = 0.002
        curve_radius = 0.002

        # 显示物体点云（带方向场）
        ps_cloud = plot_orientation_field(
            self.pcloud.vertices,
            self.pcloud.local_bases,
            name="Bone_Cloud",
            vector_length=vector_length,
            vector_radius=vector_radius,
            point_radius=point_radius,
            enable_vector=False,
            color=self.pcloud.colors if self.pcloud.colors is not None else None,
        )

        # 轨迹曲线
        ps.register_curve_network(
            "Trajectory",
            self.trajectory,
            edges="line",
            radius=curve_radius,
            color=[0.0, 0.0, 1.0],
        )

        # 轨迹局部坐标系
        if show_frames:
            if frame_step is None:
                frame_step = max(1, len(self.trajectory) // 50)

            sampled_indices = np.arange(0, len(self.trajectory), frame_step)
            sampled_positions = self.trajectory[sampled_indices]
            sampled_bases = self.trajectory_local_bases[sampled_indices]

            plot_orientation_field(
                sampled_positions,
                sampled_bases,
                name="Trajectory_Frames",
                vector_length=vector_length * 2,
                vector_radius=vector_radius * 2,
                point_radius=point_radius * 2,
                enable_x=True,
                enable_z=True,
            )

        # 标记起点和终点
        start_pos = self.pcloud.vertices[self.start_idx]
        end_pos = self.pcloud.vertices[self.end_idx]
        keypoints = np.array([start_pos, end_pos])
        ps_keypoints = ps.register_point_cloud(
            "Endpoints", keypoints, radius=0.01
        )
        kp_colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ps_keypoints.add_color_quantity("colors", kp_colors, enabled=True)

        ps.show()

    def _visualize_open3d(self, show_frames, frame_step):
        """使用 Open3D 可视化"""
        geometries = []

        # 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pcloud.vertices)
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(pcd)

        # 轨迹点
        traj_pcd = o3d.geometry.PointCloud()
        traj_pcd.points = o3d.utility.Vector3dVector(self.trajectory)
        traj_pcd.paint_uniform_color([0.0, 0.0, 1.0])
        geometries.append(traj_pcd)

        # 轨迹连线
        lines = [[i, i + 1] for i in range(len(self.trajectory) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(self.trajectory)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0.0, 0.0, 1.0])
        geometries.append(line_set)

        # 起终点标记
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        start_sphere.translate(self.pcloud.vertices[self.start_idx])
        start_sphere.compute_vertex_normals()
        start_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(start_sphere)

        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        end_sphere.translate(self.pcloud.vertices[self.end_idx])
        end_sphere.compute_vertex_normals()
        end_sphere.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(end_sphere)

        o3d.visualization.draw_geometries(
            geometries,
            window_name="骨骼弧面轨迹 (蓝=轨迹, 红=起点, 绿=终点)",
            width=1280, height=960,
        )

    def get_trajectory_poses(self):
        """
        获取轨迹的位姿序列（位置 + 姿态），可直接用于机械臂控制。

        Returns:
            positions: (N, 3) 轨迹位置
            rotations: (N, 3, 3) 旋转矩阵（从局部坐标系提取）
                旋转矩阵的列向量为：
                - 第0列：轨迹前进方向 (X)
                - 第1列：侧向 (Y)
                - 第2列：表面法线方向 (Z)
        """
        if self.trajectory is None:
            raise RuntimeError("尚未计算轨迹，请先调用 compute()")

        return self.trajectory.copy(), self.trajectory_local_bases.copy()

    def get_trajectory_quaternions(self):
        """
        获取轨迹的位姿序列（位置 + 四元数），适合某些机器人控制器。

        Returns:
            positions: (N, 3) 轨迹位置
            quaternions: (N, 4) 四元数 [w, x, y, z]
        """
        positions, rotations = self.get_trajectory_poses()

        quaternions = np.zeros((len(rotations), 4))
        for i in range(len(rotations)):
            rot = R.from_matrix(rotations[i])
            # scipy 格式 [x, y, z, w]，转换为 [w, x, y, z]
            q_scipy = rot.as_quat()
            quaternions[i] = [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]]

        return positions, quaternions

    def get_trajectory_euler(self, convention='xyz'):
        """
        获取轨迹的位姿序列（位置 + 欧拉角）。

        Args:
            convention: 欧拉角约定，如 'xyz', 'ZYX' 等

        Returns:
            positions: (N, 3) 轨迹位置
            euler_angles: (N, 3) 欧拉角（弧度）
        """
        positions, rotations = self.get_trajectory_poses()

        euler_angles = np.zeros((len(rotations), 3))
        for i in range(len(rotations)):
            rot = R.from_matrix(rotations[i])
            euler_angles[i] = rot.as_euler(convention, degrees=False)

        return positions, euler_angles

    def save_results(self, filepath=None):
        """
        保存轨迹结果到 pickle 文件。

        Args:
            filepath: 保存路径，None 则自动生成

        Returns:
            str: 实际保存路径
        """
        if self.trajectory is None:
            raise RuntimeError("尚未计算轨迹，请先调用 compute()")

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                os.path.dirname(__file__),
                "results",
                f"arc_trajectory_{timestamp}.pkl",
            )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            "trajectory": self.trajectory,
            "trajectory_local_bases": self.trajectory_local_bases,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "start_position": self.pcloud.vertices[self.start_idx],
            "end_position": self.pcloud.vertices[self.end_idx],
            "reached_end": self.reached_end,
            "num_steps_taken": self.num_steps_taken,
            "parameters": {
                "diffusion_scalar": self.diffusion_scalar,
                "step_size": self.step_size,
                "distance_to_surface": self.distance_to_surface,
                "num_init_steps": self.num_init_steps,
                "max_steps": self.max_steps,
                "endpoint_tolerance": self.endpoint_tolerance,
            },
            "pointcloud_vertices": self.pcloud.vertices,
            "pointcloud_normals": self.pcloud.normals,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"[保存完成] {filepath}")
        return filepath

    @classmethod
    def load_results(cls, filepath):
        """
        从文件加载轨迹结果。

        Args:
            filepath: 结果文件路径

        Returns:
            dict: 轨迹数据
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        print(
            f"[加载完成] 轨迹点数={len(data['trajectory'])}, "
            f"到达终点={'是' if data['reached_end'] else '否'}"
        )
        return data

    def print_summary(self):
        """打印轨迹摘要信息"""
        if self.trajectory is None:
            print("[错误] 尚未计算轨迹")
            return

        print("\n" + "=" * 60)
        print("  骨骼弧面轨迹摘要")
        print("=" * 60)
        print(f"  轨迹点数量:     {len(self.trajectory)}")
        print(f"  实际步数:        {self.num_steps_taken}")
        print(f"  到达终点:        {'是' if self.reached_end else '否'}")
        print(f"  步长:            {self.step_size * 1000:.2f} mm")
        print(f"  表面偏移:        {self.distance_to_surface * 1000:.2f} mm")
        print(f"  扩散标量:        {self.diffusion_scalar}")

        # 轨迹长度
        diffs = np.diff(self.trajectory, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(segment_lengths)
        print(f"  轨迹总长度:     {total_length * 1000:.2f} mm")
        print(f"  平均步长:        {np.mean(segment_lengths) * 1000:.4f} mm")

        # 起终点信息
        start_pos = self.pcloud.vertices[self.start_idx]
        end_pos = self.pcloud.vertices[self.end_idx]
        direct_dist = np.linalg.norm(start_pos - end_pos)
        print(f"  起终点直线距离: {direct_dist * 1000:.2f} mm")
        print(f"  弧面/直线比:    {total_length / direct_dist:.3f}")

        print(f"  局部坐标系维度: {self.trajectory_local_bases.shape}")
        print("=" * 60)
