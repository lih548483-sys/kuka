import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import struct

# 强制Matplotlib使用TkAgg后台以增强交互
import matplotlib
try:
    matplotlib.use("TkAgg")
except ImportError:
    pass # 如果没有TkAgg，回退到默认
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d

# 这里必须使用您原有的path_planner模块中的导入
# 请确保您有 path_planner.py 文件且代码正确
from path_planner import a_star_path, extract_discrete_path, smooth_path

ROOT = Path(__file__).resolve().parent
# 请确保此默认文件存在，否则运行 "--ply" 指定文件
DEFAULT_PLY = ROOT / "Bone_m_later.ply" 
OUTPUT_DIR = ROOT / "outputs"


def load_point_cloud_from_ply(path: Path) -> np.ndarray:
    """更鲁棒地从PLY文件加载点云模型。兼容ASCII和二进制格式。"""
    try:
        with path.open("rb") as handle:
            header_lines = []
            while True:
                line = handle.readline().decode("ascii", errors="ignore").strip()
                if not line:
                    break
                header_lines.append(line)
                if line == "end_header":
                    break

            format_line = next((line for line in header_lines if line.startswith("format ")), "format ascii 1.0")
            
            # 解析顶点数
            vertex_count = None
            for line in header_lines:
                if line.startswith("element vertex"):
                    parts = line.split()
                    if len(parts) >= 3:
                        vertex_count = int(parts[-1])
                        break
            if vertex_count is None:
                raise ValueError("Could not find vertex count in the PLY header.")

            # 二进制格式解析
            if "binary" in format_line:
                # 假设PLY使用单精度或双精度浮点数表示XYZ (float32/float64)
                # 医疗/工控常用float64
                payload = handle.read()
                points = []
                
                if len(payload) < vertex_count * 24: # float64 * 3 * vertex_count
                     # 如果float64不匹配，尝试 float32
                     if len(payload) >= vertex_count * 12:
                         fmt = '<3f'
                         for _ in range(vertex_count):
                             point = struct.unpack(fmt, payload[0:12])
                             points.append([float(point[0]), float(point[1]), float(point[2])])
                             payload = payload[12:]
                     else:
                         raise ValueError("PLY payload is too short for point count.")
                else:
                    fmt = '<3d'
                    for _ in range(vertex_count):
                        point = struct.unpack(fmt, payload[0:24])
                        points.append([float(point[0]), float(point[1]), float(point[2])])
                        payload = payload[24:]
                return np.asarray(points, dtype=np.float64)

            # ASCII格式解析
            lines = [line.decode("ascii", errors="ignore").strip() for line in handle.readlines()]
            points = []
            for line in lines[:vertex_count]:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue # 跳过无法解析为浮点数的行
            return np.asarray(points, dtype=np.float64)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return np.array([])
    except Exception as e:
        print(f"Error loading point cloud from PLY: {e}")
        return np.array([])


def select_ply_file() -> Path | None:
    """使用图形化界面选择PLY点云模型文件。"""
    root = tk.Tk()
    root.withdraw() # 隐藏主窗口
    root.attributes("-topmost", True) # 将对话框置顶
    path = filedialog.askopenfilename(
        title="选择点云模型文件",
        initialdir=str(DEFAULT_PLY.parent),
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


def project_to_screen(ax, point: np.ndarray) -> np.ndarray:
    """将3D空间点投影到当前Matplotlib 2D屏幕坐标上。用于匹配点击。"""
    x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
    return np.array([x2, y2], dtype=np.float64)


def set_axes_equal(ax):
    """设置3D轴具有相同的比例，并强制整个模型在一个正方体视野内完全显示。
    这解决了模型显示不全的问题。
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # 寻找三维上的最大尺寸，并设置一个包络的正方体
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def select_start_and_end(point_cloud: np.ndarray) -> tuple[int, int] | None:
    """在一个支持旋转、缩放的交互式Matplotlib 3D预览窗口内，选择切割起点和终点。
    确认选择后会自动关闭窗口并高亮标记点。
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # 绘制原始点云，使用较小的点以不遮挡细节，添加深度着色
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=point_cloud[:, 2], cmap="viridis", s=4, alpha=0.7)
    
    # 设置详细的交互指令标题
    instruction_text = (
        "选择切割路径点 (依次点击：1.起点，2.终点)\n"
        "提示：可按住鼠标左键旋转，滚动滑轮缩放。\n"
        "确认选择后，窗口会自动关闭并完成规划。"
    )
    ax.set_title(instruction_text, fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # 设置合理的初始视角
    ax.view_init(elev=20, azim=45)
    
    # **关键：设置各轴比例相等并强制完全显示边界**
    # set_axes_equal 必须在绘图之后调用以确定原始边界
    set_axes_equal(ax)
    
    fig.tight_layout()

    # 共享状态
    start_idx: int | None = None
    end_idx: int | None = None

    def on_click(event):
        nonlocal start_idx, end_idx
        if event.inaxes != ax:
            return
            
        # 1. 实时计算当前视图下所有点的屏幕投影坐标。支持视图旋转和缩放。
        click_xy = np.array([event.x, event.y], dtype=np.float64)
        screen_positions = np.array([project_to_screen(ax, point) for point in point_cloud], dtype=np.float64)
        
        # 2. 寻找点击位置最近的3D点云索引
        distances = np.linalg.norm(screen_positions - click_xy, axis=1)
        nearest_idx = int(np.argmin(distances))

        # 逻辑：第一点击是起点，第二点击是终点
        if start_idx is None:
            # 高亮起点
            start_idx = nearest_idx
            ax.scatter(
                point_cloud[start_idx, 0],
                point_cloud[start_idx, 1],
                point_cloud[start_idx, 2],
                color="lime", # 亮绿色
                s=160, # 醒目尺寸
                marker="o",
                edgecolors='white', # 添加白色边缘防混淆
                depthshade=False, # 不受深度模糊影响
                label='Selection Start'
            )
            ax.set_title("已选择起点，请点击再选择终点", fontsize=12)
            # **关键：立即刷新图形以显示高亮起点**
            fig.canvas.draw_idle() 
            
        elif end_idx is None:
            # 高亮终点
            end_idx = nearest_idx
            ax.scatter(
                point_cloud[end_idx, 0],
                point_cloud[end_idx, 1],
                point_cloud[end_idx, 2],
                color="red",
                s=160,
                marker="o",
                edgecolors='white',
                depthshade=False,
                label='Selection End'
            )
            # 确认选择后立即绘制终点并准备关闭
            fig.canvas.draw_idle() 
            plt.draw() # 强制绘图
            
            # **关键：在选择完成后自动关闭窗口**
            print("Selection complete. Closing window...")
            plt.close(fig)

    # 绑定点击事件
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    # 如果正确选择了两个点，则返回索引
    if start_idx is not None and end_idx is not None:
        return int(start_idx), int(end_idx)
    return None


def visualize_path(point_cloud: np.ndarray, path_discrete: np.ndarray, path_smooth: np.ndarray, start_idx: int, end_idx: int) -> None:
    """在3D空间内渲染结果：原始点云、高亮起点/终点、A*离散路径和平滑样条路径。供科研人员评估质量。"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # 原始点云边界设定，确保完全对齐
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=point_cloud[:, 2], cmap="viridis", s=4, alpha=0.7)
    
    # 高亮确认的起点和终点
    ax.scatter(point_cloud[start_idx, 0], point_cloud[start_idx, 1], point_cloud[start_idx, 2], color="lime", s=140, marker="o", depthshade=False, edgecolors='white', label='Start Point')
    ax.scatter(point_cloud[end_idx, 0], point_cloud[end_idx, 1], point_cloud[end_idx, 2], color="red", s=140, marker="o", depthshade=False, edgecolors='white', label='End Point')
    
    # 绘制 A* 算出的离散路径（橙色折线）
    ax.plot(path_discrete[:, 0], path_discrete[:, 1], path_discrete[:, 2], color="orange", linewidth=2.5, label='Discrete Path (A*)')
    
    # 绘制平滑后的样条路径（青色实线）
    ax.plot(path_smooth[:, 0], path_smooth[:, 1], path_smooth[:, 2], color="cyan", linewidth=3.0, label='Smoothed Path')
    
    ax.set_title("骨骼切割路径规划结果 Verification")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # 对齐坐标轴并强制边界对齐
    set_axes_equal(ax)
    
    ax.legend()
    ax.view_init(elev=20, azim=45)
    fig.tight_layout()
    plt.show()


def run_desktop_app(ply_path: Path | None = None, headless: bool = False) -> None:
    """运行骨骼切割路径规划桌面应用程序。"""
    # 1. 获取PLY点云模型文件。
    if ply_path is None:
        ply_path = select_ply_file()
    if ply_path is None or not ply_path.exists():
        # 用于保存桌面对话框的持久化状态，防止错误关闭
        # 请根据您的桌面配置，将 ROOT 文件夹权限设置正确
        root = tk.Tk()
        root.withdraw()
        if not ply_path or not ply_path.exists():
             messagebox.showinfo("已取消", "未选择有效的PLY点云模型，程序已退出。如需运行Demo，请确保根目录下有 `Bone_m_later.ply`。")
             root.destroy()
             return
        root.destroy()

    # 2. 加载点云模型，过滤非法数据点。
    point_cloud = load_point_cloud_from_ply(ply_path)
    if point_cloud.size == 0:
        raise RuntimeError("Point cloud is empty or could not be loaded.")
    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    point_cloud = point_cloud[np.isfinite(point_cloud).all(axis=1)]

    if point_cloud.shape[0] < 2:
        raise RuntimeError("Point cloud contains too few valid points for path planning.")

    # 3. 指定起点和终点，默认情况下使用图形界面交互选择。
    if not headless:
        # 在选点页面前提示。
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("步骤 1", "接下来会弹出点云窗口。\n您可以自由旋转、缩放。点击两次在模型上选择起点和终点。")
        root.destroy()
        
        selection = select_start_and_end(point_cloud)
        if selection is None:
             root = tk.Tk()
             root.withdraw()
             messagebox.showinfo("已取消", "未完成点和终点的完整选择，程序已退出。")
             root.destroy()
             return
        start_idx, end_idx = selection
    else:
        # Headless模式，通常用于自动测试。自动选择第一个点和最后一个点。
        print("Running in headless mode. Autoselect start (0) and end (-1).")
        start_idx = 0
        end_idx = len(point_cloud) - 1

    # 4. 运行 A* 流水线：图路径搜索 -> 坐标提取 -> 样条平滑。
    path_indices = a_star_path(point_cloud, start_idx, end_idx, k_neighbors=15)
    path_discrete = extract_discrete_path(point_cloud, path_indices)
    
    # 平滑分辨率设定 (单位同点云，0.5 毫米/单位)
    path_smooth = smooth_path(path_discrete, spline_res=0.5)

    # 5. 保存规划结果数据文件。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUTPUT_DIR / "path_discrete.csv", path_discrete, delimiter=",", fmt="%.8f")
    np.savetxt(OUTPUT_DIR / "path_smooth.csv", path_smooth, delimiter=",", fmt="%.8f")
    np.savetxt(OUTPUT_DIR / "selection.csv", np.array([[start_idx, end_idx]], dtype=np.int64), delimiter=",", fmt="%d")

    # 6. 核心运行数据确认与控制台输出。用于 KUKA 机械臂指令评估。
    print(f"Selected start index (original cloud): {start_idx}")
    print(f"Selected end index (original cloud): {end_idx}")
    print(f"Discrete path points (A* control points): {path_discrete.shape[0]}")
    print(f"Smoothed path points (interpolated path points): {path_smooth.shape[0]}")
    print(f"Saved outputs to: {OUTPUT_DIR}")

    # 7. 科研人员最终可视化验证结果（非 headless 模式）。
    if not headless:
        visualize_path(point_cloud, path_discrete, path_smooth, start_idx, end_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive bone-cutting path planner for robot verification")
    parser.add_argument("--ply", type=str, default=None, help="Optional path to a PLY bone model file")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows or prompting for user selection (autoselects endpoints)")
    args = parser.parse_args()

    ply_path = Path(args.ply).expanduser().resolve() if args.ply else None
    
    try:
        run_desktop_app(ply_path=ply_path, headless=args.headless)
    except RuntimeError as e:
        # 这里捕捉程序内的运行异常并显示。请根据桌面配置，将对话框状态调整正确
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("运行出错", str(e))
        root.destroy()
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("未知错误", f"发生未预期的错误: {e}")
        root.destroy()