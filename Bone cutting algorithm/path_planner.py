import heapq
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree


def _heuristic(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """
    计算 A* 算法的启发式函数（Heuristic Function）。
    采用欧几里得距离（L2 范数），在欧氏空间中满足单调性（Monotonicity）与可接受性（Admissibility）。
    """
    return float(np.linalg.norm(point_a - point_b))


def reconstruct_path(came_from: np.ndarray, idx_start: int, idx_end: int) -> np.ndarray:
    """
    从 A* 的前驱节点记录数组中逆向回溯，重构出从起点到终点的最优索引路径。
    
    参数:
        came_from: 一维数组，记录每个节点在最短路径上的前驱节点索引。
        idx_start: 起点索引。
        idx_end: 终点索引。
    """
    if idx_end < 0 or idx_start < 0:
        raise ValueError("Start and end indices must be non-negative.")

    path_indices = []
    current_idx = idx_end
    
    # 从终点开始逆向回溯，直到到达起点
    while current_idx != idx_start:
        # 如果遇到未被探索的节点（初始化为 -1），说明回溯链断裂，图结构异常
        if current_idx < 0:
            raise RuntimeError("No valid path could be reconstructed from the provided point cloud.")
        path_indices.append(int(current_idx))
        current_idx = int(came_from[current_idx])  # 移向当前节点的前驱

    path_indices.append(int(idx_start))  # 加入起点
    path_indices.reverse()              # 反转列表，使其满足起点到终点的顺序
    return np.asarray(path_indices, dtype=np.int64)


def a_star_path(point_cloud: np.ndarray, idx_start: int, idx_end: int, k_neighbors: int = 15) -> np.ndarray:
    """
    在点云构建的 K 近邻（KNN）稀疏图上运行 A* 算法，寻找两点间的最短路径。
    对于真实点云，使用无向邻接图并显式保留最近邻边，提升在密集采样点云上的连通性。
    """
    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError("point_cloud must be an array of shape (N, 3).")

    n_points = point_cloud.shape[0]
    if not (0 <= idx_start < n_points and 0 <= idx_end < n_points):
        raise ValueError("idx_start and idx_end must be valid indices in the point cloud.")
    if idx_start == idx_end:
        return np.array([idx_start], dtype=np.int64)

    k_neighbors = max(1, min(int(k_neighbors), n_points))
    tree = KDTree(point_cloud)

    query_k = min(k_neighbors + 1, n_points)
    distances, neighbor_indices = tree.query(point_cloud, k=query_k)
    if np.isscalar(neighbor_indices):
        neighbor_indices = np.array([[int(neighbor_indices)]], dtype=np.int64)
        distances = np.array([[float(distances)]], dtype=np.float64)
    else:
        neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
        distances = np.asarray(distances, dtype=np.float64)

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_points)]
    for node_idx in range(n_points):
        for neighbor_idx, edge_weight in zip(neighbor_indices[node_idx], distances[node_idx]):
            if int(neighbor_idx) == node_idx:
                continue
            adjacency[node_idx].append((int(neighbor_idx), float(edge_weight)))
            adjacency[int(neighbor_idx)].append((node_idx, float(edge_weight)))

    open_set: list[tuple[float, int]] = []
    g_score = np.full(n_points, np.inf, dtype=np.float64)
    f_score = np.full(n_points, np.inf, dtype=np.float64)
    came_from = np.full(n_points, -1, dtype=np.int64)

    g_score[idx_start] = 0.0
    f_score[idx_start] = _heuristic(point_cloud[idx_start], point_cloud[idx_end])
    heapq.heappush(open_set, (f_score[idx_start], idx_start))

    while open_set:
        current_f_score, current_idx = heapq.heappop(open_set)
        if current_f_score > f_score[current_idx] + 1e-12:
            continue

        if current_idx == idx_end:
            return reconstruct_path(came_from, idx_start, idx_end)

        for neighbor_idx, edge_weight in adjacency[current_idx]:
            tentative_g_score = g_score[current_idx] + float(edge_weight)
            if tentative_g_score < g_score[neighbor_idx]:
                came_from[neighbor_idx] = current_idx
                g_score[neighbor_idx] = tentative_g_score
                f_score[neighbor_idx] = tentative_g_score + _heuristic(point_cloud[neighbor_idx], point_cloud[idx_end])
                heapq.heappush(open_set, (f_score[neighbor_idx], neighbor_idx))

    raise RuntimeError("No path was found between the specified start and end indices.")


def extract_discrete_path(point_cloud: np.ndarray, path_indices: np.ndarray) -> np.ndarray:
    """
    根据索引数组从原始点云中提取出离散的路径点三维坐标。
    """
    return point_cloud[path_indices]


def smooth_path(path_points: np.ndarray, spline_res: float) -> np.ndarray:
    """
    利用自然三次样条曲线（Natural Cubic Spline）对离散路径进行参数化平滑与等间距重采样。
    
    参数:
        path_points: 形状为 (M, 3) 的离散路径点坐标。
        spline_res: 重采样步长（单位与点云空间坐标一致）。
    """
    path_points = np.asarray(path_points, dtype=np.float64)
    if path_points.ndim != 2 or path_points.shape[1] != 3:
        raise ValueError("path_points must be an array with shape (M, 3).")

    # 拓扑约束：小于等于2个点无法构建三次样条（至少需要3个点确定二次以上多项式）
    if path_points.shape[0] <= 2:
        return path_points

    # 关键步骤：过滤由于图搜索中回溯可能引入的连续重复点（零距离步长会导致样条自变量非严格单调递增，进而引发插值矩阵奇异）
    keep_mask = np.ones(path_points.shape[0], dtype=bool)
    keep_mask[1:] = np.any(np.linalg.norm(np.diff(path_points, axis=0), axis=1) > 1e-12, axis=0)
    path_points = path_points[keep_mask]

    if path_points.shape[0] <= 2:
        return path_points

    # 以路径的累计弦长（Cumulative Chord Length）作为样条函数的自变量 s
    segment_lengths = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative_lengths[-1]

    if total_length <= 1e-12:
        return path_points

    # 构建重采样的自变量序列 sample_s
    sample_s = np.arange(0.0, total_length + 0.5 * spline_res, spline_res, dtype=np.float64)
    
    # 强制边界对齐：确保重采样序列的终点精确落在 total_length 上
    if sample_s.size == 0 or sample_s[-1] < total_length - 1e-12:
        sample_s = np.append(sample_s, total_length)
    sample_s[0] = 0.0
    sample_s[-1] = total_length

    # 对 X, Y, Z 三个空间维度分别建立基于累计弦长 s 的三次独立样条函数
    # bc_type="natural" 表示自然边界条件（两端点的二阶导数为 0）
    splines = [CubicSpline(cumulative_lengths, path_points[:, dim], bc_type="natural") for dim in range(3)]
    
    # 沿自变量采样并重新拼接回 (K, 3) 的三维轨迹
    smoothed = np.column_stack([spline(sample_s) for spline in splines])
    return smoothed


def run_demo(output_dir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    算法演练 Demo 主函数：生成合成的三维螺旋点云，测试 A* 路径搜索与样条平滑效果。
    """
    rng = np.random.default_rng(42)

    # 1. 生成带有高斯噪声的三维螺旋线点云作为模拟环境
    t = np.linspace(0.0, 4.0 * np.pi, 500)
    base_curve = np.column_stack(
        [
            10.0 + 2.5 * np.sin(t),
            5.0 + 0.8 * np.cos(2.0 * t) + 0.4 * np.sin(t),
            1.0 + 0.15 * t,
        ]
    )
    point_cloud = base_curve + rng.normal(0.0, 0.08, size=base_curve.shape)

    # 2. 选取理论曲线的首尾点，在噪声点云中通过近邻搜索定位 A* 的起点与终点索引
    start_pos = base_curve[0]
    end_pos = base_curve[-1]
    idx_start = int(np.argmin(np.linalg.norm(point_cloud - start_pos, axis=1)))
    idx_end = int(np.argmin(np.linalg.norm(point_cloud - end_pos, axis=1)))

    # 3. 运行流水线：图路径搜索 -> 坐标提取 -> 样条平滑
    path_indices = a_star_path(point_cloud, idx_start, idx_end, k_neighbors=15)
    path_discrete = extract_discrete_path(point_cloud, path_indices)
    path_smooth = smooth_path(path_discrete, spline_res=0.5)

    # 4. 维护输出目录并持久化存储数据为 CSV 文件
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent / "outputs")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path / "path_discrete.csv", path_discrete, delimiter=",", fmt="%.8f")
    np.savetxt(output_path / "path_smooth.csv", path_smooth, delimiter=",", fmt="%.8f")

    print(f"Discrete path points: {path_discrete.shape[0]}")
    print(f"Smoothed path points: {path_smooth.shape[0]}")
    print(f"Saved outputs to: {output_path}")
    return path_discrete, path_smooth


if __name__ == "__main__":
    run_demo()