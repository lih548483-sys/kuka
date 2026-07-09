import sys
from diffused_fields.manifold import Pointcloud
from diffused_fields_robotics.local_action_primitives.action_primitives import Peeling

# 1. 顺应框架 API 规范，仅传入文件名
filename = "Bone_m_later.ply"

try:
    # 底层框架会自动前往 diffused_fields/data/pointclouds/ 目录下加载该文件
    pcloud = Pointcloud(filename=filename)
    
    # 实例化剥皮控制器
    controller = Peeling(pcloud)
    
    # 运行算法与生成轨迹
    print("正在计算扩散方向场并生成机械臂剥皮轨迹...")
    controller.run()
    
    # 结果验证
    print(f"轨迹计算完毕。生成的轨迹点数量: {len(controller.trajectory)}")
    controller.visualize_trajectory(show_tool=False, num_samples=10)
    
except Exception as e:
    print(f"算法执行期发生致命异常: {e}")
    sys.exit(1)