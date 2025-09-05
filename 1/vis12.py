import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib import cm

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']


# 常量定义
MISSILE_SPEED = 300  # m/s
DRONE_SPEED_RANGE = (70, 140)  # m/s
SMOKE_DESCENT_RATE = 3  # m/s
SMOKE_EFFECTIVE_RADIUS = 10  # m
SMOKE_EFFECTIVE_TIME = 20  # s
GRAVITY = 9.8  # m/s^2

# 目标和初始位置信息
FAKE_TARGET = np.array([0, 0, 0])
REAL_TARGET_CENTER = np.array([0, 200, 5])  # 圆柱体中心点
REAL_TARGET_RADIUS = 7  # m
REAL_TARGET_HEIGHT = 10  # m

MISSILE_M1_INIT = np.array([20000, 0, 2000])
DRONE_FY1_INIT = np.array([17800, 0, 1800])

def normalize(v):
    """归一化向量"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def missile_trajectory(init_pos, target_pos, speed, t):
    """计算导弹在时间t的位置"""
    direction = normalize(target_pos - init_pos)
    return init_pos + direction * speed * t

def drone_trajectory(init_pos, direction_deg, speed, t):
    """计算无人机在时间t的位置"""
    # 将角度转换为弧度
    direction_rad = np.radians(direction_deg)
    # 计算方向向量 (x轴正方向为0度，逆时针为正)
    direction_vector = np.array([np.cos(direction_rad), np.sin(direction_rad), 0])
    return init_pos + direction_vector * speed * t

def smoke_trajectory(release_pos, release_time,direction_deg,speed, t):
    """计算烟幕弹在时间t的位置（考虑重力）"""
    # 将角度转换为弧度
    direction_rad = np.radians(direction_deg)
    # 计算方向向量 (x轴正方向为0度，逆时针为正)
    direction_vector = np.array([np.cos(direction_rad), np.sin(direction_rad), 0])
    if t < release_time:
        return None  # 未释放
    dt = t - release_time
    # 烟幕弹初始速度与无人机相同
    return release_pos + np.array([0, 0, -0.5 * GRAVITY * dt**2]) + direction_vector * speed * dt

def smoke_cloud_trajectory(detonation_pos, detonation_time, t):
    """计算烟幕云团在时间t的位置"""
    if t < detonation_time:
        return None  # 未爆炸
    dt = t - detonation_time
    return detonation_pos + np.array([0, 0, -SMOKE_DESCENT_RATE * dt]) 

def distance_point_to_line(point, line_start, line_end):
    """计算点到直线的距离"""
    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)
    
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    
    t = np.dot(line_unitvec, point_vec_scaled)
    t = max(0, min(1, t))  # 确保t在[0,1]范围内
    
    nearest = line_start + t * line_vec
    return np.linalg.norm(point - nearest)


def is_target_in_shadow_cone(missile_pos, cloud_pos, target_pos, target_radius, target_height):
    """
    检查目标是否在烟幕云团投射的阴影锥体内
    
    参数:
    missile_pos: 导弹位置
    cloud_pos: 烟幕云团中心位置
    target_pos: 目标底面中心位置
    target_radius: 目标半径
    target_height: 目标高度
    
    返回:
    bool: 如果目标完全在阴影锥体内，则返回True
    """
    if missile_pos[0]-10 < cloud_pos[0]:
        return False
    # 计算烟幕云团到导弹的方向向量
    direction = missile_pos - cloud_pos
    distance_missile_cloud = np.linalg.norm(direction)
    
    if distance_missile_cloud < SMOKE_EFFECTIVE_RADIUS:
        # 如果导弹在云团内，肯定遮蔽成功
        return True
    
    # 单位方向向量
    unit_direction = direction / distance_missile_cloud
    

    # 计算从导弹到烟幕球体的切线所形成的锥角
    # sin(theta) = r/d，其中r是球体半径，d是导弹到球体中心的距离
    sin_theta = min(1.0, SMOKE_EFFECTIVE_RADIUS / distance_missile_cloud)
    cos_theta = np.sqrt(1 - sin_theta**2)
    
    # 检查目标圆柱体的每个关键点是否在阴影锥体内
    # 关键点：底面圆周上的点和顶面圆周上的点
    num_points = 12  # 圆周上的采样点数
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    # 底面圆周上的点
    bottom_points = []
    for angle in angles:
        x = target_pos[0] + target_radius * np.cos(angle)
        y = target_pos[1] + target_radius * np.sin(angle)
        z = target_pos[2]
        bottom_points.append([x, y, z])
    
    # 顶面圆周上的点
    top_points = []
    for angle in angles:
        x = target_pos[0] + target_radius * np.cos(angle)
        y = target_pos[1] + target_radius * np.sin(angle)
        z = target_pos[2] + target_height
        top_points.append([x, y, z])
    
    # 合并所有点
    all_points = np.vstack([bottom_points, top_points])
    
    # 检查每个点是否在阴影锥体内
    for point in all_points:
        # 从导弹到目标点的向量
        missile_to_point = point - missile_pos
        distance = np.linalg.norm(missile_to_point)
        
        if distance < 1e-10:  # 防止除以零
            continue
            
        # 计算该向量与导弹到云团方向的夹角余弦值
        cos_angle = np.dot(missile_to_point, unit_direction) / distance
        # 取绝对值
        cos_angle = abs(cos_angle)
        cos_theta = abs(cos_theta)

        # 如果夹角余弦值小于锥角余弦值，说明该点不在阴影锥体内
        if cos_angle < cos_theta:

            #打印相关的全部信息
            if distance_missile_cloud < 700:
                print("导弹位置:",missile_pos)
                print("云团位置:",cloud_pos)
                print("cos:",cos_angle,cos_theta)
                #print("目标点不在阴影锥体内:", point)
            return False

    # 所有点都在阴影锥体内，目标被完全遮蔽
    return True


# def is_target_in_shadow_cone(missile_pos, cloud_pos, target_pos, target_radius, target_height):
#     """
#     检查目标是否被烟幕云团遮蔽
    
#     新逻辑：计算目标点和导弹构成的直线到云团中心点距离是否小于10m
    
#     参数:
#     missile_pos: 导弹位置
#     cloud_pos: 烟幕云团中心位置
#     target_pos: 目标底面中心位置
#     target_radius: 目标半径
#     target_height: 目标高度
    
#     返回:
#     bool: 如果目标被完全遮蔽，则返回True
#     """

#     if missile_pos[0]-10 < cloud_pos[0]:
#         return False

#     # 计算目标上的关键点
#     num_points = 12  # 圆周上的采样点数
#     angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
#     # 底面圆周上的点
#     bottom_points = []
#     for angle in angles:
#         x = target_pos[0] + target_radius * np.cos(angle)
#         y = target_pos[1] + target_radius * np.sin(angle)
#         z = target_pos[2]
#         bottom_points.append([x, y, z])
    
#     # 顶面圆周上的点
#     top_points = []
#     for angle in angles:
#         x = target_pos[0] + target_radius * np.cos(angle)
#         y = target_pos[1] + target_radius * np.sin(angle)
#         z = target_pos[2] + target_height
#         top_points.append([x, y, z])
    
#     # 合并所有点
#     all_points = np.vstack([bottom_points, top_points])
    
#     # 检查每个点是否被遮蔽
#     for point in all_points:
#         # 计算点到导弹的直线
#         line_dir = point - missile_pos
#         line_length = np.linalg.norm(line_dir)
#         line_dir_normalized = line_dir / line_length
        
#         # 计算云团中心到直线的距离
#         # 公式: d = ||(cloud_pos - missile_pos) × line_dir_normalized|| / ||line_dir_normalized||
#         vector_to_cloud = cloud_pos - missile_pos
#         distance_to_line = np.linalg.norm(np.cross(vector_to_cloud, line_dir_normalized))
        
#         # 检查云团中心在直线上的投影点是否在导弹和目标点之间
#         projection_length = np.dot(vector_to_cloud, line_dir_normalized)
#         is_between = (projection_length > 0) and (projection_length < line_length)
        
#         # 如果直线与云团的距离大于有效半径，或者云团不在导弹和目标点之间，则该点没有被遮蔽
#         if distance_to_line > SMOKE_EFFECTIVE_RADIUS or not is_between:
#             return False
    
#     # 所有点都被遮蔽
#     return True


def calculate_shielding_effectiveness(missile_pos, cloud_pos, cloud_start_time, current_time):
    """
    计算烟幕对导弹的遮蔽效果
    
    参数:
    missile_pos: 导弹位置
    cloud_pos: 烟幕云团中心位置
    cloud_start_time: 烟幕云团形成时间
    current_time: 当前时间
    
    返回:
    int: 1表示有效遮蔽，0表示无效遮蔽
    """
    # 检查烟幕云团是否有效
    if cloud_pos is None or current_time - cloud_start_time > SMOKE_EFFECTIVE_TIME:
        return 0  # 未形成云团或云团已失效
    
    # 检查真目标是否在阴影锥体内
    target_in_shadow = is_target_in_shadow_cone(
        missile_pos, 
        cloud_pos, 
        REAL_TARGET_CENTER, 
        REAL_TARGET_RADIUS, 
        REAL_TARGET_HEIGHT
    )
    
    return 1 if target_in_shadow else 0



def visualize_problem1():
    """可视化问题1的解决方案"""
    # 问题1参数
    drone_speed = 120  # m/s
    # 计算无人机方向 (朝向假目标)
    direction_vec = FAKE_TARGET - DRONE_FY1_INIT
    direction_vec_xy = np.array([direction_vec[0], direction_vec[1], 0])

    print("无人机方向：",direction_vec_xy)
    # 计算与x轴正方向的夹角（度数）
    angle = np.degrees(np.arctan2(direction_vec_xy[1], direction_vec_xy[0]))
    if angle < 0:
        angle += 360
    
    # 1.5秒后投放，3.6秒后起爆
    release_time = 1.5
    detonation_time = release_time + 3.6
    
    # 计算投放点
    drone_release_pos = drone_trajectory(DRONE_FY1_INIT, angle, drone_speed, release_time)
    # 打印投放点坐标
    print(f"问题1: 投放点坐标: ({drone_release_pos[0]:.2f}, {drone_release_pos[1]:.2f}, {drone_release_pos[2]:.2f})")


    # 计算起爆点
    smoke_bomb_pos = smoke_trajectory(drone_release_pos, release_time,angle, drone_speed, detonation_time)
    
    # 打印起爆点坐标
    print(f"问题1: 起爆点坐标: ({smoke_bomb_pos[0]:.2f}, {smoke_bomb_pos[1]:.2f}, {smoke_bomb_pos[2]:.2f})")


    # 模拟导弹和烟幕的轨迹
    simulation_time = np.linspace(0, 15, 100000)
    missile_positions = []
    drone_positions = []
    smoke_bomb_positions = []
    smoke_cloud_positions = []
    shielding_effectiveness = []
    
    for t in simulation_time:
        # 导弹位置
        missile_pos = missile_trajectory(MISSILE_M1_INIT, FAKE_TARGET, MISSILE_SPEED, t)
        missile_positions.append(missile_pos)
        
        # 无人机位置
        drone_pos = drone_trajectory(DRONE_FY1_INIT, angle, drone_speed, t)
        drone_positions.append(drone_pos)
        
        # 烟幕弹位置
        if t >= release_time and t < detonation_time:
            bomb_pos = smoke_trajectory(drone_release_pos, release_time,angle, drone_speed, t)
            smoke_bomb_positions.append(bomb_pos)
        
        # 烟幕云团位置
        if t >= detonation_time:
            cloud_pos = smoke_cloud_trajectory(smoke_bomb_pos, detonation_time, t)
            smoke_cloud_positions.append((t, cloud_pos))
            #visualize_shadow_cone(missile_pos, cloud_pos)  # 可视化阴影锥体
            # 计算有效性
            effectiveness = calculate_shielding_effectiveness(
                missile_pos, cloud_pos, detonation_time, t)
            shielding_effectiveness.append((t, effectiveness))
    
    # 计算有效遮蔽时长
    effective_times = [(t, eff) for t, eff in shielding_effectiveness if eff > 0]
    if effective_times:
        start_time = effective_times[0][0]
        end_time = effective_times[-1][0]
        print(f"问题1: 有效遮蔽时间段: {start_time:.2f} 秒 到 {end_time:.2f} 秒")
        effective_duration = end_time - start_time
        print(f"问题1: 有效遮蔽时长约为 {effective_duration:.2f} 秒")
    else:
        print("问题1: 无有效遮蔽")
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制假目标
    ax.scatter([0], [0], [0], color='red', s=100, label='假目标')
    
    # 绘制真目标（圆柱体）
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, REAL_TARGET_HEIGHT, 10)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_cylinder = REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * np.cos(theta_grid)
    y_cylinder = REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * np.sin(theta_grid)
    ax.plot_surface(x_cylinder, y_cylinder, z_grid, alpha=0.5, color='blue')
    
    # 绘制导弹轨迹
    missile_positions = np.array(missile_positions)
    ax.plot(missile_positions[:, 0], missile_positions[:, 1], missile_positions[:, 2], 
            color='red', label='导弹M1轨迹')
    ax.scatter(MISSILE_M1_INIT[0], MISSILE_M1_INIT[1], MISSILE_M1_INIT[2], 
               color='darkred', label='导弹M1初始位置')
    
    # 绘制无人机轨迹
    drone_positions = np.array(drone_positions)
    ax.plot(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2], 
            color='green', label='无人机FY1轨迹')
    ax.scatter(DRONE_FY1_INIT[0], DRONE_FY1_INIT[1], DRONE_FY1_INIT[2], 
               color='darkgreen', label='无人机FY1初始位置')
    
    # 绘制烟幕弹投放点
    ax.scatter(drone_release_pos[0], drone_release_pos[1], drone_release_pos[2], 
               color='orange', s=80, label='烟幕弹投放点')
    
    # 绘制烟幕弹起爆点
    ax.scatter(smoke_bomb_pos[0], smoke_bomb_pos[1], smoke_bomb_pos[2], 
               color='purple', s=80, label='烟幕弹起爆点')
    
    # 绘制有效烟幕区域（只绘制特定时间点的）
    if smoke_cloud_positions:
        # 每5个时间点绘制一次云团
        for i in range(0, len(smoke_cloud_positions), 50):
            t, cloud_pos = smoke_cloud_positions[i]
            if t - detonation_time <= SMOKE_EFFECTIVE_TIME:  # 只绘制有效期内的云团
                # 绘制球体表示烟幕云团
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = cloud_pos[0] + SMOKE_EFFECTIVE_RADIUS * np.cos(u) * np.sin(v)
                y = cloud_pos[1] + SMOKE_EFFECTIVE_RADIUS * np.sin(u) * np.sin(v)
                z = cloud_pos[2] + SMOKE_EFFECTIVE_RADIUS * np.cos(v)
                ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)
        # 选择一个关键时间点进行阴影锥体可视化
        key_time_index = len(smoke_cloud_positions) // 2
        key_time, key_cloud_pos = smoke_cloud_positions[key_time_index]
        key_missile_pos = missile_trajectory(MISSILE_M1_INIT, FAKE_TARGET, MISSILE_SPEED, key_time)
        
        # 可视化阴影锥体
        #visualize_shadow_cone(key_missile_pos, key_cloud_pos, ax)
    
    # 设置坐标轴范围和标签
    ax.set_xlim([16000, 20000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([1000, 3000])
    ax.set_xlabel('X轴 (m)')
    ax.set_ylabel('Y轴 (m)')
    ax.set_zlabel('Z轴 (m)')
    ax.set_title('问题1: 烟幕干扰弹投放策略可视化')
    
    # 添加图例
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('problem1_visualization.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 运行问题1的可视化
    visualize_problem1()
    