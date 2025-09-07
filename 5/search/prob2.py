import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib import cm

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib import font_manager
import time

import pandas as pd
import os
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

# 导弹初始位置
MISSILE_POSITIONS = [
    np.array([20000, 0, 2000]),    # M1
    np.array([19000, 600, 2100]),  # M2
    np.array([18000, -600, 1900])  # M3
]

# 无人机初始位置
DRONE_POSITIONS = [
    np.array([17800, 0, 1800]),     # FY1
    np.array([12000, 1400, 1400]),  # FY2
    np.array([6000, -3000, 700]),   # FY3
    np.array([11000, 2000, 1800]),  # FY4
    np.array([13000, -2000, 1300])  # FY5
]


MISSILE_M1_INIT = np.array([20000, 0, 2000])
DRONE_FY1_INIT = np.array([17800, 0, 1800])

def normalize(v):
    """归一化向量"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def missile_time_to_target(init_pos, target_pos, speed):
    """计算导弹到达目标的时间"""
    distance = np.linalg.norm(target_pos - init_pos)
    return distance / speed

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
    if missile_pos[0]+10 < cloud_pos[0]:
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
        if cos_angle > 0:
            # 如果夹角大于90度，说明点在导弹的背后，不在阴影锥体内
            return False
        # 取绝对值
        cos_angle = abs(cos_angle)
        cos_theta = abs(cos_theta)

        # 如果夹角余弦值小于锥角余弦值，说明该点不在阴影锥体内
        if cos_angle < cos_theta:

            #打印相关的全部信息
            # if distance_missile_cloud < 700:
            #     print("导弹位置:",missile_pos)
            #     print("云团位置:",cloud_pos)
            #     print("cos:",cos_angle,cos_theta)
            #     #print("目标点不在阴影锥体内:", point)
            return False

    # 所有点都在阴影锥体内，目标被完全遮蔽
    return True


def calculate_shielding_effectiveness(missile_pos, cloud_pos, cloud_start_time, current_time):
    """计算烟幕对导弹的遮蔽效果"""
    if cloud_pos is None or current_time - cloud_start_time > SMOKE_EFFECTIVE_TIME:
        return 0  # 未形成云团或云团已失效
    
    # 检查真目标是否被烟幕遮蔽
    target_shielded = is_target_in_shadow_cone(
        missile_pos, cloud_pos, REAL_TARGET_CENTER, REAL_TARGET_RADIUS, REAL_TARGET_HEIGHT
    )
    
    return 1 if target_shielded else 0

def calculate_effective_shielding_time(drone_idx,missile_idx,drone_direction, drone_speed, release_time, detonation_delay):
    """计算指定参数下的有效遮蔽时长"""
    detonation_time = release_time + detonation_delay
    
    # 计算投放点
    release_pos = drone_trajectory(DRONE_POSITIONS[drone_idx], drone_direction, drone_speed, release_time)
    
    # 计算起爆点
    detonation_pos = smoke_trajectory(release_pos, release_time, drone_direction, drone_speed, detonation_time)
    
    limit_time = detonation_time + SMOKE_EFFECTIVE_TIME
    
    # 模拟导弹和烟幕的轨迹
    effective_duration = 0
    in_cloud = False
    start_time = None
    
    # 时间步长
    time_step = 1
    
    # 遍历导弹飞行的整个过程
    for t in np.arange(0, limit_time, time_step):
        missile_pos = missile_trajectory(MISSILE_POSITIONS[missile_idx], FAKE_TARGET, MISSILE_SPEED, t)
        
        if t >= detonation_time and t - detonation_time <= SMOKE_EFFECTIVE_TIME:
            cloud_pos = smoke_cloud_trajectory(detonation_pos, detonation_time, t)
            
            if cloud_pos is not None:
                is_effective = calculate_shielding_effectiveness(
                    missile_pos, cloud_pos, detonation_time, t
                )
                
                if is_effective and not in_cloud:
                    in_cloud = True
                    start_time = t
                elif not is_effective and in_cloud:
                    in_cloud = False
                    effective_duration += (t - start_time)
    
    # 检查如果结束时仍在云团中
    if in_cloud:
        end_time =  detonation_time + SMOKE_EFFECTIVE_TIME
        effective_duration += (end_time - start_time)
    
    return {
        'effective_duration': effective_duration,
        'release_pos': release_pos,
        'detonation_pos': detonation_pos,
        'detonation_time': detonation_time
    }

def optimize_problem(drone_idx,missile_idx):
    print("正在优化问题...")
    start_time = time.time()
    
    # 创建存储所有解决方案的列表
    all_solutions = []
    
    # 搜索参数空间
    best_params = None
    best_duration = 0
    

    # 创建结果目录
    results_dir = f"FY{drone_idx+1}_Missile{missile_idx+1}_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if DRONE_POSITIONS[drone_idx][1] - MISSILE_POSITIONS[missile_idx][1] < 0:
        directions = np.linspace(0, 180, 41)
    else:
        directions = np.linspace(180, 360, 41)



    # 搜索空间
    for direction in directions: 
        start_direction_time = time.time()
        for speed in np.linspace(100, 140, 21):  #26# 70到140 m/s，每2.5 m/s搜索一次
            for release_time in np.linspace(0, 40, 11):  #11 
                for detonation_delay in np.linspace(0, 20, 21): #11  
                
                    result = calculate_effective_shielding_time(
                        drone_idx,missile_idx,direction, speed, release_time, detonation_delay
                    )
                    
                    # 将当前解决方案添加到列表中(只保存有效时长大于4的方案)
                    if result['effective_duration'] > 0:
                        solution = {
                            'direction': direction,
                            'speed': speed,
                            'release_time': release_time,
                            'detonation_delay': detonation_delay,
                            'release_pos_x': result['release_pos'][0],
                            'release_pos_y': result['release_pos'][1],
                            'release_pos_z': result['release_pos'][2],
                            'detonation_pos_x': result['detonation_pos'][0],
                            'detonation_pos_y': result['detonation_pos'][1],
                            'detonation_pos_z': result['detonation_pos'][2],
                            'effective_duration': result['effective_duration']
                        }
                        all_solutions.append(solution)
                    
                    if result['effective_duration'] > best_duration:
                        best_duration = result['effective_duration']
                        best_params = {
                            'direction': direction,
                            'speed': speed,
                            'release_time': release_time,
                            'detonation_delay': detonation_delay,
                            'release_pos': result['release_pos'],
                            'detonation_pos': result['detonation_pos'],
                            'detonation_time': result['detonation_time'],
                            'effective_duration': best_duration
                        }
                        
                        print(f"找到更好的参数: 方向={direction}°, 速度={speed}m/s, "
                              f"投放时间={release_time:.2f}s, 起爆延迟={detonation_delay:.2f}s, "
                              f"有效时长={best_duration:.2f}s")
        
        print(f"方向{direction}°搜索完成，当前最佳有效时长: {best_duration:.2f}s")
        # 打印出该方向的搜索耗时
        end_direction_time = time.time()
        print(f"方向{direction}°搜索耗时: {end_direction_time - start_direction_time:.2f}秒")
        
        # 每完成一个方向的搜索，保存一次中间结果
        if all_solutions:
            df_partial = pd.DataFrame(all_solutions)
            df_partial.to_excel(f"{results_dir}/solutions_direction_{direction}.xlsx", index=False)
            print(f"已保存方向{direction}°的所有解决方案")
    
    # 所有方向搜索完成后，保存完整的解决方案集
    if all_solutions:
        df_all = pd.DataFrame(all_solutions)
        df_all.to_excel(f"{results_dir}/all_solutions.xlsx", index=False)
        print(f"已保存所有解决方案到 {results_dir}/all_solutions.xlsx，共 {len(all_solutions)} 个方案")
        
        # 保存前10个最佳解决方案
        df_best = df_all.sort_values(by='effective_duration', ascending=False).head(10)
        df_best.to_excel(f"{results_dir}/top10_solutions.xlsx", index=False)
        print(f"已保存前10个最佳解决方案到 {results_dir}/top10_solutions.xlsx")

    end_time = time.time()
    print(f"优化完成，总耗时: {end_time - start_time:.2f}秒")
    
    # 输出最优参数
    if best_params:
        print("\n最优烟幕干扰弹投放策略:")
        print(f"飞行方向: {best_params['direction']}°")
        print(f"飞行速度: {best_params['speed']} m/s")
        print(f"投放时间: {best_params['release_time']:.2f} s")
        print(f"起爆延迟: {best_params['detonation_delay']:.2f} s")
        print(f"投放点坐标: ({best_params['release_pos'][0]:.2f}, {best_params['release_pos'][1]:.2f}, {best_params['release_pos'][2]:.2f}) m")
        print(f"起爆点坐标: ({best_params['detonation_pos'][0]:.2f}, {best_params['detonation_pos'][1]:.2f}, {best_params['detonation_pos'][2]:.2f}) m")
        print(f"有效遮蔽时长: {best_params['effective_duration']:.2f} s")
        

        
        # 将最优解单独保存
        best_solution_df = pd.DataFrame([{
            'direction': best_params['direction'],
            'speed': best_params['speed'],
            'release_time': best_params['release_time'],
            'detonation_delay': best_params['detonation_delay'],
            'release_pos_x': best_params['release_pos'][0],
            'release_pos_y': best_params['release_pos'][1],
            'release_pos_z': best_params['release_pos'][2],
            'detonation_pos_x': best_params['detonation_pos'][0],
            'detonation_pos_y': best_params['detonation_pos'][1],
            'detonation_pos_z': best_params['detonation_pos'][2],
            'effective_duration': best_params['effective_duration']
        }])
        best_solution_df.to_excel(f"{results_dir}/best_solution.xlsx", index=False)
        print(f"已保存最优解决方案到 {results_dir}/best_solution.xlsx")


    return best_params, all_solutions



if __name__ == "__main__":
    for i in range(5):
        for j in range(3):
            if i==0 :
                continue
            best_params, all_solutions = optimize_problem(i,j)
    