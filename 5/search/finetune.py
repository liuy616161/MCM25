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
    time_step = 0.1
    
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


def fine_tune_search(drone_idx, missile_idx, num_top_solutions=5):
    """
    对指定无人机-导弹对的前几个最佳解决方案进行精细化搜索
    
    参数:
    drone_idx: 无人机索引(0-4)
    missile_idx: 导弹索引(0-2)
    num_top_solutions: 要精细化搜索的顶部解决方案数量
    """
    print(f"开始对无人机FY{drone_idx+1}和导弹M{missile_idx+1}的解决方案进行精细化搜索...")
    
    # 创建结果目录
    results_dir = f"FY{drone_idx+1}_Missile{missile_idx+1}_results"
    fine_tune_dir = f"{results_dir}/fine_tuned"
    if not os.path.exists(fine_tune_dir):
        os.makedirs(fine_tune_dir)
    
    # 读取原始所有解决方案
    try:
        all_solutions_file = f"{results_dir}/all_solutions.xlsx"
        if not os.path.exists(all_solutions_file):
            print(f"文件 {all_solutions_file} 不存在，将尝试合并分方向的解决方案文件")
            
            # 尝试合并所有方向的解决方案
            all_direction_solutions = []
            for direction in range(0, 361, 9):  # 假设方向是以9度为间隔
                direction_file = f"{results_dir}/solutions_direction_{direction}.xlsx"
                if os.path.exists(direction_file):
                    df_direction = pd.read_excel(direction_file)
                    all_direction_solutions.append(df_direction)
            
            if all_direction_solutions:
                df_all = pd.concat(all_direction_solutions, ignore_index=True)
                df_all.to_excel(all_solutions_file, index=False)
                print(f"已合并所有方向解决方案并保存到 {all_solutions_file}")
            else:
                print("未找到任何方向的解决方案文件")
                return None
        
        df_all = pd.read_excel(all_solutions_file)
    except Exception as e:
        print(f"读取解决方案文件时出错: {e}")
        return None
    
    # 对有效遮蔽时长进行排序，选取前几个解决方案
    df_top = df_all.sort_values(by='effective_duration', ascending=False).head(num_top_solutions)
    print(f"找到前{len(df_top)}个最佳解决方案，准备进行精细化搜索")
    
    # 保存原始顶部解决方案
    df_top.to_excel(f"{fine_tune_dir}/original_top{num_top_solutions}_solutions.xlsx", index=False)
    
    # 存储所有精细化搜索的解决方案
    fine_tuned_solutions = []
    
    # 对每个顶部解决方案进行精细化搜索
    for idx, solution in df_top.iterrows():
        print(f"\n对解决方案 #{idx+1} 进行精细化搜索:")
        print(f"原始参数: 方向={solution['direction']}°, 速度={solution['speed']}m/s, "
              f"投放时间={solution['release_time']:.2f}s, 起爆延迟={solution['detonation_delay']:.2f}s, "
              f"有效时长={solution['effective_duration']:.2f}s")
        
        # 获取原始参数
        orig_direction = solution['direction']
        orig_speed = solution['speed']
        orig_release_time = solution['release_time']
        orig_detonation_delay = solution['detonation_delay']
        
        # 定义精细搜索范围（较小的步长）
        # 方向: ±2度，步长0.5度
        directions = np.linspace(max(0, orig_direction-8), min(360, orig_direction+8), 11)
        # 速度: ±2.5 m/s，步长0.5 m/s
        speeds = np.linspace(max(70, orig_speed-5), min(140, orig_speed+5), 11)
        # 投放时间: ±1秒，步长0.2秒
        release_times = np.linspace(max(0, orig_release_time-3), orig_release_time+3, 11)
        # 起爆延迟: ±1秒，步长0.2秒
        detonation_delays = np.linspace(max(0, orig_detonation_delay-3), 
                                        min(10, orig_detonation_delay+2), 11)
        
        # 记录当前解决方案的最佳精细化结果
        best_fine_tuned = None
        best_duration = solution['effective_duration']
        
        # 进度计数器
        total_iterations = len(directions) * len(speeds) * len(release_times) * len(detonation_delays)
        current_iteration = 0
        last_progress = -1
        
        # 精细化搜索
        start_time = time.time()
        for direction in directions:
            for speed in speeds:
                for release_time in release_times:
                    for detonation_delay in detonation_delays:
                        current_iteration += 1
                        progress = int(current_iteration / total_iterations * 100)
                        
                        # 每10%显示一次进度
                        if progress % 10 == 0 and progress != last_progress:
                            elapsed_time = time.time() - start_time
                            estimated_total = elapsed_time / (current_iteration / total_iterations)
                            remaining_time = estimated_total - elapsed_time
                            print(f"进度: {progress}%, 剩余时间: {remaining_time:.2f}秒")
                            last_progress = progress
                        
                        # 计算当前参数的有效遮蔽时长
                        result = calculate_effective_shielding_time(
                            drone_idx, missile_idx, direction, speed, release_time, detonation_delay
                        )
                        
                        # 保存所有有效的精细化解决方案
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
                                'effective_duration': result['effective_duration'],
                                'original_solution_idx': idx
                            }
                            fine_tuned_solutions.append(solution)
                        
                        # 更新最佳解决方案
                        if result['effective_duration'] > best_duration:
                            best_duration = result['effective_duration']
                            best_fine_tuned = {
                                'direction': direction,
                                'speed': speed,
                                'release_time': release_time,
                                'detonation_delay': detonation_delay,
                                'release_pos': result['release_pos'],
                                'detonation_pos': result['detonation_pos'],
                                'detonation_time': result['detonation_time'],
                                'effective_duration': best_duration,
                                'improvement': best_duration - solution['effective_duration']
                            }
        
        # 输出当前解决方案的精细化结果
        if best_fine_tuned:
            improvement = best_fine_tuned['improvement']
            print(f"精细化结果: 方向={best_fine_tuned['direction']:.2f}°, 速度={best_fine_tuned['speed']:.2f}m/s, "
                  f"投放时间={best_fine_tuned['release_time']:.2f}s, 起爆延迟={best_fine_tuned['detonation_delay']:.2f}s, "
                  f"有效时长={best_fine_tuned['effective_duration']:.2f}s (提升了{improvement:.2f}s)")
        else:
            print("没有找到更好的解决方案")
    
    # 保存所有精细化解决方案
    if fine_tuned_solutions:
        df_fine_tuned = pd.DataFrame(fine_tuned_solutions)
        df_fine_tuned.to_excel(f"{fine_tune_dir}/all_fine_tuned_solutions.xlsx", index=False)
        print(f"已保存所有精细化解决方案到 {fine_tune_dir}/all_fine_tuned_solutions.xlsx")
        
        # 保存前10个最佳精细化解决方案
        df_best = df_fine_tuned.sort_values(by='effective_duration', ascending=False).head(10)
        df_best.to_excel(f"{fine_tune_dir}/top10_fine_tuned_solutions.xlsx", index=False)
        print(f"已保存前10个最佳精细化解决方案到 {fine_tune_dir}/top10_fine_tuned_solutions.xlsx")
        
        # 将原始最佳和精细化最佳合并比较
        best_original = df_top.sort_values(by='effective_duration', ascending=False).iloc[0].to_dict()
        best_fine_tuned = df_best.iloc[0].to_dict()
        
        comparison_df = pd.DataFrame([
            {
                'type': '原始最佳',
                'direction': best_original['direction'],
                'speed': best_original['speed'],
                'release_time': best_original['release_time'],
                'detonation_delay': best_original['detonation_delay'],
                'effective_duration': best_original['effective_duration']
            },
            {
                'type': '精细化最佳',
                'direction': best_fine_tuned['direction'],
                'speed': best_fine_tuned['speed'],
                'release_time': best_fine_tuned['release_time'],
                'detonation_delay': best_fine_tuned['detonation_delay'],
                'effective_duration': best_fine_tuned['effective_duration']
            }
        ])
        
        comparison_df.to_excel(f"{fine_tune_dir}/comparison.xlsx", index=False)
        print(f"已保存原始最佳与精细化最佳的比较到 {fine_tune_dir}/comparison.xlsx")
        
        # 计算和输出提升情况
        improvement = best_fine_tuned['effective_duration'] - best_original['effective_duration']
        improvement_percentage = improvement / best_original['effective_duration'] * 100
        print(f"\n精细化搜索使有效遮蔽时长从 {best_original['effective_duration']:.2f}s 提升到 "
              f"{best_fine_tuned['effective_duration']:.2f}s，提升了 {improvement:.2f}s ({improvement_percentage:.2f}%)")
    
    return fine_tuned_solutions

# 运行精细化搜索
if __name__ == "__main__":
    for drone_idx in range(5):
        for missile_idx in range(3):
            if drone_idx==0:
                continue
            fine_tune_search(drone_idx, missile_idx, num_top_solutions=5)