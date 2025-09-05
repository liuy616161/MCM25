import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

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

# 创建结果目录
RESULTS_DIR = "optimization_results_q3"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

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

def smoke_trajectory(release_pos, release_time, direction_deg, speed, t):
    """计算烟幕弹在时间t的位置（考虑重力）"""
    # 将角度转换为弧度
    direction_rad = np.radians(direction_deg)
    # 计算方向向量
    direction_vector = np.array([np.cos(direction_rad), np.sin(direction_rad), 0])
    if t < release_time:
        return None  # 未释放
    dt = t - release_time
    # 烟幕弹初始速度与无人机相同，并在重力作用下下落
    return release_pos + np.array([0, 0, -0.5 * GRAVITY * dt**2]) + direction_vector * speed * dt

def smoke_cloud_trajectory(detonation_pos, detonation_time, t):
    """计算烟幕云团在时间t的位置"""
    if t < detonation_time:
        return None  # 未爆炸
    dt = t - detonation_time
    return detonation_pos + np.array([0, 0, -SMOKE_DESCENT_RATE * dt]) 

def is_target_in_shadow_cone(missile_pos, cloud_pos, target_pos, target_radius, target_height):
    """检查目标是否在烟幕云团投射的阴影锥体内"""

    # 计算烟幕云团到导弹的方向向量
    direction = missile_pos - cloud_pos
    distance_missile_cloud = np.linalg.norm(direction)
    
    if distance_missile_cloud < SMOKE_EFFECTIVE_RADIUS:
        # 如果导弹在云团内，肯定遮蔽成功
        return True
    
    # 单位方向向量
    unit_direction = direction / distance_missile_cloud
    
    # 计算从导弹到烟幕球体的切线所形成的锥角
    sin_theta = min(1.0, SMOKE_EFFECTIVE_RADIUS / distance_missile_cloud)
    cos_theta = np.sqrt(1 - sin_theta**2)
    
    # 检查目标圆柱体的每个关键点是否在阴影锥体内
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

def calculate_single_smoke_effect(drone_direction, drone_speed, release_time, detonation_delay, time_step=0.01):
    """计算单个烟幕干扰弹的效果，返回有效遮蔽的时间区间列表"""
    detonation_time = release_time + detonation_delay
    
    # 计算投放点
    release_pos = drone_trajectory(DRONE_FY1_INIT, drone_direction, drone_speed, release_time)
    
    # 计算起爆点
    detonation_pos = smoke_trajectory(release_pos, release_time, drone_direction, drone_speed, detonation_time)
    
    if detonation_pos is None:
        return []
    
    # 导弹到达假目标的时间
    missile_total_time = missile_time_to_target(MISSILE_M1_INIT, FAKE_TARGET, MISSILE_SPEED)
    limit_time = min(missile_total_time, detonation_time + SMOKE_EFFECTIVE_TIME + 5)
    
    # 模拟导弹和烟幕的轨迹
    effective_intervals = []
    in_cloud = False
    start_time = None
    
    # 遍历导弹飞行的整个过程
    for t in np.arange(0, limit_time, time_step):
        missile_pos = missile_trajectory(MISSILE_M1_INIT, FAKE_TARGET, MISSILE_SPEED, t)
        
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
                    effective_intervals.append((start_time, t))
    
    # 检查如果结束时仍在云团中
    if in_cloud and start_time is not None:
        end_time = min(limit_time, detonation_time + SMOKE_EFFECTIVE_TIME)
        effective_intervals.append((start_time, end_time))
    
    return effective_intervals

def calculate_total_effective_time(time_intervals):
    """计算多个时间区间的总有效时长，处理重叠问题"""
    if not time_intervals:
        return 0
    
    # 展平列表
    flat_intervals = []
    for intervals in time_intervals:
        flat_intervals.extend(intervals)
    
    # 排序时间区间
    sorted_intervals = sorted(flat_intervals, key=lambda x: x[0])
    
    merged = []
    for interval in sorted_intervals:
        # 如果merged为空或当前区间与前一个不重叠
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # 合并重叠区间
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    
    # 计算总时长
    total_time = sum(end - start for start, end in merged)
    return total_time

def calculate_multi_smoke_effect(params):
    """计算多个烟幕干扰弹的总效果
    
    参数格式:
    params = [direction, speed, t_r1, dt_d1, t_r2, dt_d2, t_r3, dt_d3]
    """
    direction, speed, t_r1, dt_d1, t_r2, dt_d2, t_r3, dt_d3 = params
    
    # 计算三个烟幕弹的有效遮蔽时间区间
    intervals1 = calculate_single_smoke_effect(direction, speed, t_r1, dt_d1)
    intervals2 = calculate_single_smoke_effect(direction, speed, t_r2, dt_d2)
    intervals3 = calculate_single_smoke_effect(direction, speed, t_r3, dt_d3)
    
    # 计算总有效遮蔽时长
    total_time = calculate_total_effective_time([intervals1, intervals2, intervals3])
    
    return {
        'params': params,
        'effective_duration': total_time,
        'intervals': [intervals1, intervals2, intervals3]
    }



def calculate_table_data(direction=5.12, speed=136.44, 
                         t_r1=0.00, dt_d1=0.00, 
                         t_r2=1.00, dt_d2=0.00, 
                         t_r3=6.70, dt_d3=0.74):
    """
    计算烟幕干扰弹投放和起爆点坐标以及有效干扰时长
    
    参数:
    direction - 无人机飞行方向（度）
    speed - 无人机飞行速度（m/s）
    t_r1, dt_d1 - 第一枚烟幕弹的投放时间和起爆延迟
    t_r2, dt_d2 - 第二枚烟幕弹的投放时间和起爆延迟
    t_r3, dt_d3 - 第三枚烟幕弹的投放时间和起爆延迟
    """
    # 计算投放点
    release_pos1 = drone_trajectory(DRONE_FY1_INIT, direction, speed, t_r1)
    release_pos2 = drone_trajectory(DRONE_FY1_INIT, direction, speed, t_r2)
    release_pos3 = drone_trajectory(DRONE_FY1_INIT, direction, speed, t_r3)
    
    # 计算起爆点
    detonation_time1 = t_r1 + dt_d1
    detonation_time2 = t_r2 + dt_d2
    detonation_time3 = t_r3 + dt_d3
    
    detonation_pos1 = smoke_trajectory(release_pos1, t_r1, direction, speed, detonation_time1)
    detonation_pos2 = smoke_trajectory(release_pos2, t_r2, direction, speed, detonation_time2)
    detonation_pos3 = smoke_trajectory(release_pos3, t_r3, direction, speed, detonation_time3)
    
    # 计算各烟幕弹的有效干扰时长
    intervals1 = calculate_single_smoke_effect(direction, speed, t_r1, dt_d1)
    intervals2 = calculate_single_smoke_effect(direction, speed, t_r2, dt_d2)
    intervals3 = calculate_single_smoke_effect(direction, speed, t_r3, dt_d3)
    
    effective_duration1 = sum(end - start for start, end in intervals1)
    effective_duration2 = sum(end - start for start, end in intervals2)
    effective_duration3 = sum(end - start for start, end in intervals3)
    
    # 计算总有效干扰时长
    total_effective_duration = calculate_total_effective_time([intervals1, intervals2, intervals3])
    
    # 整理表格数据
    table_data = []
    
    # 添加第一枚烟幕弹数据
    table_data.append({
        '无人机运动方向': direction,
        '无人机运动速度 (m/s)': speed,
        '烟幕干扰弹编号': 1,
        '烟幕干扰弹投放点的x坐标 (m)': round(release_pos1[0], 2),
        '烟幕干扰弹投放点的y坐标 (m)': round(release_pos1[1], 2),
        '烟幕干扰弹投放点的z坐标 (m)': round(release_pos1[2], 2),
        '烟幕干扰弹起爆点的x坐标 (m)': round(detonation_pos1[0], 2),
        '烟幕干扰弹起爆点的y坐标 (m)': round(detonation_pos1[1], 2),
        '烟幕干扰弹起爆点的z坐标 (m)': round(detonation_pos1[2], 2),
        '有效干扰时长 (s)': round(effective_duration1, 2)
    })
    
    # 添加第二枚烟幕弹数据
    table_data.append({
        '无人机运动方向': direction,
        '无人机运动速度 (m/s)': speed,
        '烟幕干扰弹编号': 2,
        '烟幕干扰弹投放点的x坐标 (m)': round(release_pos2[0], 2),
        '烟幕干扰弹投放点的y坐标 (m)': round(release_pos2[1], 2),
        '烟幕干扰弹投放点的z坐标 (m)': round(release_pos2[2], 2),
        '烟幕干扰弹起爆点的x坐标 (m)': round(detonation_pos2[0], 2),
        '烟幕干扰弹起爆点的y坐标 (m)': round(detonation_pos2[1], 2),
        '烟幕干扰弹起爆点的z坐标 (m)': round(detonation_pos2[2], 2),
        '有效干扰时长 (s)': round(effective_duration2, 2)
    })
    
    # 添加第三枚烟幕弹数据
    table_data.append({
        '无人机运动方向': direction,
        '无人机运动速度 (m/s)': speed,
        '烟幕干扰弹编号': 3,
        '烟幕干扰弹投放点的x坐标 (m)': round(release_pos3[0], 2),
        '烟幕干扰弹投放点的y坐标 (m)': round(release_pos3[1], 2),
        '烟幕干扰弹投放点的z坐标 (m)': round(release_pos3[2], 2),
        '烟幕干扰弹起爆点的x坐标 (m)': round(detonation_pos3[0], 2),
        '烟幕干扰弹起爆点的y坐标 (m)': round(detonation_pos3[1], 2),
        '烟幕干扰弹起爆点的z坐标 (m)': round(detonation_pos3[2], 2),
        '有效干扰时长 (s)': round(effective_duration3, 2)
    })
    
    # 创建DataFrame并打印结果
    df = pd.DataFrame(table_data)
    print(f"总有效干扰时长: {total_effective_duration:.2f} 秒")
    return df

# 使用给定参数调用函数
result_df = calculate_table_data(direction=5.12, speed=136.44, 
                                t_r1=0.00, dt_d1=0.00, 
                                t_r2=1.00, dt_d2=0.00, 
                                t_r3=6.70, dt_d3=0.74)

# 打印结果表格
print(result_df[['烟幕干扰弹编号', 
                '烟幕干扰弹投放点的x坐标 (m)', '烟幕干扰弹投放点的y坐标 (m)', '烟幕干扰弹投放点的z坐标 (m)',
                '烟幕干扰弹起爆点的x坐标 (m)', '烟幕干扰弹起爆点的y坐标 (m)', '烟幕干扰弹起爆点的z坐标 (m)',
                '有效干扰时长 (s)']])

# 保存结果到Excel文件
result_df.to_excel(f'{RESULTS_DIR}/smoke_bomb_coordinates.xlsx', index=False)
print(f"已保存结果到 {RESULTS_DIR}/smoke_bomb_coordinates.xlsx")