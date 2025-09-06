import pandas as pd
import numpy as np
import itertools
import concurrent.futures
import os
import time
from tqdm import tqdm

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
DRONE_FY2_INIT = np.array([12000, 1400, 1400])
DRONE_FY3_INIT = np.array([6000, -3000, 700])

DRONE_LOCATIONS=[DRONE_FY1_INIT,DRONE_FY2_INIT,DRONE_FY3_INIT]

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


def calculate_effective_shielding_time(drone_idx, drone_direction, drone_speed, release_time, detonation_delay):
    """计算指定参数下的有效遮蔽时长"""
    detonation_time = release_time + detonation_delay

    # 计算投放点
    release_pos = drone_trajectory(DRONE_LOCATIONS[drone_idx], drone_direction, drone_speed, release_time)
    
    # 计算起爆点
    detonation_pos = smoke_trajectory(release_pos, release_time, drone_direction, drone_speed, detonation_time)
    
    if detonation_pos is None:
        return {
            'effective_duration': 0,
            'release_pos': release_pos,
            'detonation_pos': None,
            'detonation_time': detonation_time,
            'start_time': None,
            'end_time': None,
            'shielding_intervals': []
        }
    
    limit_time = detonation_time + SMOKE_EFFECTIVE_TIME
    
    # 模拟导弹和烟幕的轨迹
    effective_duration = 0
    in_cloud = False
    start_time = None
    shielding_intervals = []
    
    # 时间步长
    time_step = 0.01
    
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
                    effective_duration += (t - start_time)
                    shielding_intervals.append((start_time, t))
    
    # 检查如果结束时仍在云团中
    end_time = None
    if in_cloud and start_time is not None:
        end_time = min(limit_time, detonation_time + SMOKE_EFFECTIVE_TIME)
        effective_duration += (end_time - start_time)
        shielding_intervals.append((start_time, end_time))
    
    # 如果没有遮蔽区间，设置默认的开始和结束时间
    if not shielding_intervals:
        first_start_time = None
        last_end_time = None
    else:
        first_start_time = shielding_intervals[0][0]
        last_end_time = shielding_intervals[-1][1]
    
    return {
        'effective_duration': effective_duration,
        'release_pos': release_pos,
        'detonation_pos': detonation_pos,
        'detonation_time': detonation_time,
        'start_time': first_start_time,
        'end_time': last_end_time,
        'shielding_intervals': shielding_intervals
    }

def calculate_shielding_times(df, idx):
    """计算每个方案的遮蔽开始和结束时间"""
    drone_idx = idx - 1
    
    # 创建新列
    df['shielding_start'] = None
    df['shielding_end'] = None
    df['shielding_intervals'] = None
    df['release_pos_x'] = None
    df['release_pos_y'] = None
    df['release_pos_z'] = None
    df['detonation_pos_x'] = None
    df['detonation_pos_y'] = None
    df['detonation_pos_z'] = None
    
    print(f"计算FY{idx}的遮蔽时间...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        direction = row['direction']
        speed = row['speed']
        release_time = row['release_time']
        detonation_delay = row['detonation_delay']
        
        result = calculate_effective_shielding_time(drone_idx, direction, speed, release_time, detonation_delay)
        
        # 更新DataFrame
        df.at[i, 'shielding_start'] = result['start_time']
        df.at[i, 'shielding_end'] = result['end_time']
        df.at[i, 'shielding_intervals'] = str(result['shielding_intervals'])
        
        # 添加投放点和起爆点坐标
        if result['release_pos'] is not None:
            df.at[i, 'release_pos_x'] = result['release_pos'][0]
            df.at[i, 'release_pos_y'] = result['release_pos'][1]
            df.at[i, 'release_pos_z'] = result['release_pos'][2]
        
        if result['detonation_pos'] is not None:
            df.at[i, 'detonation_pos_x'] = result['detonation_pos'][0]
            df.at[i, 'detonation_pos_y'] = result['detonation_pos'][1]
            df.at[i, 'detonation_pos_z'] = result['detonation_pos'][2]
        
        # 验证计算的effective_duration是否与Excel中的一致
        calculated_duration = result['effective_duration']
        original_duration = row['effective_duration']
        
        # 如果有显著差异，使用计算的值
        if abs(calculated_duration - original_duration) > 0.1:
            print(f"警告: FY{idx}行{i}的计算时长({calculated_duration:.2f})与原始时长({original_duration:.2f})不同")
            df.at[i, 'effective_duration'] = calculated_duration
    
    return df


def read_excel_file(file_path):
    """读取Excel文件数据"""
    return pd.read_excel(file_path)

def select_top_solutions(df, n=100):
    """选择有效遮蔽时长最长的前n个方案"""
    # 确保有效遮蔽时长列为数值类型
    df['effective_duration'] = pd.to_numeric(df['effective_duration'])
    return df.nlargest(min(n, len(df)), 'effective_duration')

def calculate_total_shielding_time(intervals):
    """计算多个时间区间的总有效时长，处理重叠问题"""
    if not intervals:
        return 0
    
    # 按开始时间排序区间
    sorted_intervals = sorted(intervals)
    
    # 合并重叠区间
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        previous = merged[-1]
        if current[0] <= previous[1]:  # 重叠区间
            # 更新前一个区间的结束时间(如果当前区间结束时间更晚)
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:  # 不重叠区间
            merged.append(current)
    
    # 计算总时长
    total_time = sum(end - start for start, end in merged)
    return total_time


def evaluate_combination(combo):
    """评估一个三无人机组合的总遮蔽时长"""
    sol1, sol2, sol3 = combo
    
    # 确保遮蔽时间有效
    intervals = []
    
    if sol1['shielding_start'] is not None and sol1['shielding_end'] is not None:
        intervals.append((sol1['shielding_start'], sol1['shielding_end']))
    
    if sol2['shielding_start'] is not None and sol2['shielding_end'] is not None:
        intervals.append((sol2['shielding_start'], sol2['shielding_end']))
    
    if sol3['shielding_start'] is not None and sol3['shielding_end'] is not None:
        intervals.append((sol3['shielding_start'], sol3['shielding_end']))
    
    total_time = calculate_total_shielding_time(intervals)
    return total_time, combo

def save_results_to_excel(max_time, best_combo, output_file):
    """保存最优组合结果到Excel文件"""
    # 创建结果数据结构
    results_data = []
    
    # 提取合并后的遮蔽区间
    sol1, sol2, sol3 = best_combo
    intervals = []
    
    for sol in [sol1, sol2, sol3]:
        if sol['shielding_start'] is not None and sol['shielding_end'] is not None:
            intervals.append((sol['shielding_start'], sol['shielding_end']))
    
    # 合并区间
    sorted_intervals = sorted(intervals)
    merged = []
    for interval in sorted_intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    
    # 添加总遮蔽时长
    results_data.append({
        '无人机': '总计',
        '遮蔽时长(秒)': max_time,
        '方向(度)': '',
        '速度(m/s)': '',
        '投放时间(秒)': '',
        '起爆延迟(秒)': '',
        '遮蔽开始时间(秒)': merged[0][0] if merged else None,
        '遮蔽结束时间(秒)': merged[-1][1] if merged else None,
        '投放点x坐标(m)': '',
        '投放点y坐标(m)': '',
        '投放点z坐标(m)': '',
        '起爆点x坐标(m)': '',
        '起爆点y坐标(m)': '',
        '起爆点z坐标(m)': ''
    })
    
    # 添加每个无人机的详细信息
    for i, solution in enumerate(best_combo):
        drone_name = f"FY{i+1}"
        
        # 获取坐标信息
        release_pos_x = solution.get('release_pos_x', solution.get('release_pos', [None])[0] if isinstance(solution.get('release_pos', None), (list, np.ndarray)) else None)
        release_pos_y = solution.get('release_pos_y', solution.get('release_pos', [None, None])[1] if isinstance(solution.get('release_pos', None), (list, np.ndarray)) else None)
        release_pos_z = solution.get('release_pos_z', solution.get('release_pos', [None, None, None])[2] if isinstance(solution.get('release_pos', None), (list, np.ndarray)) else None)
        
        detonation_pos_x = solution.get('detonation_pos_x', solution.get('detonation_pos', [None])[0] if isinstance(solution.get('detonation_pos', None), (list, np.ndarray)) else None)
        detonation_pos_y = solution.get('detonation_pos_y', solution.get('detonation_pos', [None, None])[1] if isinstance(solution.get('detonation_pos', None), (list, np.ndarray)) else None)
        detonation_pos_z = solution.get('detonation_pos_z', solution.get('detonation_pos', [None, None, None])[2] if isinstance(solution.get('detonation_pos', None), (list, np.ndarray)) else None)
        
        results_data.append({
            '无人机': drone_name,
            '遮蔽时长(秒)': solution['effective_duration'],
            '方向(度)': solution['direction'],
            '速度(m/s)': solution['speed'],
            '投放时间(秒)': solution['release_time'],
            '起爆延迟(秒)': solution['detonation_delay'],
            '遮蔽开始时间(秒)': solution['shielding_start'],
            '遮蔽结束时间(秒)': solution['shielding_end'],
            '投放点x坐标(m)': release_pos_x,
            '投放点y坐标(m)': release_pos_y,
            '投放点z坐标(m)': release_pos_z,
            '起爆点x坐标(m)': detonation_pos_x,
            '起爆点y坐标(m)': detonation_pos_y,
            '起爆点z坐标(m)': detonation_pos_z
        })
    
    # 添加合并后的时间区间信息
    for i, (start, end) in enumerate(merged):
        results_data.append({
            '无人机': f'合并区间 {i+1}',
            '遮蔽时长(秒)': end - start,
            '方向(度)': '',
            '速度(m/s)': '',
            '投放时间(秒)': '',
            '起爆延迟(秒)': '',
            '遮蔽开始时间(秒)': start,
            '遮蔽结束时间(秒)': end,
            '投放点x坐标(m)': '',
            '投放点y坐标(m)': '',
            '投放点z坐标(m)': '',
            '起爆点x坐标(m)': '',
            '起爆点y坐标(m)': '',
            '起爆点z坐标(m)': ''
        })
    
    # 创建DataFrame并保存
    results_df = pd.DataFrame(results_data)
    results_df.to_excel(output_file, index=False)
    print(f"已保存优化结果到 {output_file}")

def save_top_solutions(df, output_file, drone_idx):
    """保存前100个最佳方案的详细信息"""
    # 确保所有需要的列都存在
    columns_to_check = [
        'effective_duration', 'direction', 'speed', 'release_time', 
        'detonation_delay', 'shielding_start', 'shielding_end',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'detonation_pos_x', 'detonation_pos_y', 'detonation_pos_z'
    ]
    
    for col in columns_to_check:
        if col not in df.columns:
            df[col] = None
    
    # 添加无人机标识
    df['drone_id'] = f"FY{drone_idx}"
    
    # 保存数据
    df.to_excel(output_file, index=False)
    print(f"已保存 FY{drone_idx} 前100个最佳方案到 {output_file}")

def find_optimal_combination_parallel(df1, df2, df3, num_workers=8):
    """使用并行处理找出最优组合"""
    # 转换DataFrame为字典列表以便处理
    solutions1 = df1.to_dict('records')
    solutions2 = df2.to_dict('records')
    solutions3 = df3.to_dict('records')
    
    # 生成所有组合
    all_combinations = list(itertools.product(solutions1, solutions2, solutions3))
    total_combinations = len(all_combinations)
    print(f"总组合数: {total_combinations}")
    
    # 并行评估所有组合
    best_overall_time = 0
    best_overall_combo = None
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有组合进行评估
        futures = [executor.submit(evaluate_combination, combo) for combo in all_combinations]
        
        # 处理结果
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="评估组合"):
            try:
                time_result, combo = future.result()
                
                if time_result > best_overall_time:
                    best_overall_time = time_result
                    best_overall_combo = combo
                    print(f"找到更好的组合，遮蔽时长: {best_overall_time:.4f}秒")
                    
            except Exception as e:
                print(f"评估组合时出错: {e}")
    
    return best_overall_time, best_overall_combo

def main():
    # 设置输出目录
    output_dir = "multi_drone_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 文件路径
    fy1_file = "FY1.xlsx"
    fy2_file = "FY2.xlsx"
    fy3_file = "FY3.xlsx"
    
    start_time = time.time()
    
    # 读取每个无人机的数据
    print("读取无人机数据...")
    fy1_df = read_excel_file(fy1_file)
    fy2_df = read_excel_file(fy2_file)
    fy3_df = read_excel_file(fy3_file)
    
    # 选择每个无人机的前100个最佳方案
    print("选择前100个最佳方案...")
    fy1_top = select_top_solutions(fy1_df)
    fy2_top = select_top_solutions(fy2_df)
    fy3_top = select_top_solutions(fy3_df)
    
    print(f"FY1: {len(fy1_top)}个方案")
    print(f"FY2: {len(fy2_top)}个方案")
    print(f"FY3: {len(fy3_top)}个方案")
    
    # 计算遮蔽开始和结束时间
    print("计算遮蔽时间...")
    fy1_top = calculate_shielding_times(fy1_top, 1)
    fy2_top = calculate_shielding_times(fy2_top, 2)
    fy3_top = calculate_shielding_times(fy3_top, 3)
    
    # 保存每个无人机的Top100方案详细信息
    save_top_solutions(fy1_top, f"{output_dir}/FY1_top100_detailed.xlsx", 1)
    save_top_solutions(fy2_top, f"{output_dir}/FY2_top100_detailed.xlsx", 2)
    save_top_solutions(fy3_top, f"{output_dir}/FY3_top100_detailed.xlsx", 3)
    
    # 找出最优组合
    print("寻找最优组合...")
    max_time, best_combo = find_optimal_combination_parallel(fy1_top, fy2_top, fy3_top)
    
    # 保存最优组合结果
    output_file = f"{output_dir}/optimal_combination_detailed.xlsx"
    save_results_to_excel(max_time, best_combo, output_file)
    
    # 创建合并结果概要文件
    summary_file = f"{output_dir}/optimization_summary.xlsx"
    summary_data = [{
        '总有效遮蔽时长(秒)': max_time,
        '计算时间(秒)': time.time() - start_time,
        'FY1方向(度)': best_combo[0]['direction'],
        'FY1速度(m/s)': best_combo[0]['speed'],
        'FY1投放时间(秒)': best_combo[0]['release_time'],
        'FY1起爆延迟(秒)': best_combo[0]['detonation_delay'],
        'FY2方向(度)': best_combo[1]['direction'],
        'FY2速度(m/s)': best_combo[1]['speed'],
        'FY2投放时间(秒)': best_combo[1]['release_time'],
        'FY2起爆延迟(秒)': best_combo[1]['detonation_delay'],
        'FY3方向(度)': best_combo[2]['direction'],
        'FY3速度(m/s)': best_combo[2]['speed'],
        'FY3投放时间(秒)': best_combo[2]['release_time'],
        'FY3起爆延迟(秒)': best_combo[2]['detonation_delay']
    }]
    pd.DataFrame(summary_data).to_excel(summary_file, index=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 打印结果
    print("\n优化结果:")
    print(f"最大总遮蔽时长: {max_time:.4f}秒")
    print(f"优化完成，耗时: {elapsed_time:.2f}秒")
    print(f"详细结果已保存到: {output_file}")
    print(f"结果概要已保存到: {summary_file}")
    
    print("\n最优组合详情:")
    for i, solution in enumerate(best_combo):
        drone_name = f"FY{i+1}"
        print(f"\n{drone_name}:")
        print(f"  方向: {solution['direction']}°")
        print(f"  速度: {solution['speed']} m/s")
        print(f"  投放时间: {solution['release_time']}秒")
        print(f"  起爆延迟: {solution['detonation_delay']}秒")
        print(f"  有效遮蔽时长: {solution['effective_duration']}秒")
        if solution['shielding_start'] is not None and solution['shielding_end'] is not None:
            print(f"  遮蔽时间区间: {solution['shielding_start']:.4f}秒 - {solution['shielding_end']:.4f}秒")
        else:
            print(f"  遮蔽时间区间: 无有效遮蔽")

if __name__ == "__main__":
    main()