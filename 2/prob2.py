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

def calculate_effective_shielding_time(drone_direction, drone_speed, release_time, detonation_delay):
    """计算指定参数下的有效遮蔽时长"""
    detonation_time = release_time + detonation_delay
    
    # 计算投放点
    release_pos = drone_trajectory(DRONE_FY1_INIT, drone_direction, drone_speed, release_time)
    
    # 计算起爆点
    detonation_pos = smoke_trajectory(release_pos, release_time, drone_direction, drone_speed, detonation_time)
    
    limit_time = detonation_time + SMOKE_EFFECTIVE_TIME
    
    # 模拟导弹和烟幕的轨迹
    effective_duration = 0
    in_cloud = False
    start_time = None
    
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
    
    # 检查如果结束时仍在云团中
    if in_cloud:
        end_time = min(missile_total_time, detonation_time + SMOKE_EFFECTIVE_TIME)
        effective_duration += (end_time - start_time)
    
    return {
        'effective_duration': effective_duration,
        'release_pos': release_pos,
        'detonation_pos': detonation_pos,
        'detonation_time': detonation_time
    }


def calculate_effective_shielding_time_2(drone_direction, drone_speed, release_time, detonation_delay):
    """计算指定参数下的有效遮蔽时长"""
    detonation_time = release_time + detonation_delay
    
    # 计算投放点
    release_pos = drone_trajectory(DRONE_FY1_INIT, drone_direction, drone_speed, release_time)
    
    # 计算起爆点
    detonation_pos = smoke_trajectory(release_pos, release_time, drone_direction, drone_speed, detonation_time)
    
    limit_time = detonation_time + SMOKE_EFFECTIVE_TIME
    
    # 模拟导弹和烟幕的轨迹
    effective_duration = 0
    in_cloud = False
    start_time = None
    
    # 时间步长
    time_step = 0.001
    
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
    
    # 检查如果结束时仍在云团中
    if in_cloud:
        end_time = min(missile_total_time, detonation_time + SMOKE_EFFECTIVE_TIME)
        effective_duration += (end_time - start_time)
    
    return {
        'effective_duration': effective_duration,
        'release_pos': release_pos,
        'detonation_pos': detonation_pos,
        'detonation_time': detonation_time
    }
def optimize_problem2():
    """优化问题2: 寻找最佳烟幕干扰弹投放策略，并保存所有解决方案"""
    print("正在优化问题2...")
    start_time = time.time()
    
    # 创建存储所有解决方案的列表
    all_solutions = []
    
    # 搜索参数空间
    best_params = None
    best_duration = 0
    
    # 创建结果目录
    results_dir = "optimization_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    directions = np.linspace(3, 9, 41) #41

    # 搜索空间
    for direction in directions: 
        start_direction_time = time.time()
        for speed in np.linspace(90, 140, 26):  #26# 70到140 m/s，每2.5 m/s搜索一次
            for release_time in np.linspace(0, 1.5, 11):  #11 
                for detonation_delay in np.linspace(0, 1, 11): #11  
                    if release_time + detonation_delay > 7:
                        continue
                    result = calculate_effective_shielding_time(
                        direction, speed, release_time, detonation_delay
                    )
                    
                    # 将当前解决方案添加到列表中(只保存有效时长大于4的方案)
                    if result['effective_duration'] > 4:
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

        # 对每个解决方案进行精细化搜索
        all_fine_tuned_solutions = []
        best_solutions = []
        
        for i, row in df_best.iterrows():
            solution_id = i + 1
            base_solution = row.to_dict()
            
            # 进行精细化搜索
            best_params, fine_tuned_solutions = fine_tune_solution(base_solution, solution_id, results_dir)
            
            # 添加到所有解决方案列表
            all_fine_tuned_solutions.extend(fine_tuned_solutions)
            best_solutions.append(best_params)
        
        # 保存所有精细化搜索的解决方案
        if all_fine_tuned_solutions:
            df_all = pd.DataFrame(all_fine_tuned_solutions)
            df_all.to_excel(f"{results_dir}/all_fine_tuned_solutions.xlsx", index=False)
            print(f"已保存所有精细化搜索的解决方案，共 {len(all_fine_tuned_solutions)} 个方案")
        
        # 比较各个起点的最佳解决方案
        if best_solutions:
            df_best = pd.DataFrame([
                {
                    'solution_id': params['solution_id'],
                    'direction': params['direction'],
                    'speed': params['speed'],
                    'release_time': params['release_time'],
                    'detonation_delay': params['detonation_delay'],
                    'effective_duration': params['effective_duration']
                }
                for params in best_solutions
            ])
            
            # 按有效时长排序
            df_best = df_best.sort_values(by='effective_duration', ascending=False)
            df_best.to_excel(f"{results_dir}/best_solutions_comparison.xlsx", index=False)
            print("\n各起点精细化搜索的最佳解决方案比较:")
            print(df_best[['solution_id', 'direction', 'speed', 'release_time', 'detonation_delay', 'effective_duration']])
            
            # 找出全局最优解
            global_best = df_best.iloc[0]
            print("\n全局最优解决方案:")
            print(f"来自解决方案 #{global_best['solution_id']}")
            print(f"方向: {global_best['direction']:.2f}°")
            print(f"速度: {global_best['speed']:.2f} m/s")
            print(f"投放时间: {global_best['release_time']:.2f} s")
            print(f"起爆延迟: {global_best['detonation_delay']:.2f} s")
            print(f"有效遮蔽时长: {global_best['effective_duration']:.2f} s")
            
            
            # 将最优解单独保存
            best_solution_id = global_best['solution_id']
            best_solution_data = df_all[df_all['base_solution_id'] == best_solution_id]
            best_solution_data = best_solution_data.sort_values(by='effective_duration', ascending=False)
            best_solution_data.head(10).to_excel(f"{results_dir}/global_best_solution_details.xlsx", index=False)
            print(f"已保存全局最优解决方案详情到 {results_dir}/global_best_solution_details.xlsx")


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
        
        # 可视化最优结果
        visualize_solution(best_params)
    return best_params, all_solutions



def fine_tune_solution(base_solution, solution_id, results_dir):
    """对一个基础解决方案进行精细化搜索"""
    print(f"\n开始对解决方案 #{solution_id} 进行精细化搜索:")
    print(f"基础参数: 方向={base_solution['direction']}°, 速度={base_solution['speed']}m/s, "
          f"投放时间={base_solution['release_time']:.2f}s, 起爆延迟={base_solution['detonation_delay']:.2f}s, "
          f"有效时长={base_solution['effective_duration']:.2f}s")
    
    start_time = time.time()
    
    # 提取基础参数
    direction = base_solution['direction']
    speed = base_solution['speed']
    release_time = base_solution['release_time']
    detonation_delay = base_solution['detonation_delay']
    best_duration = base_solution['effective_duration']
    
    # 创建精细化搜索结果列表
    fine_tuning_solutions = []
    best_params = {
        'direction': direction,
        'speed': speed,
        'release_time': release_time,
        'detonation_delay': detonation_delay,
        'effective_duration': best_duration,
        'solution_id': solution_id
    }
    
    # 定义搜索范围
    direction_range = np.linspace(direction - 0.075, direction + 0.075, 5)
    speed_range = np.linspace(max(70, speed - 0.5), min(140, speed + 0.5), 5)
    release_time_range = np.linspace(max(0.1, release_time - 0.25), release_time + 0.25, 5)
    detonation_delay_range = np.linspace(max(0.1, detonation_delay - 0.25), detonation_delay + 0.25, 5)
    
    total_combinations = len(direction_range) * len(speed_range) * len(release_time_range) * len(detonation_delay_range)
    print(f"搜索范围: 方向={direction-0.5}°到{direction+0.5}°, 速度={max(70, speed-1.5)}m/s到{min(140, speed+1.5)}m/s")
    print(f"         投放时间={max(0.1, release_time-0.25)}s到{release_time+0.25}s, 起爆延迟={max(0.1, detonation_delay-0.25)}s到{detonation_delay+0.25}s")
    print(f"总搜索组合数: {total_combinations}")
    
    # 进行精细化搜索
    completed = 0
    for d in direction_range:
        for s in speed_range:
            for rt in release_time_range:
                for dd in detonation_delay_range:
                    # 计算有效遮蔽时长
                    result = calculate_effective_shielding_time_2(d, s, rt, dd)
                    
                    # 只保存有效的解决方案
                    if result['effective_duration'] > 0:
                        solution = {
                            'direction': d,
                            'speed': s,
                            'release_time': rt,
                            'detonation_delay': dd,
                            'release_pos_x': result['release_pos'][0],
                            'release_pos_y': result['release_pos'][1],
                            'release_pos_z': result['release_pos'][2],
                            'detonation_pos_x': result['detonation_pos'][0],
                            'detonation_pos_y': result['detonation_pos'][1],
                            'detonation_pos_z': result['detonation_pos'][2],
                            'effective_duration': result['effective_duration'],
                            'base_solution_id': solution_id
                        }
                        fine_tuning_solutions.append(solution)
                    
                    # 更新最佳解决方案
                    if result['effective_duration'] > best_params['effective_duration']:
                        best_params = {
                            'direction': d,
                            'speed': s,
                            'release_time': rt,
                            'detonation_delay': dd,
                            'release_pos': result['release_pos'],
                            'detonation_pos': result['detonation_pos'],
                            'detonation_time': result['detonation_time'],
                            'effective_duration': result['effective_duration'],
                            'solution_id': solution_id
                        }
                        
                        print(f"发现更优解: 方向={d:.2f}°, 速度={s:.2f}m/s, "
                              f"投放时间={rt:.2f}s, 起爆延迟={dd:.2f}s, "
                              f"有效时长={result['effective_duration']:.4f}s")
                    
                    # 更新进度
                    completed += 1
                    if completed % 500 == 0:
                        progress = (completed / total_combinations) * 100
                        print(f"完成: {completed}/{total_combinations} ({progress:.1f}%)")
    
    # 保存精细化搜索结果
    if fine_tuning_solutions:
        df_fine = pd.DataFrame(fine_tuning_solutions)
        file_path = f"{results_dir}/fine_tuning_solution_{solution_id}.xlsx"
        df_fine.to_excel(file_path, index=False)
        print(f"已保存精细化搜索结果到 {file_path}，共 {len(fine_tuning_solutions)} 个方案")
    
    # 输出统计信息
    end_time = time.time()
    print(f"精细化搜索耗时: {end_time - start_time:.2f}秒")
    print(f"改进幅度: {(best_params['effective_duration'] - best_duration) / best_duration * 100:.2f}%")
    
    return best_params, fine_tuning_solutions


def visualize_solution(params):
    """可视化最优解决方案"""
    # 提取参数
    direction = params['direction']
    speed = params['speed']
    release_time = params['release_time']
    detonation_time = params['detonation_time']
    release_pos = params['release_pos']
    detonation_pos = params['detonation_pos']
    
    # 导弹到达假目标的时间
    missile_total_time = missile_time_to_target(MISSILE_M1_INIT, FAKE_TARGET, MISSILE_SPEED)
    
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
    ax.plot_surface(x_cylinder, y_cylinder, z_grid, alpha=0.5, color='blue', label='真目标')
    
    # 绘制导弹轨迹
    missile_times = np.linspace(0, missile_total_time, 1000)
    missile_positions = []
    for t in missile_times:
        missile_positions.append(missile_trajectory(MISSILE_M1_INIT, FAKE_TARGET, MISSILE_SPEED, t))
    missile_positions = np.array(missile_positions)
    ax.plot(missile_positions[:, 0], missile_positions[:, 1], missile_positions[:, 2], 
            color='red', label='导弹M1轨迹')
    ax.scatter(MISSILE_M1_INIT[0], MISSILE_M1_INIT[1], MISSILE_M1_INIT[2], 
               color='darkred', label='导弹M1初始位置')
    
    # 绘制无人机轨迹
    drone_times = np.linspace(0, release_time, 100)
    drone_positions = []
    for t in drone_times:
        drone_positions.append(drone_trajectory(DRONE_FY1_INIT, direction, speed, t))
    drone_positions = np.array(drone_positions)
    ax.plot(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2], 
            color='green', label='无人机FY1轨迹')
    ax.scatter(DRONE_FY1_INIT[0], DRONE_FY1_INIT[1], DRONE_FY1_INIT[2], 
               color='darkgreen', label='无人机FY1初始位置')
    
    # 绘制烟幕弹投放点
    ax.scatter(release_pos[0], release_pos[1], release_pos[2], 
               color='orange', s=100, label='烟幕弹投放点')
    
    # 绘制烟幕弹起爆点
    ax.scatter(detonation_pos[0], detonation_pos[1], detonation_pos[2], 
               color='purple', s=100, label='烟幕弹起爆点')
    
    # 寻找有效遮蔽时间段
    effective_periods = []
    is_effective = False
    start_time = None
    
    for t in np.arange(0, missile_total_time, 0.1):
        if t >= detonation_time and t - detonation_time <= SMOKE_EFFECTIVE_TIME:
            missile_pos = missile_trajectory(MISSILE_M1_INIT, FAKE_TARGET, MISSILE_SPEED, t)
            cloud_pos = smoke_cloud_trajectory(detonation_pos, detonation_time, t)
            
            if cloud_pos is not None:
                effectiveness = calculate_shielding_effectiveness(
                    missile_pos, cloud_pos, detonation_time, t
                )
                
                if effectiveness and not is_effective:
                    is_effective = True
                    start_time = t
                elif not effectiveness and is_effective:
                    is_effective = False
                    effective_periods.append((start_time, t))
    
    if is_effective and start_time is not None:
        effective_periods.append((start_time, min(missile_total_time, detonation_time + SMOKE_EFFECTIVE_TIME)))
    
    # 绘制有效遮蔽时间段的烟幕云团
    for period_start, period_end in effective_periods:
        # 选择几个时间点绘制云团
        for t in np.linspace(period_start, period_end, 5):
            cloud_pos = smoke_cloud_trajectory(detonation_pos, detonation_time, t)
            if cloud_pos is not None:
                # 绘制烟幕云团
                u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:15j]
                x_sphere = cloud_pos[0] + SMOKE_EFFECTIVE_RADIUS * np.cos(u) * np.sin(v)
                y_sphere = cloud_pos[1] + SMOKE_EFFECTIVE_RADIUS * np.sin(u) * np.sin(v)
                z_sphere = cloud_pos[2] + SMOKE_EFFECTIVE_RADIUS * np.cos(v)
                ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.3, linewidth=0.5)
                
                # 绘制遮蔽锥体示意
                missile_pos = missile_trajectory(MISSILE_M1_INIT, FAKE_TARGET, MISSILE_SPEED, t)
                # 简化处理：只绘制从导弹到真目标中心的线
                ax.plot([missile_pos[0], REAL_TARGET_CENTER[0]], 
                        [missile_pos[1], REAL_TARGET_CENTER[1]], 
                        [missile_pos[2], REAL_TARGET_CENTER[2]], 'r--', alpha=0.3)
    
    # 添加遮蔽时长信息
    ax.text2D(0.05, 0.95, f"有效遮蔽时长: {params['effective_duration']:.2f}秒", 
              transform=ax.transAxes, fontsize=12)
    
    # 设置坐标轴范围和标签
    ax.set_xlim([0, 20000])
    ax.set_ylim([-500, 500])
    ax.set_zlim([0, 2500])
    ax.set_xlabel('X轴 (m)')
    ax.set_ylabel('Y轴 (m)')
    ax.set_zlabel('Z轴 (m)')
    ax.set_title('问题2: 最优烟幕干扰弹投放策略')
    
    # 添加图例
    ax.legend(loc='upper right')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('problem2_optimal_solution.png', dpi=300)
    #plt.show()


if __name__ == "__main__":
    # 运行问题2的优化
    best_params, all_solutions = optimize_problem2()
    