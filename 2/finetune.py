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
    
    # 导弹到达假目标的时间
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
    direction_range = np.linspace(direction - 0.5, direction + 0.5, 5)
    speed_range = np.linspace(max(70, speed - 1.5), min(140, speed + 1.5), 5)
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
                    result = calculate_effective_shielding_time(d, s, rt, dd)
                    
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
                              f"有效时长={result['effective_duration']:.2f}s")
                    
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

def main():
    # 创建结果目录
    results_dir = "multi_point_optimization_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 读取前10个最佳解决方案
    try:
        df_top10 = pd.read_excel("optimization_results/top10_solutions.xlsx")
        print(f"成功加载前10个最佳解决方案，共 {len(df_top10)} 个方案")
    except Exception as e:
        print(f"加载前10个最佳解决方案失败: {e}")
        return
    
    # 对每个解决方案进行精细化搜索
    all_fine_tuned_solutions = []
    best_solutions = []
    
    for i, row in df_top10.iterrows():
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
        
        # 可视化比较
        plt.figure(figsize=(12, 8))
        plt.bar(df_best['solution_id'], df_best['effective_duration'], color='skyblue')
        plt.axhline(y=df_top10['effective_duration'].max(), color='r', linestyle='-', label='原始最佳解')
        
        # 添加数据标签
        for i, value in enumerate(df_best['effective_duration']):
            plt.text(i+1, value + 0.05, f'{value:.2f}', ha='center')
        
        plt.xlabel('解决方案ID')
        plt.ylabel('有效遮蔽时长 (秒)')
        plt.title('各起点精细化搜索的最佳解决方案比较')
        plt.xticks(df_best['solution_id'])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{results_dir}/best_solutions_comparison.png", dpi=300)
        
        # 将最优解单独保存
        best_solution_id = global_best['solution_id']
        best_solution_data = df_all[df_all['base_solution_id'] == best_solution_id]
        best_solution_data = best_solution_data.sort_values(by='effective_duration', ascending=False)
        best_solution_data.head(10).to_excel(f"{results_dir}/global_best_solution_details.xlsx", index=False)
        print(f"已保存全局最优解决方案详情到 {results_dir}/global_best_solution_details.xlsx")

if __name__ == "__main__":
    main()