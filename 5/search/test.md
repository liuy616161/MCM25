import numpy as np
import pandas as pd
import os
import time
from itertools import product
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import itertools

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
    direction_rad = np.radians(direction_deg)
    direction_vector = np.array([np.cos(direction_rad), np.sin(direction_rad), 0])
    return init_pos + direction_vector * speed * t

def smoke_trajectory(release_pos, release_time, direction_deg, speed, t):
    """计算烟幕弹在时间t的位置"""
    direction_rad = np.radians(direction_deg)
    direction_vector = np.array([np.cos(direction_rad), np.sin(direction_rad), 0])
    if t < release_time:
        return None
    dt = t - release_time
    return release_pos + np.array([0, 0, -0.5 * GRAVITY * dt**2]) + direction_vector * speed * dt

def smoke_cloud_trajectory(detonation_pos, detonation_time, t):
    """计算烟幕云团在时间t的位置"""
    if t < detonation_time:
        return None
    dt = t - detonation_time
    return detonation_pos + np.array([0, 0, -SMOKE_DESCENT_RATE * dt])

def is_target_in_shadow_cone(missile_pos, cloud_pos, target_pos, target_radius, target_height):
    """检查目标是否在烟幕云团投射的阴影锥体内"""
    if missile_pos[0]+10 < cloud_pos[0]:
        return False
    
    direction = missile_pos - cloud_pos
    distance_missile_cloud = np.linalg.norm(direction)
    
    if distance_missile_cloud < SMOKE_EFFECTIVE_RADIUS:
        return True
    
    unit_direction = direction / distance_missile_cloud
    sin_theta = min(1.0, SMOKE_EFFECTIVE_RADIUS / distance_missile_cloud)
    cos_theta = np.sqrt(1 - sin_theta**2)
    
    num_points = 12
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    bottom_points = []
    for angle in angles:
        x = target_pos[0] + target_radius * np.cos(angle)
        y = target_pos[1] + target_radius * np.sin(angle)
        z = target_pos[2]
        bottom_points.append([x, y, z])
    
    top_points = []
    for angle in angles:
        x = target_pos[0] + target_radius * np.cos(angle)
        y = target_pos[1] + target_radius * np.sin(angle)
        z = target_pos[2] + target_height
        top_points.append([x, y, z])
    
    all_points = np.vstack([bottom_points, top_points])
    
    for point in all_points:
        missile_to_point = point - missile_pos
        distance = np.linalg.norm(missile_to_point)
        
        if distance < 1e-10:
            continue
            
        cos_angle = np.dot(missile_to_point, unit_direction) / distance
        if cos_angle > 0:
            return False
            
        cos_angle = abs(cos_angle)
        cos_theta = abs(cos_theta)
        
        if cos_angle < cos_theta:
            return False
    
    return True

def calculate_shielding_effectiveness(missile_pos, cloud_pos, cloud_start_time, current_time):
    """计算烟幕对导弹的遮蔽效果"""
    if cloud_pos is None or current_time - cloud_start_time > SMOKE_EFFECTIVE_TIME:
        return 0
    
    target_shielded = is_target_in_shadow_cone(
        missile_pos, cloud_pos, REAL_TARGET_CENTER, REAL_TARGET_RADIUS, REAL_TARGET_HEIGHT
    )
    
    return 1 if target_shielded else 0

def calculate_missile_defense_effect(missile_idx, defense_plan, time_step=0.01):
    """计算特定导弹的防御效果（有效遮蔽时间区间）"""
    # 导弹到达假目标的时间
    missile_init_pos = MISSILE_POSITIONS[missile_idx]
    distance = np.linalg.norm(FAKE_TARGET - missile_init_pos)
    missile_total_time = distance / MISSILE_SPEED
    
    # 按时间排序防御计划
    sorted_defense = sorted(defense_plan, key=lambda x: x['release_time'] + x['detonation_delay'])
    
    if not sorted_defense:
        return []
    
    earliest_defense = sorted_defense[0]['release_time'] + sorted_defense[0]['detonation_delay']
    latest_defense = sorted_defense[-1]['release_time'] + sorted_defense[-1]['detonation_delay'] + SMOKE_EFFECTIVE_TIME
    
    # 确定模拟的时间范围
    start_time = max(0, earliest_defense - 1)
    end_time = min(missile_total_time, latest_defense + 1)
    
    # 模拟导弹和烟幕的轨迹
    effective_intervals = []
    in_cloud = False
    start_time_record = None
    
    # 为每个烟幕弹计算并缓存起爆点和有效时间窗口
    defense_cache = []
    for defense in sorted_defense:
        drone_idx = defense['drone_idx']
        direction = defense['direction']
        speed = defense['speed']
        release_time = defense['release_time']
        detonation_delay = defense['detonation_delay']
        detonation_time = release_time + detonation_delay
        
        # 计算投放点和起爆点
        drone_init_pos = DRONE_POSITIONS[drone_idx]
        release_pos = drone_trajectory(drone_init_pos, direction, speed, release_time)
        detonation_pos = smoke_trajectory(release_pos, release_time, direction, speed, detonation_time)
        
        if detonation_pos is not None:
            defense_cache.append({
                'detonation_pos': detonation_pos,
                'detonation_time': detonation_time,
                'end_time': detonation_time + SMOKE_EFFECTIVE_TIME
            })
    
    # 遍历导弹飞行的时间步长
    for t in np.arange(start_time, end_time, time_step):
        missile_pos = missile_trajectory(missile_init_pos, FAKE_TARGET, MISSILE_SPEED, t)
        
        # 检查所有烟幕云团对该导弹的遮蔽效果
        is_effective = 0
        
        # 只考虑当前时间有效的烟幕弹
        active_defenses = [d for d in defense_cache if d['detonation_time'] <= t <= d['end_time']]
        
        for defense in active_defenses:
            detonation_pos = defense['detonation_pos']
            detonation_time = defense['detonation_time']
            
            # 计算云团位置
            cloud_pos = smoke_cloud_trajectory(detonation_pos, detonation_time, t)
            
            if cloud_pos is not None:
                # 计算遮蔽效果
                effect = calculate_shielding_effectiveness(
                    missile_pos, cloud_pos, detonation_time, t
                )
                is_effective = max(is_effective, effect)
        
        # 记录有效遮蔽区间
        if is_effective and not in_cloud:
            in_cloud = True
            start_time_record = t
        elif not is_effective and in_cloud:
            in_cloud = False
            effective_intervals.append((start_time_record, t))
    
    # 检查如果结束时仍在云团中
    if in_cloud and start_time_record is not None:
        effective_intervals.append((start_time_record, end_time))
    
    return effective_intervals

def calculate_single_bomb_intervals(drone_idx, missile_idx, direction, speed, release_time, detonation_delay):
    """计算单个烟幕弹的有效遮蔽区间"""
    defense_plan = [{
        'drone_idx': drone_idx,
        'direction': direction,
        'speed': speed,
        'release_time': release_time,
        'detonation_delay': detonation_delay
    }]
    
    return calculate_missile_defense_effect(missile_idx, defense_plan)

def calculate_total_effective_time(time_intervals):
    """计算多个时间区间的总有效时长，处理重叠问题"""
    if not time_intervals:
        return 0
    
    sorted_intervals = sorted(time_intervals)
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        previous = merged[-1]
        if current[0] <= previous[1]:  # 重叠区间
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:  # 不重叠区间
            merged.append(current)
    
    total_time = sum(end - start for start, end in merged)
    return total_time

def get_top_solutions_with_intervals(drone_idx, missile_idx, num_solutions=3):
    """获取指定无人机-导弹组合的前几个最佳解决方案并预计算其遮蔽区间"""
    solutions = []
    
    # 无人机1使用固定参数
    if drone_idx == 0:
        if missile_idx == 0:  # 只对M1有效
            # FY1对M1的三个烟幕弹参数
            bomb_params = [
                {
                    'direction': 4.87985955499065,
                    'speed': 140,
                    'release_time': 0,
                    'detonation_delay': 0
                },
                {
                    'direction': 4.87985955499065,
                    'speed': 140,
                    'release_time': 1,
                    'detonation_delay': 0
                },
                {
                    'direction': 4.87985955499065,
                    'speed': 140,
                    'release_time': 7.00425122294099,
                    'detonation_delay': 0.203560474384496
                }
            ]
            
            # 计算每个烟幕弹的有效遮蔽区间
            for i, params in enumerate(bomb_params):
                intervals = calculate_single_bomb_intervals(
                    drone_idx, missile_idx, 
                    params['direction'], params['speed'], 
                    params['release_time'], params['detonation_delay']
                )
                
                effective_duration = calculate_total_effective_time(intervals)
                
                solutions.append({
                    'direction': params['direction'],
                    'speed': params['speed'],
                    'release_time': params['release_time'],
                    'detonation_delay': params['detonation_delay'],
                    'effective_duration': effective_duration,
                    'intervals': intervals
                })
            
            print(f"已预计算无人机FY{drone_idx+1}对导弹M{missile_idx+1}的{len(solutions)}个烟幕弹遮蔽区间")
            return solutions
    
    # 其他无人机从文件加载解决方案
    try:
        fine_tune_dir = f"FY{drone_idx+1}_Missile{missile_idx+1}_results/fine_tuned"
        
        # 优先使用精细化的解决方案
        if os.path.exists(f"{fine_tune_dir}/top10_fine_tuned_solutions.xlsx"):
            df = pd.read_excel(f"{fine_tune_dir}/top10_fine_tuned_solutions.xlsx")
            if not df.empty:
                top_df = df.sort_values(by='effective_duration', ascending=False).head(num_solutions)
                for _, row in top_df.iterrows():
                    # 计算该方案的有效遮蔽区间
                    intervals = calculate_single_bomb_intervals(
                        drone_idx, missile_idx, 
                        row['direction'], row['speed'], 
                        row['release_time'], row['detonation_delay']
                    )
                    
                    solutions.append({
                        'direction': row['direction'],
                        'speed': row['speed'],
                        'release_time': row['release_time'],
                        'detonation_delay': row['detonation_delay'],
                        'effective_duration': row['effective_duration'],
                        'intervals': intervals
                    })
        
        # 如果精细化解决方案不足，使用原始解决方案补充
        if len(solutions) < num_solutions:
            results_dir = f"FY{drone_idx+1}_Missile{missile_idx+1}_results"
            if os.path.exists(f"{results_dir}/top10_solutions.xlsx"):
                df = pd.read_excel(f"{results_dir}/top10_solutions.xlsx")
                if not df.empty:
                    top_df = df.sort_values(by='effective_duration', ascending=False).head(num_solutions - len(solutions))
                    for _, row in top_df.iterrows():
                        # 计算该方案的有效遮蔽区间
                        intervals = calculate_single_bomb_intervals(
                            drone_idx, missile_idx, 
                            row['direction'], row['speed'], 
                            row['release_time'], row['detonation_delay']
                        )
                        
                        solutions.append({
                            'direction': row['direction'],
                            'speed': row['speed'],
                            'release_time': row['release_time'],
                            'detonation_delay': row['detonation_delay'],
                            'effective_duration': row['effective_duration'],
                            'intervals': intervals
                        })
        
        print(f"已预计算无人机FY{drone_idx+1}对导弹M{missile_idx+1}的{len(solutions)}个解决方案遮蔽区间")
    except Exception as e:
        print(f"加载解决方案时出错: {e}")
    
    return solutions

def optimize_defense_strategy(num_combinations_per_drone=2):

    """优化整体防御策略，考虑不同无人机和导弹的组合，使用预计算的遮蔽区间"""
    print("开始优化整体防御策略...")
    
    # 创建结果目录
    results_dir = "combined_defense_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 获取每个无人机-导弹组合的前几个最佳解决方案和遮蔽区间
    print("预计算每个无人机-导弹组合的遮蔽区间...")
    drone_missile_solutions = {}
    
    for drone_idx in range(5):
        drone_missile_solutions[drone_idx] = {}
        for missile_idx in range(3):
            solutions = get_top_solutions_with_intervals(drone_idx, missile_idx, num_combinations_per_drone)
            if solutions:
                drone_missile_solutions[drone_idx][missile_idx] = solutions
    
    # 为每个无人机创建可能的烟幕弹组合
    drone_combinations = {}
    
    # 无人机1固定参数
    drone_combinations[0] = [{
        'direction': 4.87985955499065,
        'speed': 140,
        'bombs': [
            {
                'release_time': 0, 
                'detonation_delay': 0, 
                'target_missile': 0,
                'intervals': drone_missile_solutions[0][0][0]['intervals']
            },
            {
                'release_time': 1, 
                'detonation_delay': 0, 
                'target_missile': 0,
                'intervals': drone_missile_solutions[0][0][1]['intervals']
            },
            {
                'release_time': 7.00425122294099, 
                'detonation_delay': 0.203560474384496, 
                'target_missile': 0,
                'intervals': drone_missile_solutions[0][0][2]['intervals']
            }
        ]
    }]
    
    # 为其他无人机创建组合 - 改进的方法
    for drone_idx in range(1, 5):
        drone_combinations[drone_idx] = []
        
        # 1. 收集该无人机所有可能的方向和速度组合
        direction_speed_options = set()
        for missile_idx in range(3):
            if missile_idx in drone_missile_solutions[drone_idx]:
                for solution in drone_missile_solutions[drone_idx][missile_idx]:
                    direction_speed_options.add((solution['direction'], solution['speed']))
        
        # 将集合转换为列表以便迭代
        direction_speed_options = list(direction_speed_options)
        
        # 2. 对每个方向和速度组合，找到针对每个导弹的最佳参数
        for direction, speed in direction_speed_options:
            # 为每个导弹找到在给定方向和速度下的最佳解决方案
            missile_best_params = {}
            
            for missile_idx in range(3):
                if missile_idx in drone_missile_solutions[drone_idx]:
                    # 筛选具有相同方向和速度的解决方案
                    matching_solutions = [
                        s for s in drone_missile_solutions[drone_idx][missile_idx]
                        if abs(s['direction'] - direction) < 0.01 and abs(s['speed'] - speed) < 0.01
                    ]
                    
                    if matching_solutions:
                        # 按有效时长排序，选择最佳的
                        best_solution = max(matching_solutions, key=lambda x: x['effective_duration'])
                        missile_best_params[missile_idx] = {
                            'release_time': best_solution['release_time'],
                            'detonation_delay': best_solution['detonation_delay'],
                            'effective_duration': best_solution['effective_duration'],
                            'intervals': best_solution['intervals']
                        }
            
            # 3. 创建所有可能的组合（最多3枚烟幕弹）
            if missile_best_params:  # 如果有任何匹配的解决方案
                # 生成所有可能的导弹目标组合（最多3个）
                for r in range(1, min(4, len(missile_best_params) + 1)):
                    for missile_targets in itertools.combinations(missile_best_params.keys(), r):
                        bombs = []
                        for target in missile_targets:
                            bombs.append({
                                'release_time': missile_best_params[target]['release_time'],
                                'detonation_delay': missile_best_params[target]['detonation_delay'],
                                'target_missile': target,
                                'intervals': missile_best_params[target]['intervals']
                            })
                        
                        # 确保投放时间顺序合理（间隔至少1秒）
                        if len(bombs) > 1:
                            bombs.sort(key=lambda x: x['release_time'])
                            for i in range(1, len(bombs)):
                                if bombs[i]['release_time'] - bombs[i-1]['release_time'] < 1.0:
                                    bombs[i]['release_time'] = bombs[i-1]['release_time'] + 1.0
                        
                        drone_combinations[drone_idx].append({
                            'direction': direction,
                            'speed': speed,
                            'bombs': bombs
                        })
    

    # 创建所有无人机组合的列表
    print("生成所有可能的防御策略...")
    all_drone_combinations = list(product(
        drone_combinations[0],
        drone_combinations[1],
        drone_combinations[2],
        drone_combinations[3],
        drone_combinations[4]
    ))
    
    print(f"共生成{len(all_drone_combinations)}种可能的防御策略组合，开始评估...")
    
    # 定义评估函数 - 使用预计算的区间
    def evaluate_combination(combo):
        # 每个导弹的所有烟幕弹遮蔽区间
        missile_intervals = {0: [], 1: [], 2: []}
        
        # 收集每个无人机对每个导弹的遮蔽区间
        for drone_idx, drone_solution in enumerate(combo):
            for bomb in drone_solution['bombs']:
                target_missile = bomb['target_missile']
                missile_intervals[target_missile].extend(bomb['intervals'])
        
        # 计算每个导弹的总有效遮蔽时长
        missile_times = []
        for missile_idx in range(3):
            total_time = calculate_total_effective_time(missile_intervals[missile_idx])
            missile_times.append(total_time)
        
        # 创建结果对象
        return {
            'drone_solutions': {i: combo[i] for i in range(5)},
            'total_time': sum(missile_times),
            'times_by_missile': missile_times,
            'intervals_by_missile': missile_intervals
        }
    
    # 并行评估所有组合
    num_cores = multiprocessing.cpu_count()
    print(f"使用{num_cores}个CPU核心并行评估...")
    
    all_results = Parallel(n_jobs=num_cores)(
        delayed(evaluate_combination)(combo) 
        for combo in tqdm(all_drone_combinations)
    )
    
    # 按总时长排序
    sorted_results = sorted(all_results, key=lambda x: x['total_time'], reverse=True)
    
    # 保存前100个结果
    top_results = sorted_results[:100]
    
    # 将结果转换为DataFrame
    result_data = []
    for idx, result in enumerate(top_results):
        row = {
            'rank': idx + 1,
            'total_time': result['total_time'],
            'M1_time': result['times_by_missile'][0],
            'M2_time': result['times_by_missile'][1],
            'M3_time': result['times_by_missile'][2]
        }
        
        # 添加每个无人机的参数
        for drone_idx, solution in result['drone_solutions'].items():
            row[f'FY{drone_idx+1}_direction'] = solution['direction']
            row[f'FY{drone_idx+1}_speed'] = solution['speed']
            
            for i, bomb in enumerate(solution['bombs']):
                row[f'FY{drone_idx+1}_bomb{i+1}_missile'] = f"M{bomb['target_missile']+1}"
                row[f'FY{drone_idx+1}_bomb{i+1}_release_time'] = bomb['release_time']
                row[f'FY{drone_idx+1}_bomb{i+1}_detonation_delay'] = bomb['detonation_delay']
        
        result_data.append(row)
    
    # 创建DataFrame并保存
    df_results = pd.DataFrame(result_data)
    df_results.to_excel(f"{results_dir}/top100_defense_strategies.xlsx", index=False)
    print(f"已保存前100个最佳防御策略到 {results_dir}/top100_defense_strategies.xlsx")
    
    # 输出最佳策略
    # 如果其中一个遮蔽时长为0则跳过，选择下一个

    best_result = sorted_results[0]
    for best_result in sorted_results:
        if all(t > 0 for t in best_result['times_by_missile']):
            break
    print("\n最佳防御策略:")
    print(f"总有效遮蔽时长: {best_result['total_time']:.2f}秒")
    print(f"导弹M1有效遮蔽时长: {best_result['times_by_missile'][0]:.2f}秒")
    print(f"导弹M2有效遮蔽时长: {best_result['times_by_missile'][1]:.2f}秒")
    print(f"导弹M3有效遮蔽时长: {best_result['times_by_missile'][2]:.2f}秒")
    
    print("\n无人机参数:")
    for drone_idx, solution in best_result['drone_solutions'].items():
        print(f"无人机FY{drone_idx+1}: 方向={solution['direction']:.2f}°, 速度={solution['speed']:.2f}m/s")
        for i, bomb in enumerate(solution['bombs']):
            print(f"  烟幕弹{i+1}: 目标=M{bomb['target_missile']+1}, "
                  f"投放时间={bomb['release_time']:.2f}s, 起爆延迟={bomb['detonation_delay']:.2f}s")
    
    # 将最佳策略保存为详细格式
    best_df = pd.DataFrame([{
        'total_time': best_result['total_time'],
        'M1_time': best_result['times_by_missile'][0],
        'M2_time': best_result['times_by_missile'][1],
        'M3_time': best_result['times_by_missile'][2]
    }])
    
    best_df.to_excel(f"{results_dir}/best_strategy_summary.xlsx", index=False)
    
    # 保存详细的无人机参数
    drone_data = []
    for drone_idx, solution in best_result['drone_solutions'].items():
        for i, bomb in enumerate(solution['bombs']):
            drone_data.append({
                '无人机': f'FY{drone_idx+1}',
                '飞行方向(°)': solution['direction'],
                '飞行速度(m/s)': solution['speed'],
                '烟幕弹编号': i+1,
                '目标导弹': f'M{bomb["target_missile"]+1}',
                '投放时间(s)': bomb['release_time'],
                '起爆延迟(s)': bomb['detonation_delay']
            })
    
    pd.DataFrame(drone_data).to_excel(f"{results_dir}/best_strategy_details.xlsx", index=False)
    print(f"已保存最佳策略详细信息到 {results_dir}/best_strategy_details.xlsx")
    
    return sorted_results[0]

if __name__ == "__main__":
    start_time = time.time()
    optimize_defense_strategy(num_combinations_per_drone=2)
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f}秒") 
    
    
    
    
    
    
    
    对于以下部分，选择好无人机对某个导弹最佳参数后， 还应该一起考虑如何释放另外两枚导弹
    
    
    
    # 为其他无人机创建组合 - 改进的方法
    for drone_idx in range(1, 5):
        drone_combinations[drone_idx] = []
        
        # 1. 收集该无人机所有可能的方向和速度组合
        direction_speed_options = set()
        for missile_idx in range(3):
            if missile_idx in drone_missile_solutions[drone_idx]:
                for solution in drone_missile_solutions[drone_idx][missile_idx]:
                    direction_speed_options.add((solution['direction'], solution['speed']))
        
        # 将集合转换为列表以便迭代
        direction_speed_options = list(direction_speed_options)
        
        # 2. 对每个方向和速度组合，找到针对每个导弹的最佳参数
        for direction, speed in direction_speed_options:
            # 为每个导弹找到在给定方向和速度下的最佳解决方案
            missile_best_params = {}
            
            for missile_idx in range(3):
                if missile_idx in drone_missile_solutions[drone_idx]:
                    # 筛选具有相同方向和速度的解决方案
                    matching_solutions = [
                        s for s in drone_missile_solutions[drone_idx][missile_idx]
                        if abs(s['direction'] - direction) < 0.01 and abs(s['speed'] - speed) < 0.01
                    ]
                    
                    if matching_solutions:
                        # 按有效时长排序，选择最佳的
                        best_solution = max(matching_solutions, key=lambda x: x['effective_duration'])
                        missile_best_params[missile_idx] = {
                            'release_time': best_solution['release_time'],
                            'detonation_delay': best_solution['detonation_delay'],
                            'effective_duration': best_solution['effective_duration'],
                            'intervals': best_solution['intervals']
                        }
            
            # 3. 创建所有可能的组合（最多3枚烟幕弹）
            if missile_best_params:  # 如果有任何匹配的解决方案
                # 生成所有可能的导弹目标组合（最多3个）
                for r in range(1, min(4, len(missile_best_params) + 1)):
                    for missile_targets in itertools.combinations(missile_best_params.keys(), r):
                        bombs = []
                        for target in missile_targets:
                            bombs.append({
                                'release_time': missile_best_params[target]['release_time'],
                                'detonation_delay': missile_best_params[target]['detonation_delay'],
                                'target_missile': target,
                                'intervals': missile_best_params[target]['intervals']
                            })
                        
                        # 确保投放时间顺序合理（间隔至少1秒）
                        if len(bombs) > 1:
                            bombs.sort(key=lambda x: x['release_time'])
                            for i in range(1, len(bombs)):
                                if bombs[i]['release_time'] - bombs[i-1]['release_time'] < 1.0:
                                    bombs[i]['release_time'] = bombs[i-1]['release_time'] + 1.0
                        
                        drone_combinations[drone_idx].append({
                            'direction': direction,
                            'speed': speed,
                            'bombs': bombs
                        } 