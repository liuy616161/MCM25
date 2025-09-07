import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
import os
import multiprocessing
from deap import base, creator, tools, algorithms
from tqdm import tqdm
import pickle

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
MIN_BOMB_INTERVAL = 1.0  # 同一无人机两次投放的最小间隔

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

# 创建结果目录
RESULTS_DIR = "multi_drone_defense_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# 基础函数
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
    # 如果导弹已经超过云团，则无法形成有效遮蔽
    if missile_pos[0] < cloud_pos[0]:
        return False
    
    # 计算烟幕云团到导弹的方向向量
    direction = missile_pos - cloud_pos
    distance_missile_cloud = np.linalg.norm(direction)
    
    # 如果导弹在云团内，肯定遮蔽成功
    if distance_missile_cloud < SMOKE_EFFECTIVE_RADIUS:
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

def calculate_shielding_effectiveness(missile_idx, missile_pos, cloud_pos, cloud_start_time, current_time):
    """计算烟幕对导弹的遮蔽效果. 动态优化方案"""
    if cloud_pos is None or current_time - cloud_start_time > SMOKE_EFFECTIVE_TIME:
        return 0  # 未形成云团或云团已失效
    
    # 检查真目标是否被烟幕遮蔽
    target_shielded = is_target_in_shadow_cone(
        missile_pos, cloud_pos, REAL_TARGET_CENTER, REAL_TARGET_RADIUS, REAL_TARGET_HEIGHT
    )
    
    return 1 if target_shielded else 0

def calculate_missile_defense_effect(missile_idx, defense_plan, time_step=0.01):
    """计算特定导弹的防御效果（有效遮蔽时间区间）"""
    # 获取导弹初始位置
    missile_init_pos = MISSILE_POSITIONS[missile_idx]
    
    # 导弹到达假目标的时间
    missile_total_time = missile_time_to_target(missile_init_pos, FAKE_TARGET, MISSILE_SPEED)
    
    # 模拟导弹和烟幕的轨迹
    effective_intervals = []
    in_cloud = False
    start_time = None
    
    # 遍历导弹飞行的整个过程
    for t in np.arange(0, missile_total_time, time_step):
        missile_pos = missile_trajectory(missile_init_pos, FAKE_TARGET, MISSILE_SPEED, t)
        
        # 检查所有烟幕云团对该导弹的遮蔽效果
        is_effective = 0
        for bomb in defense_plan:
            drone_idx = bomb['drone_idx']
            direction = bomb['direction']
            speed = bomb['speed']
            release_time = bomb['release_time']
            detonation_delay = bomb['detonation_delay']
            
            # 计算烟幕弹起爆时间和位置
            detonation_time = release_time + detonation_delay
            
            if t >= detonation_time:
                # 计算投放点
                drone_init_pos = DRONE_POSITIONS[drone_idx]
                release_pos = drone_trajectory(drone_init_pos, direction, speed, release_time)
                
                # 计算起爆点
                detonation_pos = smoke_trajectory(release_pos, release_time, direction, speed, detonation_time)
                
                if detonation_pos is not None:
                    # 计算云团位置
                    cloud_pos = smoke_cloud_trajectory(detonation_pos, detonation_time, t)
                    
                    if cloud_pos is not None:
                        # 计算遮蔽效果
                        effect = calculate_shielding_effectiveness(
                            missile_idx, missile_pos, cloud_pos, detonation_time, t
                        )
                        is_effective = max(is_effective, effect)
        
        # 记录有效遮蔽区间
        if is_effective and not in_cloud:
            in_cloud = True
            start_time = t
        elif not is_effective and in_cloud:
            in_cloud = False
            effective_intervals.append((start_time, t))
    
    # 检查如果结束时仍在云团中
    if in_cloud and start_time is not None:
        effective_intervals.append((start_time, missile_total_time))
    
    return effective_intervals

def calculate_total_effective_time(time_intervals):
    """计算多个时间区间的总有效时长，处理重叠问题"""
    if not time_intervals:
        return 0
    
    # 排序时间区间
    sorted_intervals = sorted(time_intervals)
    
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

# 遗传算法相关函数
def create_individual():
    """创建个体（完整的防御策略）"""
    # 5架无人机，每架最多3枚烟幕弹，共15枚
    individual = []
    
    # 为每架无人机创建参数
    for drone_idx in range(5):
        # 飞行方向和速度（一旦确定就不变）
        direction = random.uniform(0, 360)
        speed = random.uniform(70, 140)
        
        # 最多投放3枚烟幕弹
        num_bombs = random.randint(1, 3)  # 随机决定投放数量
        release_times = []
        
        for _ in range(num_bombs):
            # 目标导弹
            target_missile = random.randint(0, 2)  # 0, 1, 2 对应 M1, M2, M3
            
            # 投放时间（确保同一无人机的投放间隔>=1秒）
            if not release_times:
                release_time = random.uniform(0, 10)
            else:
                release_time = max(release_times) + MIN_BOMB_INTERVAL + random.uniform(0, 2)
            release_times.append(release_time)
            
            # 起爆延迟
            detonation_delay = random.uniform(0, 2.0)
            
            # 添加参数到个体
            individual.extend([drone_idx, target_missile, direction, speed, release_time, detonation_delay])
    
    return individual

def check_constraints(individual):
    """检查个体是否满足约束条件"""
    # 解析个体
    drone_params = {}
    
    # 每6个参数代表一枚烟幕弹
    for i in range(0, len(individual), 6):
        drone_idx = int(individual[i])
        direction = individual[i+2]
        speed = individual[i+3]
        release_time = individual[i+4]
        
        # 初始化无人机参数
        if drone_idx not in drone_params:
            drone_params[drone_idx] = {
                'direction': direction,
                'speed': speed,
                'release_times': [release_time]
            }
        else:
            # 检查方向和速度是否一致
            if (drone_params[drone_idx]['direction'] != direction or 
                drone_params[drone_idx]['speed'] != speed):
                return False
            
            # 添加投放时间并检查间隔
            times = drone_params[drone_idx]['release_times']
            for t in times:
                if abs(release_time - t) < MIN_BOMB_INTERVAL:
                    return False
            times.append(release_time)
            drone_params[drone_idx]['release_times'] = sorted(times)
    
    # 检查每架无人机的烟幕弹数量是否超过3个
    for drone_idx, params in drone_params.items():
        if len(params['release_times']) > 3:
            return False
    
    return True
def fix_individual(individual):
    """修复个体使其满足约束条件"""
    # 解析个体
    drone_bombs = {}
    
    # 每6个参数代表一枚烟幕弹
    for i in range(0, len(individual), 6):
        drone_idx = int(individual[i])
        target_missile = int(individual[i+1])
        direction = individual[i+2]
        speed = individual[i+3]
        release_time = individual[i+4]
        detonation_delay = individual[i+5]
        
        # 收集每架无人机的烟幕弹
        if drone_idx not in drone_bombs:
            drone_bombs[drone_idx] = []
        
        drone_bombs[drone_idx].append({
            'index': i,
            'target_missile': target_missile,
            'direction': direction,
            'speed': speed,
            'release_time': release_time,
            'detonation_delay': detonation_delay
        })
    
    # 修复每架无人机的参数
    new_individual = []
    for drone_idx in range(5):
        bombs = drone_bombs.get(drone_idx, [])
        
        # 如果少于 3 枚，补充到 3 枚
        while len(bombs) < 3:
            direction = bombs[0]['direction'] if bombs else random.uniform(0, 360)
            speed = bombs[0]['speed'] if bombs else random.uniform(70, 140)
            release_time = max([b['release_time'] for b in bombs] + [0]) + MIN_BOMB_INTERVAL + random.uniform(0, 2)
            bombs.append({
                'index': -1,  # 新增的烟幕弹，索引待定
                'target_missile': random.randint(0, 2),
                'direction': direction,
                'speed': speed,
                'release_time': release_time,
                'detonation_delay': random.uniform(0, 2.0)
            })
        
        # 限制每架无人机最多 3 枚烟幕弹
        bombs = sorted(bombs, key=lambda x: x['release_time'])[:3]
        
        # 确保方向和速度一致
        first_direction = bombs[0]['direction']
        first_speed = bombs[0]['speed']
        
        # 确保投放间隔>=1秒
        for i in range(1, len(bombs)):
            prev_time = bombs[i-1]['release_time']
            curr_time = bombs[i]['release_time']
            if curr_time - prev_time < MIN_BOMB_INTERVAL:
                bombs[i]['release_time'] = prev_time + MIN_BOMB_INTERVAL + 0.1 * random.random()
        
        # 添加到新个体
        for bomb in bombs:
            new_individual.extend([
                drone_idx,
                bomb['target_missile'],
                first_direction,
                first_speed,
                bomb['release_time'],
                bomb['detonation_delay']
            ])
    
    return new_individual

def evaluate_individual(individual):
    """评估个体适应度（总有效遮蔽时长）"""
    # 解析个体为具体的防御计划
    defense_plans = {0: [], 1: [], 2: []}  # 每个导弹的防御计划
    
    # 每6个参数代表一枚烟幕弹
    for i in range(0, len(individual), 6):
        drone_idx = int(individual[i])
        target_missile = int(individual[i+1])
        direction = individual[i+2]
        speed = individual[i+3]
        release_time = individual[i+4]
        detonation_delay = individual[i+5]
        
        # 添加到对应导弹的防御计划
        defense_plans[target_missile].append({
            'drone_idx': drone_idx,
            'direction': direction,
            'speed': speed,
            'release_time': release_time,
            'detonation_delay': detonation_delay
        })
    
    # 计算每个导弹的有效遮蔽时间区间
    effective_times = []
    for missile_idx in range(3):
        intervals = calculate_missile_defense_effect(missile_idx, defense_plans[missile_idx])
        effective_time = calculate_total_effective_time(intervals)
        effective_times.append(effective_time)
    
    # 计算总有效遮蔽时长
    total_effective_time = sum(effective_times)
    
    # 计算平衡性指标（防止某些导弹被忽略）
    balance_penalty = 0
    for time in effective_times:
        if time < 1.0:  # 如果某个导弹几乎没有防御
            balance_penalty += 1
    
    # 返回考虑平衡性的适应度
    fitness = total_effective_time - balance_penalty
    
    # 防止负适应度
    return (max(0, fitness),)

def mutate_individual(individual, indpb):
    """变异操作"""
    # 每6个参数代表一枚烟幕弹
    for i in range(0, len(individual), 6):
        # 变异目标导弹
        if random.random() < indpb-0.25:
            individual[i+1] = random.randint(0, 2)
            individual[i+1] = individual[i+1] % 3

        # 变异方向
        if random.random() < indpb:
            individual[i+2] += random.gauss(-10, 10)
            individual[i+2] = individual[i+2] % 360
            if individual[i+2] < 0:
                individual[i+2] += 360
        
        # 变异速度
        if random.random() < indpb:
            individual[i+3] += random.gauss(-10, 10)
            individual[i+3] = ((individual[i+3]-70)%70)+70
            individual[i+3] = max(70, min(140, individual[i+3]))
        
        # 变异投放时间
        if random.random() < indpb:
            individual[i+4] += random.gauss(-4, 4)
            individual[i+4] =individual[i+4] % 40
            if individual[i+4] < 0:
                individual[i+4] = 0
        
        # 变异起爆延迟
        if random.random() < indpb:
            individual[i+5] += random.gauss(-4, 4)
            individual[i+5] = individual[i+5] % 15
            individual[i+5] = max(0, min(15.0, individual[i+5]))


    # 修复个体确保满足约束
    individual = fix_individual(individual)
    
    # 确保返回的是Individual对象
    if not isinstance(individual, creator.Individual):
        individual = creator.Individual(individual)

    return individual,

def crossover_individuals(ind1, ind2):
    """交叉操作"""
    # 按无人机分组交叉
    drone_bombs1 = {}
    drone_bombs2 = {}
    
    # 解析个体1
    for i in range(0, len(ind1), 6):
        drone_idx = int(ind1[i])
        if drone_idx not in drone_bombs1:
            drone_bombs1[drone_idx] = []
        drone_bombs1[drone_idx].append(ind1[i:i+6])
    
    # 解析个体2
    for i in range(0, len(ind2), 6):
        drone_idx = int(ind2[i])
        if drone_idx not in drone_bombs2:
            drone_bombs2[drone_idx] = []
        drone_bombs2[drone_idx].append(ind2[i:i+6])
    
    # 交叉每架无人机的烟幕弹配置
    for drone_idx in range(5):
        if drone_idx in drone_bombs1 and drone_idx in drone_bombs2:
            if random.random() < 0.5:
                # 交换整个无人机的配置
                temp = drone_bombs1[drone_idx]
                drone_bombs1[drone_idx] = drone_bombs2[drone_idx]
                drone_bombs2[drone_idx] = temp
    
    # 重建个体
    new_ind1 = []
    new_ind2 = []
    
    for drone_idx in range(5):
        if drone_idx in drone_bombs1:
            for bomb in drone_bombs1[drone_idx]:
                new_ind1.extend(bomb)
        
        if drone_idx in drone_bombs2:
            for bomb in drone_bombs2[drone_idx]:
                new_ind2.extend(bomb)
    
    # 修复个体确保满足约束
    new_ind1 = fix_individual(new_ind1)
    new_ind2 = fix_individual(new_ind2)
    
    # 转换为DEAP的Individual对象 - 这是关键修复
    new_ind1 = creator.Individual(new_ind1)
    new_ind2 = creator.Individual(new_ind2)
    
    return new_ind1, new_ind2

def decode_solution(individual):
    """解码个体为可读的防御策略"""
    solution = {
        'drones': {},
        'missiles': {0: [], 1: [], 2: []}
    }
    
    # 每6个参数代表一枚烟幕弹
    for i in range(0, len(individual), 6):
        drone_idx = int(individual[i])
        target_missile = int(individual[i+1])
        direction = individual[i+2]
        speed = individual[i+3]
        release_time = individual[i+4]
        detonation_delay = individual[i+5]
        
        # 更新无人机信息
        if drone_idx not in solution['drones']:
            solution['drones'][drone_idx] = {
                'direction': direction,
                'speed': speed,
                'bombs': []
            }
        
        # 添加烟幕弹信息
        bomb_info = {
            'target_missile': target_missile,
            'release_time': release_time,
            'detonation_delay': detonation_delay
        }
        solution['drones'][drone_idx]['bombs'].append(bomb_info)
        
        # 添加到导弹防御计划
        solution['missiles'][target_missile].append({
            'drone_idx': drone_idx,
            'direction': direction,
            'speed': speed,
            'release_time': release_time,
            'detonation_delay': detonation_delay
        })
    
    return solution

def run_genetic_algorithm(idx,population_size=100, generations=50):
    """运行遗传算法优化防御策略"""
    # 创建遗传算法工具箱
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 注册遗传操作
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # 创建多进程池
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    # 创建种群
    print("创建初始种群...")
    population = toolbox.population(n=population_size)
    
    # 添加种子解决方案
    # 为不同的导弹优先分配特定的无人机
    seed_individual = []
    
    # 为M1分配FY1和FY2
    seed_individual.extend([0, 0, 4.8, 140.0, 0.0,0.0])  # FY1-导弹M1-烟幕弹1
    seed_individual.extend([0, 0, 4.8, 140.0, 1.0, 0.0])  # FY1-导弹M1-烟幕弹2
    seed_individual.extend([1, 0, 270.0, 135.0, 6.0, 5.0]) # FY2-导弹M1-烟幕弹1
    seed_individual.extend([2, 0, 95.0, 130.0, 24.0, 3.0]) # FY3-导弹M1-烟幕弹3
    seed_individual.extend([3, 0, 300.0, 125.0, 12.0, 10.0]) # FY4-导弹M1-烟幕弹2
    seed_individual.extend([4, 0, 117.0, 135.0, 16.0, 6.0]) # FY5-导弹M3-烟幕弹1
    
    # 为M2分配FY4和部分FY2
    seed_individual.extend([3, 1, 300.0, 120.0, 4.0, 10.0]) # FY4-导弹M2-烟幕弹1
    seed_individual.extend([2, 1, 95.0, 130.0, 23.0, 4.0]) # FY3-导弹M2-烟幕弹2
    seed_individual.extend([1, 1, 10.0, 125.0, 4.0, 3.0]) # FY2-导弹M2-烟幕弹2
    seed_individual.extend([4, 1, 117.0, 125.0, 20.0, 0.0]) # FY5-导弹M3-烟幕弹1
    
    # 为M3分配FY3和FY5
    seed_individual.extend([2, 2, 95.0, 130.0, 22.0, 0.0]) # FY3-导弹M3-烟幕弹1
    seed_individual.extend([2, 2, 350.0, 130.0, 0.5, 0.5]) # FY3-导弹M3-烟幕弹1
    seed_individual.extend([2, 2, 350.0, 130.0, 2.0, 0.4]) # FY3-导弹M3-烟幕弹2
    seed_individual.extend([4, 2, 117.0, 125.0, 12.0, 1.0]) # FY5-导弹M3-烟幕弹1
    seed_individual.extend([1, 2, 10.0, 125.0, 8.0, 8.0]) # FY2-导弹M3-烟幕弹3
    
    # 修复种子个体并添加到种群
    seed_individual = fix_individual(seed_individual)
    population[0] = creator.Individual(seed_individual)
    
    # 评估初始种群
    print("评估初始种群...")
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # 记录最佳适应度
    best_fitness_values = []
    best_individual = None
    best_fitness = 0
    best_individuals = []
    
    # 进化
    print("开始进化...")
    for generation in range(generations):
        # 选择下一代
        print(f"第 {generation+1}/{generations} 代")
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # 应用交叉
        for i in range(1, len(offspring), 2):
            if random.random() < 0.7:  # 交叉概率
                offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values
                del offspring[i].fitness.values
        
        # 应用变异
        for i in range(len(offspring)):
            if random.random() < 0.3:  # 变异概率
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
        
        # 评估新一代中需要重新评估的个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # 替换种群
        population[:] = offspring
        
        # 找出当前代的最佳个体
        current_best = tools.selBest(population, 1)[0]
        current_best_fitness = current_best.fitness.values[0]
        best_fitness_values.append(current_best_fitness)
        
        # 更新全局最佳个体
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = toolbox.clone(current_best)
            print(f"第 {generation+1} 代: 新的最佳适应度 = {best_fitness:.4f}")
        
        if best_individual is not None:
            # 保存最佳个体到best_individuals列表
            best_individuals.append((generation+1, best_fitness, decode_solution(best_individual)))
        

        if (generation + 1) % 2 == 0:
            print(f"第 {generation+1} 代最佳适应度: {best_fitness:.4f}")
        
        if generation == generations - 1:
            #保存最后一代全部个体到excel
            df_all = pd.DataFrame(population, columns=[
                'drone_idx', 'target_missile', 'direction', 'speed', 'release_time', 'detonation_delay'
            ] * (len(best_individual) // 6))
            df_all.to_excel(f"{RESULTS_DIR}/{idx}/all_individuals_final_gen.xlsx", index=False)
            #保存best_individuals到excel
            df_best = pd.DataFrame(best_individuals, columns=['Generation', 'Best_Fitness', 'Solution'])
            df_best.to_excel(f"{RESULTS_DIR}/{idx}/best_individuals_over_gens.xlsx", index=False)
            print(f"已保存最后一代全部个体和历代最佳个体到 {RESULTS_DIR}/{idx} 目录下的Excel文件中。")


    # 关闭进程池
    pool.close()
    
    # 绘制进化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, generations + 1), best_fitness_values, 'b-', marker='o')
    plt.xlabel('代数')
    plt.ylabel('最佳适应度')
    plt.title('遗传算法进化过程')
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/{idx}/evolution_curve.png", dpi=300)
    
    # 返回最佳个体
    solution = decode_solution(best_individual)
    
    # 计算每个导弹的有效遮蔽时长
    missile_shielding_times = []
    for missile_idx in range(3):
        intervals = calculate_missile_defense_effect(missile_idx, solution['missiles'][missile_idx])
        effective_time = calculate_total_effective_time(intervals)
        missile_shielding_times.append(effective_time)
    
    # 保存结果
    solution['total_fitness'] = best_fitness
    solution['missile_shielding_times'] = missile_shielding_times
    
    # 保存最佳解决方案
    with open(f"{RESULTS_DIR}/{idx}/best_solution.pkl", 'wb') as f:
        pickle.dump(solution, f)
    
    print("优化完成!")
    print(f"总有效遮蔽时长: {best_fitness:.4f}秒")
    for i, time in enumerate(missile_shielding_times):
        print(f"导弹M{i+1}的有效遮蔽时长: {time:.4f}秒")
    
    return solution

def visualize_solution(idx,solution):
    """可视化防御策略"""
    # 创建3D图形
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制假目标和真目标
    ax.scatter([0], [0], [0], color='red', s=100, label='假目标')
    
    # 绘制真目标（圆柱体）
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, REAL_TARGET_HEIGHT, 10)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_cylinder = REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * np.cos(theta_grid)
    y_cylinder = REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * np.sin(theta_grid)
    ax.plot_surface(x_cylinder, y_cylinder, z_grid, alpha=0.5, color='blue', label='真目标')
    
    # 绘制导弹轨迹
    colors = ['darkred', 'darkgreen', 'darkblue']
    for missile_idx in range(3):
        missile_init_pos = MISSILE_POSITIONS[missile_idx]
        missile_total_time = missile_time_to_target(missile_init_pos, FAKE_TARGET, MISSILE_SPEED)
        
        # 计算轨迹点
        times = np.linspace(0, missile_total_time, 100)
        trajectory_points = []
        for t in times:
            pos = missile_trajectory(missile_init_pos, FAKE_TARGET, MISSILE_SPEED, t)
            trajectory_points.append(pos)
        
        trajectory_points = np.array(trajectory_points)
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
                color=colors[missile_idx], label=f'导弹M{missile_idx+1}轨迹')
        ax.scatter(missile_init_pos[0], missile_init_pos[1], missile_init_pos[2], 
                  color=colors[missile_idx], s=80, marker='^', label=f'导弹M{missile_idx+1}初始位置')
    
    # 绘制无人机轨迹和烟幕弹投放
    drone_colors = ['purple', 'orange', 'green', 'brown', 'pink']
    marker_styles = ['o', 's', 'p', '*', 'D']
    
    for drone_idx, drone_info in solution['drones'].items():
        direction = drone_info['direction']
        speed = drone_info['speed']
        bombs = drone_info['bombs']
        
        # 绘制无人机初始位置
        drone_init_pos = DRONE_POSITIONS[drone_idx]
        ax.scatter(drone_init_pos[0], drone_init_pos[1], drone_init_pos[2], 
                  color=drone_colors[drone_idx], s=100, marker=marker_styles[drone_idx], 
                  label=f'无人机FY{drone_idx+1}初始位置')
        
        # 计算无人机轨迹
        max_time = max([b['release_time'] for b in bombs]) + 1
        times = np.linspace(0, max_time, 50)
        trajectory_points = []
        for t in times:
            pos = drone_trajectory(drone_init_pos, direction, speed, t)
            trajectory_points.append(pos)
        
        trajectory_points = np.array(trajectory_points)
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
                color=drone_colors[drone_idx], linestyle='--', label=f'无人机FY{drone_idx+1}轨迹')
        
        # 绘制烟幕弹投放和爆炸
        for i, bomb in enumerate(bombs):
            target_missile = bomb['target_missile']
            release_time = bomb['release_time']
            detonation_delay = bomb['detonation_delay']
            detonation_time = release_time + detonation_delay
            
            # 计算投放点
            release_pos = drone_trajectory(drone_init_pos, direction, speed, release_time)
            ax.scatter(release_pos[0], release_pos[1], release_pos[2], 
                      color=colors[target_missile], s=50, alpha=0.7,
                      label=f'FY{drone_idx+1}烟幕弹{i+1}投放点(→M{target_missile+1})' if i == 0 else "")
            
            # 计算起爆点
            detonation_pos = smoke_trajectory(release_pos, release_time, direction, speed, detonation_time)
            if detonation_pos is not None:
                ax.scatter(detonation_pos[0], detonation_pos[1], detonation_pos[2], 
                          color=colors[target_missile], s=80, alpha=0.7, marker='*',
                          label=f'FY{drone_idx+1}烟幕弹{i+1}起爆点(→M{target_missile+1})' if i == 0 else "")
    
    # 设置坐标轴范围和标签
    ax.set_xlim([-1000, 21000])
    ax.set_ylim([-3500, 3500])
    ax.set_zlim([0, 2500])
    ax.set_xlabel('X轴 (m)')
    ax.set_ylabel('Y轴 (m)')
    ax.set_zlabel('Z轴 (m)')
    ax.set_title('多无人机协同防御策略可视化')
    
    # 优化图例显示（移除重复项）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{idx}/defense_strategy_3d.png", dpi=300)
    plt.close()
    
    # 创建时间线图
    plt.figure(figsize=(14, 8))
    
    # 为每个导弹创建子图
    for missile_idx in range(3):
        plt.subplot(3, 1, missile_idx+1)
        
        # 计算该导弹的有效遮蔽区间
        intervals = calculate_missile_defense_effect(missile_idx, solution['missiles'][missile_idx])
        
        # 绘制有效遮蔽区间
        for i, (start, end) in enumerate(intervals):
            plt.barh(0, end-start, left=start, height=0.6, color=colors[missile_idx], alpha=0.6)
            plt.text(start + (end-start)/2, 0, f"{end-start:.1f}s", 
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 设置标题和标签
        plt.title(f'导弹M{missile_idx+1}的有效遮蔽时间区间')
        plt.xlabel('时间 (s)')
        plt.yticks([])
        plt.grid(True, alpha=0.3)
        
        # 添加有效遮蔽总时长
        effective_time = calculate_total_effective_time(intervals)
        plt.text(0.05, 0.8, f"总有效遮蔽时长: {effective_time:.2f}秒", 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{idx}/shielding_timeline.png", dpi=300)
    plt.close()

def save_solution_to_excel(idx,solution):
    """将解决方案保存到Excel文件"""
    # 创建无人机配置表
    drone_data = []
    for drone_idx, drone_info in solution['drones'].items():
        for i, bomb in enumerate(drone_info['bombs']):
            drone_data.append({
                '无人机': f'FY{drone_idx+1}',
                '烟幕弹编号': i+1,
                '飞行方向(°)': round(drone_info['direction'], 2),
                '飞行速度(m/s)': round(drone_info['speed'], 2),
                '目标导弹': f'M{bomb["target_missile"]+1}',
                '投放时间(s)': round(bomb['release_time'], 2),
                '起爆延迟(s)': round(bomb['detonation_delay'], 2)
            })
    
    df_drones = pd.DataFrame(drone_data)
    
    # 创建导弹防御表
    missile_data = []
    for missile_idx in range(3):
        effective_time = solution['missile_shielding_times'][missile_idx]
        
        # 导弹总体信息
        missile_data.append({
            '导弹': f'M{missile_idx+1}',
            '烟幕弹编号': '',
            '防御无人机': '',
            '有效遮蔽时长(s)': round(effective_time, 2)
        })
        
        # 具体防御烟幕弹信息
        for i, defense in enumerate(solution['missiles'][missile_idx]):
            drone_idx = defense['drone_idx']
            missile_data.append({
                '导弹': f'M{missile_idx+1}',
                '烟幕弹编号': i+1,
                '防御无人机': f'FY{drone_idx+1}',
                '投放时间(s)': round(defense['release_time'], 2),
                '起爆延迟(s)': round(defense['detonation_delay'], 2),
                '有效遮蔽时长(s)': ''
            })
    
    df_missiles = pd.DataFrame(missile_data)
    
    # 创建总结表
    summary_data = [{
        '总有效遮蔽时长(s)': round(solution['total_fitness'], 2),
        'M1有效遮蔽时长(s)': round(solution['missile_shielding_times'][0], 2),
        'M2有效遮蔽时长(s)': round(solution['missile_shielding_times'][1], 2),
        'M3有效遮蔽时长(s)': round(solution['missile_shielding_times'][2], 2),
        '烟幕弹总数': sum(len(drone_info['bombs']) for drone_info in solution['drones'].values())
    }]
    
    df_summary = pd.DataFrame(summary_data)
    
    # 保存到Excel文件
    with pd.ExcelWriter(f"{RESULTS_DIR}/{idx}/defense_strategy.xlsx") as writer:
        df_drones.to_excel(writer, sheet_name='无人机配置', index=False)
        df_missiles.to_excel(writer, sheet_name='导弹防御', index=False)
        df_summary.to_excel(writer, sheet_name='总结', index=False)
    
    print(f"解决方案已保存到 {RESULTS_DIR}/{idx}/defense_strategy.xlsx")

def main():
    """主函数"""
    

    for i in range(40):
        
        if not os.path.exists(f"{RESULTS_DIR}/{i}"):
            os.makedirs(f"{RESULTS_DIR}/{i}")
        
        start_time = time.time()
        print("开始优化多无人机多导弹防御策略...")
        
        # 运行遗传算法
        solution = run_genetic_algorithm(i,population_size=100, generations=50)
        
        # 可视化解决方案
        print("生成可视化结果...")
        visualize_solution(i,solution)
        
        # 保存到Excel
        save_solution_to_excel(i,solution)
        
        end_time = time.time()
        print(f"总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()