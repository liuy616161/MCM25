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
from functools import lru_cache
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import json

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

# 轨迹计算函数
@lru_cache(maxsize=10000)
def missile_trajectory_cached(init_pos_tuple, target_pos_tuple, speed, t):
    """缓存版本的导弹轨迹计算"""
    init_pos = np.array(init_pos_tuple)
    target_pos = np.array(target_pos_tuple)
    direction = normalize(target_pos - init_pos)
    return tuple(init_pos + direction * speed * t)

def missile_trajectory(init_pos, target_pos, speed, t):
    """计算导弹在时间t的位置"""
    return np.array(missile_trajectory_cached(tuple(init_pos), tuple(target_pos), speed, t))

@lru_cache(maxsize=10000)
def drone_trajectory_cached(init_pos_tuple, direction_deg, speed, t):
    """缓存版本的无人机轨迹计算"""
    init_pos = np.array(init_pos_tuple)
    direction_rad = np.radians(direction_deg)
    direction_vector = np.array([np.cos(direction_rad), np.sin(direction_rad), 0])
    return tuple(init_pos + direction_vector * speed * t)

def drone_trajectory(init_pos, direction_deg, speed, t):
    """计算无人机在时间t的位置"""
    return np.array(drone_trajectory_cached(tuple(init_pos), direction_deg, speed, t))

@lru_cache(maxsize=10000)
def smoke_trajectory_cached(release_pos_tuple, release_time, direction_deg, speed, t):
    """缓存版本的烟幕弹轨迹计算"""
    if t < release_time:
        return None
    
    release_pos = np.array(release_pos_tuple)
    direction_rad = np.radians(direction_deg)
    direction_vector = np.array([np.cos(direction_rad), np.sin(direction_rad), 0])
    dt = t - release_time
    
    return tuple(release_pos + np.array([0, 0, -0.5 * GRAVITY * dt**2]) + direction_vector * speed * dt)

def smoke_trajectory(release_pos, release_time, direction_deg, speed, t):
    """计算烟幕弹在时间t的位置（考虑重力）"""
    result = smoke_trajectory_cached(tuple(release_pos), release_time, direction_deg, speed, t)
    return None if result is None else np.array(result)

@lru_cache(maxsize=10000)
def smoke_cloud_trajectory_cached(detonation_pos_tuple, detonation_time, t):
    """缓存版本的烟幕云团轨迹计算"""
    if t < detonation_time:
        return None
    
    detonation_pos = np.array(detonation_pos_tuple)
    dt = t - detonation_time
    return tuple(detonation_pos + np.array([0, 0, -SMOKE_DESCENT_RATE * dt]))

def smoke_cloud_trajectory(detonation_pos, detonation_time, t):
    """计算烟幕云团在时间t的位置"""
    result = smoke_cloud_trajectory_cached(tuple(detonation_pos), detonation_time, t)
    return None if result is None else np.array(result)

# 预计算导弹轨迹
def precompute_missile_trajectories():
    """预计算导弹轨迹，避免重复计算"""
    missile_trajectories = {}
    # 使用自适应时间步长
    time_steps = []
    # 早期阶段使用更细的时间步长
    time_steps.extend(np.arange(0, 20, 0.01))
    # 中期阶段使用较粗的时间步长
    time_steps.extend(np.arange(20, 40, 0.05))
    # 后期阶段使用更粗的时间步长
    time_steps.extend(np.arange(40, 70, 0.1))
    
    print("预计算导弹轨迹...")
    for missile_idx in range(3):
        missile_init_pos = MISSILE_POSITIONS[missile_idx]
        missile_trajectories[missile_idx] = {}
        for t in tqdm(time_steps, desc=f"导弹M{missile_idx+1}轨迹", leave=False):
            pos = missile_trajectory(missile_init_pos, FAKE_TARGET, MISSILE_SPEED, t)
            missile_trajectories[missile_idx][t] = pos
    
    return missile_trajectories

# 锥体遮蔽判断
def is_target_in_shadow_cone(missile_pos, cloud_pos, target_pos, target_radius, target_height):
    """检查目标是否在烟幕云团投射的阴影锥体内 - 优化版本"""
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
    
    # 优化：先检查目标中心点是否在锥体内，如果远离锥体，直接返回False
    target_center_vector = REAL_TARGET_CENTER - missile_pos
    distance_center = np.linalg.norm(target_center_vector)
    
    if distance_center > 0:
        cos_center_angle = np.dot(target_center_vector, unit_direction) / distance_center
        # 如果目标中心点与锥体中心线夹角过大，可能整个目标都不在锥体内
        if cos_center_angle > 0 or abs(cos_center_angle) < cos_theta * 0.5:
            return False
    
    # 检查目标圆柱体的关键点
    # 优化：根据目标与锥体的相对位置，动态调整采样点数量
    relative_position = abs(np.dot(normalize(REAL_TARGET_CENTER - missile_pos), unit_direction))
    
    # 如果目标靠近锥体边缘，使用更多采样点；否则使用较少采样点
    num_points = 16 if relative_position > 0.8 else 8
    
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    # 生成圆柱体表面的采样点 - 向量化操作
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    # 底面和顶面的圆周点
    x_coords = target_pos[0] + target_radius * cos_angles
    y_coords = target_pos[1] + target_radius * sin_angles
    
    # 合并底面点
    bottom_points = np.column_stack([x_coords, y_coords, np.full(num_points, target_pos[2])])
    
    # 合并顶面点
    top_points = np.column_stack([x_coords, y_coords, np.full(num_points, target_pos[2] + target_height)])
    
    # 合并所有点
    all_points = np.vstack([bottom_points, top_points])
    
    # 向量化检查所有点
    vectors_to_points = all_points - missile_pos
    distances = np.linalg.norm(vectors_to_points, axis=1)
    
    # 计算夹角余弦值
    cos_angles = np.sum(vectors_to_points * unit_direction, axis=1) / distances
    
    # 检查是否有点不在锥体内
    for cos_angle in cos_angles:
        if cos_angle > 0:  # 点在导弹背后
            return False
        
        if abs(cos_angle) < abs(cos_theta):  # 点在锥体外
            return False
    
    # 所有点都在阴影锥体内
    return True

def calculate_shielding_effectiveness(missile_idx, missile_pos, cloud_pos, cloud_start_time, current_time):
    """计算烟幕对导弹的遮蔽效果"""
    if cloud_pos is None or current_time - cloud_start_time > SMOKE_EFFECTIVE_TIME:
        return 0  # 未形成云团或云团已失效
    
    # 检查真目标是否被烟幕遮蔽
    target_shielded = is_target_in_shadow_cone(
        missile_pos, cloud_pos, REAL_TARGET_CENTER, REAL_TARGET_RADIUS, REAL_TARGET_HEIGHT
    )
    
    return 1 if target_shielded else 0

# 防御效果计算模块
class DefenseSimulator:
    """防御模拟器类 - 封装模拟逻辑"""
    
    def __init__(self, use_precomputed=True):
        """初始化防御模拟器"""
        self.use_precomputed = use_precomputed
        if use_precomputed:
            self.missile_trajectories = precompute_missile_trajectories()
        else:
            self.missile_trajectories = None
        
        # 计算各导弹到达假目标的时间
        self.missile_total_times = []
        for missile_idx in range(3):
            missile_init_pos = MISSILE_POSITIONS[missile_idx]
            total_time = missile_time_to_target(missile_init_pos, FAKE_TARGET, MISSILE_SPEED)
            self.missile_total_times.append(total_time)
    
    def get_missile_position(self, missile_idx, t):
        """获取导弹在时间t的位置"""
        if self.use_precomputed:
            # 使用预计算轨迹 - 找到最接近的时间点
            times = list(self.missile_trajectories[missile_idx].keys())
            closest_time = min(times, key=lambda x: abs(x - t))
            return self.missile_trajectories[missile_idx][closest_time]
        else:
            # 实时计算
            missile_init_pos = MISSILE_POSITIONS[missile_idx]
            return missile_trajectory(missile_init_pos, FAKE_TARGET, MISSILE_SPEED, t)
    
    def calculate_missile_defense_effect(self, missile_idx, defense_plan, time_step=0.01):
        """计算特定导弹的防御效果（有效遮蔽时间区间）"""
        # 导弹到达假目标的时间
        missile_total_time = self.missile_total_times[missile_idx]
        
        # 优化：按时间排序防御计划，避免每个时间点都检查所有烟幕弹
        sorted_defense = sorted(defense_plan, key=lambda x: x['release_time'] + x['detonation_delay'])
        
        # 确定最早和最晚的可能防御时间
        if not sorted_defense:
            return []
        
        earliest_defense = sorted_defense[0]['release_time'] + sorted_defense[0]['detonation_delay']
        latest_defense = sorted_defense[-1]['release_time'] + sorted_defense[-1]['detonation_delay'] + SMOKE_EFFECTIVE_TIME
        
        # 确定模拟的时间范围
        start_time = max(0, earliest_defense - 1)  # 提前1秒开始模拟
        end_time = min(missile_total_time, latest_defense + 1)  # 延后1秒结束模拟
        
        # 模拟导弹和烟幕的轨迹
        effective_intervals = []
        in_cloud = False
        start_time_record = None
        
        # 自适应时间步长
        if end_time - start_time > 30:
            # 如果模拟时间很长，使用较大的时间步长
            time_steps = np.arange(start_time, end_time, 0.05)
        else:
            # 否则使用标准时间步长
            time_steps = np.arange(start_time, end_time, time_step)
        
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
        for t in time_steps:
            missile_pos = self.get_missile_position(missile_idx, t)
            
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
                        missile_idx, missile_pos, cloud_pos, detonation_time, t
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
    
    def calculate_defense_effectiveness(self, defense_plans):
        """计算整体防御效果"""
        effective_times = []
        intervals_by_missile = {}
        
        for missile_idx in range(3):
            intervals = self.calculate_missile_defense_effect(
                missile_idx, defense_plans[missile_idx]
            )
            intervals_by_missile[missile_idx] = intervals
            effective_time = calculate_total_effective_time(intervals)
            effective_times.append(effective_time)
        
        return {
            'total_time': sum(effective_times),
            'times_by_missile': effective_times,
            'intervals_by_missile': intervals_by_missile
        }

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

# 遗传算法相关函数和类
class GeneticOptimizer:
    """遗传优化器类 - 封装GA逻辑"""
    
    def __init__(self, simulator, config=None):
        """初始化遗传优化器"""
        self.simulator = simulator
        
        # 默认配置
        default_config = {
            'population_size': 100,
            'generations': 50,
            'crossover_prob': 0.7,
            'mutation_prob': 0.3,
            'mutation_indpb': 0.2,
            'tournament_size': 3,
            'early_stop_generations': 10,
            'seed_solutions': True
        }
        
        # 使用用户配置覆盖默认配置
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # 初始化DEAP工具
        self.setup_deap()
    
    def setup_deap(self):
        """设置DEAP遗传算法框架"""
        # 创建适应度类和个体类
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # 创建工具箱
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 注册遗传操作
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", self.crossover_individuals)
        self.toolbox.register("mutate", self.mutate_individual, 
                             indpb=self.config['mutation_indpb'])
        self.toolbox.register("select", tools.selTournament, 
                             tournsize=self.config['tournament_size'])
    
    def create_individual(self):
        """创建个体（完整的防御策略）"""
        individual = []
        
        # 为每架无人机创建参数
        for drone_idx in range(5):
            # 飞行方向和速度
            direction = random.uniform(0, 360)
            speed = random.uniform(70, 140)
            
            # 每架无人机3枚烟幕弹
            release_times = []
            
            for _ in range(3):
                # 目标导弹
                target_missile = random.randint(0, 2)
                
                # 投放时间（确保同一无人机的投放间隔>=1秒）
                if not release_times:
                    release_time = random.uniform(0, 15)
                else:
                    release_time = max(release_times) + MIN_BOMB_INTERVAL + random.uniform(0, 2)
                release_times.append(release_time)
                
                # 起爆延迟
                detonation_delay = random.uniform(0, 2.0)
                
                # 添加参数到个体
                individual.extend([drone_idx, target_missile, direction, speed, 
                                 release_time, detonation_delay])
        
        return individual
    
    def fix_individual(self, individual):
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
        for drone_idx in range(5):
            bombs = drone_bombs.get(drone_idx, [])
            
            # 确保每架无人机有3枚烟幕弹
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
            
            # 限制每架无人机最多3枚烟幕弹
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
            
            # 更新无人机的烟幕弹
            drone_bombs[drone_idx] = bombs
        
        # 重建个体
        new_individual = []
        for drone_idx in range(5):
            bombs = drone_bombs[drone_idx]
            
            # 重建每架无人机的烟幕弹
            for bomb in bombs:
                direction = bomb['direction']
                speed = bomb['speed']
                
                new_individual.extend([
                    drone_idx,
                    bomb['target_missile'],
                    direction,
                    speed,
                    bomb['release_time'],
                    bomb['detonation_delay']
                ])
        
        return new_individual
    
    def parse_defense_plans(self, individual):
        """解析个体为防御计划"""
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
        
        return defense_plans
    
    def evaluate_individual(self, individual):
        """评估个体适应度（增强版本）"""
        # 解析防御计划
        defense_plans = self.parse_defense_plans(individual)
        
        # 计算防御效果
        results = self.simulator.calculate_defense_effectiveness(defense_plans)
        
        # 基本适应度 - 总有效遮蔽时长
        total_effective_time = results['total_time']
        missile_times = results['times_by_missile']
        
        # 平衡性奖励/惩罚
        balance_reward = 0
        min_time = min(missile_times)
        
        # 如果最短防御时间太短，给予惩罚
        if min_time < 3.0:
            balance_reward -= (3.0 - min_time) * 1.5  # 平衡性惩罚
        
        # 如果所有导弹都有良好防御，给予奖励
        if min_time >= 5.0:
            balance_reward += 3.0
        
        # 烟幕弹使用效率奖励/惩罚
        efficiency_factor = 0
        
        # 计算每枚烟幕弹的平均有效时长
        total_bombs = sum(len(plans) for plans in defense_plans.values())
        avg_effect_per_bomb = total_effective_time / total_bombs if total_bombs > 0 else 0
        
        # 如果平均效果好，给予奖励
        if avg_effect_per_bomb > 2.0:
            efficiency_factor += 2.0
        
        # 最终适应度
        fitness = total_effective_time + balance_reward + efficiency_factor
        
        return (max(0, fitness),)
    
    def mutate_individual(self, individual, indpb):
        """变异操作 - 自适应变异"""
        # 复制个体
        mutant = individual.copy()
        
        # 每6个参数代表一枚烟幕弹
        for i in range(0, len(mutant), 6):
            # 变异目标导弹
            if random.random() < indpb * 0.75:
                mutant[i+1] = random.randint(0, 2)
            
            # 变异方向 - 在已有方向基础上小幅度变化
            if random.random() < indpb:
                mutant[i+2] += random.gauss(0, 10)  # 标准差10度
                mutant[i+2] = mutant[i+2] % 360
                if mutant[i+2] < 0:
                    mutant[i+2] += 360
            
            # 变异速度 - 在速度范围内小幅度变化
            if random.random() < indpb:
                mutant[i+3] += random.gauss(0, 5)  # 标准差5m/s
                mutant[i+3] = max(70, min(140, mutant[i+3]))
            
            # 变异投放时间 - 小幅度变化
            if random.random() < indpb:
                mutant[i+4] += random.gauss(0, 2)  # 标准差2秒
                mutant[i+4] = max(0, mutant[i+4])
            
            # 变异起爆延迟 - 小幅度变化
            if random.random() < indpb:
                mutant[i+5] += random.gauss(0, 0.5)  # 标准差0.5秒
                mutant[i+5] = max(0, min(5.0, mutant[i+5]))
        
        # 修复变异后的个体
        mutant = self.fix_individual(mutant)
        
        # 确保返回的是Individual对象
        if not isinstance(mutant, creator.Individual):
            mutant = creator.Individual(mutant)
        
        return mutant,
    
    def crossover_individuals(self, ind1, ind2):
        """交叉操作 - 多策略交叉"""
        # 随机选择交叉策略
        strategy = random.choice(['drone_based', 'missile_based', 'uniform'])
        
        if strategy == 'drone_based':
            # 按无人机分组交叉
            new_ind1, new_ind2 = self._crossover_drone_based(ind1, ind2)
        elif strategy == 'missile_based':
            # 按目标导弹分组交叉
            new_ind1, new_ind2 = self._crossover_missile_based(ind1, ind2)
        else:
            # 均匀交叉 - 以烟幕弹为单位
            new_ind1, new_ind2 = self._crossover_uniform(ind1, ind2)
        
        # 修复交叉后的个体
        new_ind1 = self.fix_individual(new_ind1)
        new_ind2 = self.fix_individual(new_ind2)
        
        # 转换为DEAP的Individual对象
        new_ind1 = creator.Individual(new_ind1)
        new_ind2 = creator.Individual(new_ind2)
        
        return new_ind1, new_ind2
    
    def _crossover_drone_based(self, ind1, ind2):
        """按无人机分组的交叉操作"""
        # 解析个体
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
        
        return new_ind1, new_ind2
    
    def _crossover_missile_based(self, ind1, ind2):
        """按目标导弹分组的交叉操作"""
        # 解析个体
        missile_bombs1 = {0: [], 1: [], 2: []}
        missile_bombs2 = {0: [], 1: [], 2: []}
        
        # 解析个体1
        for i in range(0, len(ind1), 6):
            target_missile = int(ind1[i+1])
            missile_bombs1[target_missile].append(ind1[i:i+6])
        
        # 解析个体2
        for i in range(0, len(ind2), 6):
            target_missile = int(ind2[i+1])
            missile_bombs2[target_missile].append(ind2[i:i+6])
        
        # 随机选择一个或多个导弹目标进行交换
        num_to_swap = random.randint(1, 2)
        targets_to_swap = random.sample(range(3), num_to_swap)
        
        for target in targets_to_swap:
            temp = missile_bombs1[target]
            missile_bombs1[target] = missile_bombs2[target]
            missile_bombs2[target] = temp
        
        # 重建个体
        new_ind1 = []
        new_ind2 = []
        
        for target in range(3):
            for bomb in missile_bombs1[target]:
                new_ind1.extend(bomb)
            
            for bomb in missile_bombs2[target]:
                new_ind2.extend(bomb)
        
        return new_ind1, new_ind2
    
    def _crossover_uniform(self, ind1, ind2):
        """均匀交叉 - 以烟幕弹为单位"""
        # 解析为烟幕弹单位
        bombs1 = [ind1[i:i+6] for i in range(0, len(ind1), 6)]
        bombs2 = [ind2[i:i+6] for i in range(0, len(ind2), 6)]
        
        # 创建新个体
        new_bombs1 = []
        new_bombs2 = []
        
        # 对每个位置随机选择来源
        for i in range(min(len(bombs1), len(bombs2))):
            if random.random() < 0.5:
                new_bombs1.append(bombs1[i])
                new_bombs2.append(bombs2[i])
            else:
                new_bombs1.append(bombs2[i])
                new_bombs2.append(bombs1[i])
        
        # 处理长度不同的情况
        if len(bombs1) > len(bombs2):
            new_bombs1.extend(bombs1[len(bombs2):])
        elif len(bombs2) > len(bombs1):
            new_bombs2.extend(bombs2[len(bombs1):])
        
        # 展平为个体
        new_ind1 = []
        new_ind2 = []
        
        for bomb in new_bombs1:
            new_ind1.extend(bomb)
        
        for bomb in new_bombs2:
            new_ind2.extend(bomb)
        
        return new_ind1, new_ind2
    
    def create_seed_solutions(self):
        """创建种子解决方案"""
        # 种子解决方案1 - 均匀分配策略
        seed1 = []
        
        # 为每架无人机分配任务
        for drone_idx in range(5):
            direction = random.uniform(0, 360)
            speed = random.uniform(120, 140)
            
            # 每架无人机防御一个主要导弹，辅助另一个导弹
            primary_missile = drone_idx % 3
            secondary_missile = (primary_missile + 1) % 3
            
            # 添加2枚针对主要导弹的烟幕弹
            seed1.extend([drone_idx, primary_missile, direction, speed, 0.5, 0.5])
            seed1.extend([drone_idx, primary_missile, direction, speed, 2.0, 0.5])
            
            # 添加1枚针对次要导弹的烟幕弹
            seed1.extend([drone_idx, secondary_missile, direction, speed, 3.5, 0.5])
        
        # 种子解决方案2 - 基于距离的分配策略
        seed2 = []
        
        # 针对M1的防御 - 使用FY1和FY2（距离最近）
        seed2.extend([0, 0, 5.0, 130.0, 0.5, 0.5])  # FY1-导弹M1-烟幕弹1
        seed2.extend([0, 0, 5.0, 130.0, 2.0, 0.5])  # FY1-导弹M1-烟幕弹2
        seed2.extend([0, 0, 5.0, 130.0, 3.5, 0.5])  # FY1-导弹M1-烟幕弹3
        seed2.extend([1, 0, 10.0, 125.0, 0.5, 0.5]) # FY2-导弹M1-烟幕弹1
        seed2.extend([1, 0, 10.0, 125.0, 2.0, 0.5]) # FY2-导弹M1-烟幕弹2
        
        # 针对M2的防御 - 使用FY2和FY4（位置适合）
        seed2.extend([1, 1, 10.0, 125.0, 3.5, 0.5]) # FY2-导弹M2-烟幕弹3
        seed2.extend([3, 1, 30.0, 120.0, 0.5, 0.5]) # FY4-导弹M2-烟幕弹1
        seed2.extend([3, 1, 30.0, 120.0, 2.0, 0.5]) # FY4-导弹M2-烟幕弹2
        seed2.extend([3, 1, 30.0, 120.0, 3.5, 0.5]) # FY4-导弹M2-烟幕弹3
        
        # 针对M3的防御 - 使用FY3和FY5（位置适合）
        seed2.extend([2, 2, 350.0, 130.0, 0.5, 0.5]) # FY3-导弹M3-烟幕弹1
        seed2.extend([2, 2, 350.0, 130.0, 2.0, 0.5]) # FY3-导弹M3-烟幕弹2
        seed2.extend([2, 2, 350.0, 130.0, 3.5, 0.5]) # FY3-导弹M3-烟幕弹3
        seed2.extend([4, 2, 340.0, 125.0, 0.5, 0.5]) # FY5-导弹M3-烟幕弹1
        seed2.extend([4, 2, 340.0, 125.0, 2.0, 0.5]) # FY5-导弹M3-烟幕弹2
        
        # 修复种子解决方案
        seed1 = self.fix_individual(seed1)
        seed2 = self.fix_individual(seed2)
        
        return [creator.Individual(seed1), creator.Individual(seed2)]
    
    def run(self, run_id="default"):
        """运行遗传算法优化"""
        # 创建结果目录
        run_dir = f"{RESULTS_DIR}/{run_id}"
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        
        # 创建多进程池
        pool = multiprocessing.Pool()
        self.toolbox.register("map", pool.map)
        
        # 创建种群
        print(f"运行 {run_id}: 创建初始种群...")
        population = self.toolbox.population(n=self.config['population_size'])
        
        # 添加种子解决方案
        if self.config['seed_solutions']:
            seed_solutions = self.create_seed_solutions()
            for i, seed in enumerate(seed_solutions):
                if i < len(population):
                    population[i] = seed
        
        # 评估初始种群
        print(f"运行 {run_id}: 评估初始种群...")
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # 记录统计信息
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # 记录最佳适应度和个体
        best_fitness_values = []
        best_individual = None
        best_fitness = 0
        best_individuals = []
        
        # 早停变量
        generations_no_improvement = 0
        
        # 进化
        print(f"运行 {run_id}: 开始进化...")
        for generation in range(self.config['generations']):
            # 选择下一代
            print(f"运行 {run_id}: 第 {generation+1}/{self.config['generations']} 代")
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # 应用交叉
            for i in range(1, len(offspring), 2):
                if random.random() < self.config['crossover_prob']:
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
            
            # 应用变异
            for i in range(len(offspring)):
                if random.random() < self.config['mutation_prob']:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
            
            # 评估新一代中需要重新评估的个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # 替换种群
            population[:] = offspring
            
            # 统计信息
            record = stats.compile(population)
            
            # 找出当前代的最佳个体
            current_best = tools.selBest(population, 1)[0]
            current_best_fitness = current_best.fitness.values[0]
            best_fitness_values.append(current_best_fitness)
            
            # 更新全局最佳个体
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = self.toolbox.clone(current_best)
                print(f"运行 {run_id}: 第 {generation+1} 代: 新的最佳适应度 = {best_fitness:.4f}")
                generations_no_improvement = 0
            else:
                generations_no_improvement += 1
            
            # 记录每代最佳个体
            if best_individual is not None:
                solution = self.decode_solution(best_individual)
                best_individuals.append({
                    'generation': generation+1,
                    'fitness': best_fitness,
                    'solution': solution
                })
            
            # 每隔几代输出统计信息
            if (generation + 1) % 5 == 0 or generation == 0:
                print(f"运行 {run_id}: 第 {generation+1} 代统计: Avg={record['avg']:.4f}, "
                      f"Max={record['max']:.4f}, Min={record['min']:.4f}, Std={record['std']:.4f}")
            
            # 早停检查
            if generations_no_improvement >= self.config['early_stop_generations']:
                print(f"运行 {run_id}: {self.config['early_stop_generations']}代内无改进，提前停止")
                break
        
        # 关闭进程池
        pool.close()
        
        # 绘制进化曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(best_fitness_values) + 1), best_fitness_values, 'b-', marker='o')
        plt.xlabel('代数')
        plt.ylabel('最佳适应度')
        plt.title(f'遗传算法进化过程 ({run_id})')
        plt.grid(True)
        plt.savefig(f"{run_dir}/evolution_curve.png", dpi=300)
        plt.close()
        
        # 保存进化数据
        evolution_data = {
            'generation': list(range(1, len(best_fitness_values) + 1)),
            'best_fitness': best_fitness_values
        }
        pd.DataFrame(evolution_data).to_csv(f"{run_dir}/evolution_data.csv", index=False)
        
        # 保存最佳个体及其解码结果
        if best_individual is not None:
            solution = self.decode_solution(best_individual)
            with open(f"{run_dir}/best_solution.pkl", 'wb') as f:
                pickle.dump(solution, f)
            
            # 保存历代最佳个体
            with open(f"{run_dir}/best_individuals.pkl", 'wb') as f:
                pickle.dump(best_individuals, f)
            
            # 计算详细的防御效果
            defense_plans = self.parse_defense_plans(best_individual)
            results = self.simulator.calculate_defense_effectiveness(defense_plans)
            
            print(f"运行 {run_id}: 优化完成!")
            print(f"总有效遮蔽时长: {results['total_time']:.4f}秒")
            for i, time in enumerate(results['times_by_missile']):
                print(f"导弹M{i+1}的有效遮蔽时长: {time:.4f}秒")
            
            # 返回最佳解决方案
            return solution
        
        return None
    
    def decode_solution(self, individual):
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
        
        # 计算防御效果
        defense_plans = self.parse_defense_plans(individual)
        results = self.simulator.calculate_defense_effectiveness(defense_plans)
        
        solution['total_fitness'] = results['total_time']
        solution['missile_shielding_times'] = results['times_by_missile']
        solution['intervals_by_missile'] = results['intervals_by_missile']
        
        return solution

# 可视化和结果分析函数
def visualize_solution(solution, run_id):
    """可视化防御策略 - 增强版"""
    run_dir = f"{RESULTS_DIR}/{run_id}"
    
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
    missile_labels = ['M1', 'M2', 'M3']
    
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
                color=colors[missile_idx], label=f'导弹{missile_labels[missile_idx]}轨迹')
        ax.scatter(missile_init_pos[0], missile_init_pos[1], missile_init_pos[2], 
                  color=colors[missile_idx], s=80, marker='^', label=f'导弹{missile_labels[missile_idx]}初始位置')
    
    # 绘制无人机轨迹和烟幕弹投放
    drone_colors = ['purple', 'orange', 'green', 'brown', 'pink']
    marker_styles = ['o', 's', 'p', '*', 'D']
    drone_labels = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
    
    for drone_idx, drone_info in solution['drones'].items():
        direction = drone_info['direction']
        speed = drone_info['speed']
        bombs = drone_info['bombs']
        
        # 绘制无人机初始位置
        drone_init_pos = DRONE_POSITIONS[drone_idx]
        ax.scatter(drone_init_pos[0], drone_init_pos[1], drone_init_pos[2], 
                  color=drone_colors[drone_idx], s=100, marker=marker_styles[drone_idx], 
                  label=f'无人机{drone_labels[drone_idx]}初始位置')
        
        # 计算无人机轨迹
        max_time = max([b['release_time'] for b in bombs]) + 1 if bombs else 0
        times = np.linspace(0, max_time, 50)
        trajectory_points = []
        for t in times:
            pos = drone_trajectory(drone_init_pos, direction, speed, t)
            trajectory_points.append(pos)
        
        trajectory_points = np.array(trajectory_points)
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
                color=drone_colors[drone_idx], linestyle='--', label=f'无人机{drone_labels[drone_idx]}轨迹')
        
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
                      label=f'{drone_labels[drone_idx]}烟幕弹{i+1}投放点(→{missile_labels[target_missile]})' 
                            if i == 0 else "")
            
            # 计算起爆点
            detonation_pos = smoke_trajectory(release_pos, release_time, direction, speed, detonation_time)
            if detonation_pos is not None:
                ax.scatter(detonation_pos[0], detonation_pos[1], detonation_pos[2], 
                          color=colors[target_missile], s=80, alpha=0.7, marker='*',
                          label=f'{drone_labels[drone_idx]}烟幕弹{i+1}起爆点(→{missile_labels[target_missile]})' 
                                if i == 0 else "")
                
                # 绘制云团
                # 简化表示，只绘制代表性球体
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = detonation_pos[0] + SMOKE_EFFECTIVE_RADIUS * np.cos(u) * np.sin(v)
                y = detonation_pos[1] + SMOKE_EFFECTIVE_RADIUS * np.sin(u) * np.sin(v)
                z = detonation_pos[2] + SMOKE_EFFECTIVE_RADIUS * np.cos(v)
                ax.plot_wireframe(x, y, z, color=colors[target_missile], alpha=0.1)
    
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
    plt.savefig(f"{run_dir}/defense_strategy_3d.png", dpi=300)
    plt.close()
    
    # 创建时间线图
    plt.figure(figsize=(14, 10))
    
    # 为每个导弹创建子图
    for missile_idx in range(3):
        plt.subplot(3, 1, missile_idx+1)
        
        # 获取该导弹的遮蔽区间
        if 'intervals_by_missile' in solution:
            intervals = solution['intervals_by_missile'][missile_idx]
        else:
            # 使用模拟器计算
            simulator = DefenseSimulator(use_precomputed=False)
            intervals = simulator.calculate_missile_defense_effect(
                missile_idx, solution['missiles'][missile_idx]
            )
        
        # 绘制有效遮蔽区间
        for i, (start, end) in enumerate(intervals):
            plt.barh(0, end-start, left=start, height=0.6, color=colors[missile_idx], alpha=0.6)
            plt.text(start + (end-start)/2, 0, f"{end-start:.1f}s", 
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 设置标题和标签
        plt.title(f'导弹{missile_labels[missile_idx]}的有效遮蔽时间区间')
        plt.xlabel('时间 (s)')
        plt.yticks([])
        plt.grid(True, alpha=0.3)
        
                # 添加有效遮蔽总时长
        effective_time = calculate_total_effective_time(intervals)
        plt.text(0.05, 0.8, f"总有效遮蔽时长: {effective_time:.2f}秒", 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{run_dir}/shielding_timeline.png", dpi=300)
    plt.close()
    
    # 创建交互式3D可视化（使用Plotly）
    try:
        # 创建图形
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
        
        # 添加假目标
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='假目标'
        ))
        
        # 添加真目标（简化为点阵）
        cylinder_points = []
        for theta in np.linspace(0, 2*np.pi, 20):
            for z in np.linspace(0, REAL_TARGET_HEIGHT, 5):
                x = REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * np.cos(theta)
                y = REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * np.sin(theta)
                cylinder_points.append([x, y, z + REAL_TARGET_CENTER[2]])
        
        cylinder_points = np.array(cylinder_points)
        fig.add_trace(go.Scatter3d(
            x=cylinder_points[:, 0], y=cylinder_points[:, 1], z=cylinder_points[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.5),
            name='真目标'
        ))
        
        # 添加导弹轨迹
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
            fig.add_trace(go.Scatter3d(
                x=trajectory_points[:, 0], y=trajectory_points[:, 1], z=trajectory_points[:, 2],
                mode='lines',
                line=dict(color=colors[missile_idx], width=3),
                name=f'导弹{missile_labels[missile_idx]}轨迹'
            ))
            
            # 添加导弹初始位置
            fig.add_trace(go.Scatter3d(
                x=[missile_init_pos[0]], y=[missile_init_pos[1]], z=[missile_init_pos[2]],
                mode='markers',
                marker=dict(size=8, color=colors[missile_idx], symbol='diamond'),
                name=f'导弹{missile_labels[missile_idx]}初始位置'
            ))
        
        # 添加无人机轨迹和烟幕弹
        for drone_idx, drone_info in solution['drones'].items():
            direction = drone_info['direction']
            speed = drone_info['speed']
            bombs = drone_info['bombs']
            
            # 无人机初始位置
            drone_init_pos = DRONE_POSITIONS[drone_idx]
            fig.add_trace(go.Scatter3d(
                x=[drone_init_pos[0]], y=[drone_init_pos[1]], z=[drone_init_pos[2]],
                mode='markers',
                marker=dict(size=8, color=drone_colors[drone_idx]),
                name=f'无人机{drone_labels[drone_idx]}初始位置'
            ))
            
            # 无人机轨迹
            max_time = max([b['release_time'] for b in bombs]) + 1 if bombs else 0
            times = np.linspace(0, max_time, 50)
            trajectory_points = []
            for t in times:
                pos = drone_trajectory(drone_init_pos, direction, speed, t)
                trajectory_points.append(pos)
            
            trajectory_points = np.array(trajectory_points)
            fig.add_trace(go.Scatter3d(
                x=trajectory_points[:, 0], y=trajectory_points[:, 1], z=trajectory_points[:, 2],
                mode='lines',
                line=dict(color=drone_colors[drone_idx], width=2, dash='dash'),
                name=f'无人机{drone_labels[drone_idx]}轨迹'
            ))
            
            # 烟幕弹投放和爆炸点
            for i, bomb in enumerate(bombs):
                target_missile = bomb['target_missile']
                release_time = bomb['release_time']
                detonation_delay = bomb['detonation_delay']
                detonation_time = release_time + detonation_delay
                
                # 投放点
                release_pos = drone_trajectory(drone_init_pos, direction, speed, release_time)
                fig.add_trace(go.Scatter3d(
                    x=[release_pos[0]], y=[release_pos[1]], z=[release_pos[2]],
                    mode='markers',
                    marker=dict(size=6, color=colors[target_missile]),
                    name=f'{drone_labels[drone_idx]}烟幕弹{i+1}投放点(→{missile_labels[target_missile]})'
                ))
                
                # 起爆点
                detonation_pos = smoke_trajectory(release_pos, release_time, direction, speed, detonation_time)
                if detonation_pos is not None:
                    fig.add_trace(go.Scatter3d(
                        x=[detonation_pos[0]], y=[detonation_pos[1]], z=[detonation_pos[2]],
                        mode='markers',
                        marker=dict(size=10, color=colors[target_missile], symbol='star'),
                        name=f'{drone_labels[drone_idx]}烟幕弹{i+1}起爆点(→{missile_labels[target_missile]})'
                    ))
        
        # 更新布局
        fig.update_layout(
            title='多无人机协同防御策略3D交互可视化',
            scene=dict(
                xaxis_title='X轴 (m)',
                yaxis_title='Y轴 (m)',
                zaxis_title='Z轴 (m)',
                aspectmode='data'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # 保存为HTML文件
        fig.write_html(f"{run_dir}/interactive_visualization.html")
    except Exception as e:
        print(f"创建交互式可视化时出错: {e}")

def save_solution_to_excel(solution, run_id):
    """将解决方案保存到Excel文件"""
    run_dir = f"{RESULTS_DIR}/{run_id}"
    
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
    
    # 创建遮蔽时间区间表
    intervals_data = []
    for missile_idx in range(3):
        if 'intervals_by_missile' in solution:
            intervals = solution['intervals_by_missile'][missile_idx]
            for i, (start, end) in enumerate(intervals):
                intervals_data.append({
                    '导弹': f'M{missile_idx+1}',
                    '区间编号': i+1,
                    '开始时间(s)': round(start, 2),
                    '结束时间(s)': round(end, 2),
                    '持续时间(s)': round(end - start, 2)
                })
    
    df_intervals = pd.DataFrame(intervals_data)
    
    # 保存到Excel文件
    with pd.ExcelWriter(f"{run_dir}/defense_strategy.xlsx") as writer:
        df_drones.to_excel(writer, sheet_name='无人机配置', index=False)
        df_missiles.to_excel(writer, sheet_name='导弹防御', index=False)
        df_summary.to_excel(writer, sheet_name='总结', index=False)
        df_intervals.to_excel(writer, sheet_name='遮蔽时间区间', index=False)
    
    print(f"解决方案已保存到 {run_dir}/defense_strategy.xlsx")

def run_experiment(experiment_name, configurations):
    """运行实验，测试不同配置"""
    # 创建实验目录
    experiment_dir = f"{RESULTS_DIR}/{experiment_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    # 记录实验结果
    results = {}
    
    # 创建模拟器
    simulator = DefenseSimulator()
    
    # 对每种配置运行实验
    for config_name, config in configurations.items():
        config_results = []
        
        # 对每种配置运行多次实验
        for run in range(config['runs']):
            run_id = f"{experiment_name}/{config_name}/run{run+1}"
            
            # 创建优化器
            optimizer = GeneticOptimizer(simulator, config)
            
            # 运行优化
            solution = optimizer.run(run_id)
            
            if solution:
                config_results.append({
                    'run': run+1,
                    'total_fitness': solution['total_fitness'],
                    'missile_times': solution['missile_shielding_times'],
                    'solution': solution
                })
        
        # 计算统计量
        if config_results:
            fitness_values = [result['total_fitness'] for result in config_results]
            best_run = max(config_results, key=lambda x: x['total_fitness'])
            
            results[config_name] = {
                'mean': np.mean(fitness_values),
                'std': np.std(fitness_values),
                'max': np.max(fitness_values),
                'min': np.min(fitness_values),
                'best_run': best_run['run'],
                'best_solution': best_run['solution']
            }
    
    # 保存结果汇总
    result_summary = []
    for config_name, result in results.items():
        result_summary.append({
            '配置名称': config_name,
            '平均适应度': round(result['mean'], 2),
            '标准差': round(result['std'], 2),
            '最大适应度': round(result['max'], 2),
            '最小适应度': round(result['min'], 2),
            '最佳运行': result['best_run']
        })
    
    df_summary = pd.DataFrame(result_summary)
    df_summary.to_excel(f"{experiment_dir}/experiment_summary.xlsx", index=False)
    
    # 绘制比较图表
    plt.figure(figsize=(12, 8))
    
    # 绘制总适应度比较
    config_names = list(results.keys())
    mean_values = [result['mean'] for result in results.values()]
    std_values = [result['std'] for result in results.values()]
    
    plt.bar(config_names, mean_values, yerr=std_values, capsize=10, alpha=0.7)
    plt.ylabel('平均适应度(总有效遮蔽时长)')
    plt.title('不同配置的性能比较')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{experiment_dir}/configuration_comparison.png", dpi=300)
    plt.close()
    
    # 为每个配置的最佳解决方案生成可视化
    for config_name, result in results.items():
        best_solution = result['best_solution']
        visualize_solution(best_solution, f"{experiment_name}/{config_name}/best")
        save_solution_to_excel(best_solution, f"{experiment_name}/{config_name}/best")
    
    return results

def main():
    """主函数 - 实验设计"""
    print("开始多无人机多导弹防御系统优化实验")
    
    # 定义实验配置
    experiment_configurations = {
        'baseline': {
            'population_size': 100,
            'generations': 50,
            'crossover_prob': 0.7,
            'mutation_prob': 0.3,
            'mutation_indpb': 0.2,
            'tournament_size': 3,
            'early_stop_generations': 10,
            'seed_solutions': True,
            'runs': 3  # 每种配置运行次数
        },
        'large_population': {
            'population_size': 200,
            'generations': 25,
            'crossover_prob': 0.7,
            'mutation_prob': 0.3,
            'mutation_indpb': 0.2,
            'tournament_size': 3,
            'early_stop_generations': 10,
            'seed_solutions': True,
            'runs': 3
        },
        'high_mutation': {
            'population_size': 100,
            'generations': 50,
            'crossover_prob': 0.6,
            'mutation_prob': 0.4,
            'mutation_indpb': 0.3,
            'tournament_size': 3,
            'early_stop_generations': 10,
            'seed_solutions': True,
            'runs': 3
        },
        'no_seeds': {
            'population_size': 100,
            'generations': 50,
            'crossover_prob': 0.7,
            'mutation_prob': 0.3,
            'mutation_indpb': 0.2,
            'tournament_size': 3,
            'early_stop_generations': 10,
            'seed_solutions': False,
            'runs': 3
        }
    }
    
    # 运行实验
    results = run_experiment('parameter_study', experiment_configurations)
    
    # 找出最佳配置
    best_config = max(results.items(), key=lambda x: x[1]['max'])
    print(f"\n实验完成! 最佳配置: {best_config[0]}")
    print(f"最大有效遮蔽时长: {best_config[1]['max']:.2f}秒")
    print(f"平均有效遮蔽时长: {best_config[1]['mean']:.2f}秒")
    
    # 运行最终优化
    print("\n使用最佳配置进行最终优化...")
    
    # 使用最佳配置，但增加迭代次数
    final_config = experiment_configurations[best_config[0]].copy()
    final_config['generations'] = 100
    final_config['population_size'] = 150
    final_config['runs'] = 1
    
    simulator = DefenseSimulator()
    optimizer = GeneticOptimizer(simulator, final_config)
    
    # 运行最终优化
    final_solution = optimizer.run("final_solution")
    
    # 可视化和保存最终结果
    if final_solution:
        visualize_solution(final_solution, "final_solution")
        save_solution_to_excel(final_solution, "final_solution")
        
        print("\n最终优化完成!")
        print(f"总有效遮蔽时长: {final_solution['total_fitness']:.2f}秒")
        for i, time in enumerate(final_solution['missile_shielding_times']):
            print(f"导弹M{i+1}的有效遮蔽时长: {time:.2f}秒")
    
    print("\n所有结果已保存到目录:", RESULTS_DIR)

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    random.seed(42)
    
    # 运行主函数
    main()