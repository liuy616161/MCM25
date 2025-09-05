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
    if missile_pos[0] + 10 < cloud_pos[0]:
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

def validate_params(params):
    """验证参数是否合法"""
    direction, speed, t_r1, dt_d1, t_r2, dt_d2, t_r3, dt_d3 = params
    
    # 检查方向范围
    if not (0 <= direction < 360):
        return False
    
    # 检查速度范围
    if not (70 <= speed <= 140):
        return False
    
    # 检查投放时间顺序
    if not (t_r1 < t_r2 < t_r3):
        return False
    
    # 检查起爆延迟范围
    if not (0.1 <= dt_d1 <= 1.0 and 0.1 <= dt_d2 <= 1.0 and 0.1 <= dt_d3 <= 1.0):
        return False
    
    # 检查总投放时间约束
    if t_r3 + dt_d3 > 7.0:
        return False
    
    return True

def constrain_params(params):
    """约束参数在有效范围内"""
    direction, speed, t_r1, dt_d1, t_r2, dt_d2, t_r3, dt_d3 = params
    
    # 约束方向
    direction = direction % 360
    
    # 约束速度
    speed = max(70, min(140, speed))
    
    # 约束起爆延迟
    dt_d1 = max(0.1, min(1.0, dt_d1))
    dt_d2 = max(0.1, min(1.0, dt_d2))
    dt_d3 = max(0.1, min(1.0, dt_d3))
    
    # 约束投放时间顺序和总时间
    if t_r1 < 0: t_r1 = 0
    t_r2 = max(t_r1 + 0.1, t_r2)
    t_r3 = max(t_r2 + 0.1, t_r3)
    
    # 确保总投放时间不超过约束
    if t_r3 + dt_d3 > 7.0:
        scale = 7.0 / (t_r3 + dt_d3)
        t_r1 *= scale
        t_r2 *= scale
        t_r3 *= scale
    
    return [direction, speed, t_r1, dt_d1, t_r2, dt_d2, t_r3, dt_d3]

def adaptive_local_search(initial_params, iterations=50, initial_step_sizes=None):
    """自适应局部搜索算法"""
    if initial_step_sizes is None:
        initial_step_sizes = [1.0, 2.0, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05]
    
    best_params = initial_params
    best_result = calculate_multi_smoke_effect(best_params)
    best_fitness = best_result['effective_duration']
    
    step_sizes = initial_step_sizes.copy()
    
    for iteration in range(iterations):
        improved = False
        
        # 对每个参数进行扰动
        for i in range(len(best_params)):
            for direction in [1, -1]:
                new_params = best_params.copy()
                new_params[i] += direction * step_sizes[i]
                
                # 确保参数在有效范围内
                new_params = constrain_params(new_params)
                
                # 检查是否为有效参数
                if not validate_params(new_params):
                    continue
                
                result = calculate_multi_smoke_effect(new_params)
                fitness = result['effective_duration']
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = new_params
                    best_result = result
                    improved = True
                    print(f"迭代 {iteration}: 找到更好的解，有效时长: {best_fitness:.4f}")
        
        # 自适应调整步长
        if not improved:
            step_sizes = [s * 0.7 for s in step_sizes]  # 缩小搜索步长
            print(f"迭代 {iteration}: 无改进，缩小步长至 {step_sizes}")
        else:
            print(f"迭代 {iteration}: 当前最佳参数: {best_params}")
        
        # 步长过小时终止
        if max(step_sizes) < 0.01:
            print(f"步长过小，提前终止搜索，迭代次数: {iteration+1}/{iterations}")
            break
    
    return best_result

def genetic_algorithm(population_size=40, generations=30, elite_size=4, mutation_rate=0.1):
    """遗传算法优化"""
    # 从问题2的结果初始化方向和速度范围
    direction_range = (1, 10)
    speed_range = (130, 140)
    
    # 创建初始种群
    population = []
    for _ in range(population_size):
        # 创建一个随机个体
        direction = random.uniform(*direction_range)
        speed = random.uniform(*speed_range)
        
        # 随机生成三个投放时间和起爆延迟
        t_r1 = random.uniform(0.1, 2.0)
        dt_d1 = random.uniform(0.1, 1.0)
        
        t_r2 = random.uniform(t_r1 + 0.1, 4.0)
        dt_d2 = random.uniform(0.1, 1.0)
        
        t_r3 = random.uniform(t_r2 + 0.1, 6.0)
        dt_d3 = random.uniform(0.1, 1.0)
        
        # 确保总投放时间不超过约束
        if t_r3 + dt_d3 > 7.0:
            scale = 7.0 / (t_r3 + dt_d3)
            t_r1 *= scale
            t_r2 *= scale
            t_r3 *= scale
        
        params = [direction, speed, t_r1, dt_d1, t_r2, dt_d2, t_r3, dt_d3]
        population.append(params)
    
    # 初始种群加入一些手动优化的起点
    # 起点1：从问题2的最优解扩展
    population[0] = [5.0, 136.5, 0.3, 0.4, 2.5, 0.5, 4.7, 0.6]
    # 起点2：均匀间隔的投放
    population[1] = [5.0, 135.0, 0.2, 0.3, 2.7, 0.4, 5.2, 0.5]
    # 起点3：注重早期遮蔽
    population[2] = [5.0, 135.0, 0.1, 0.5, 1.5, 0.6, 3.0, 0.7]
    # 起点4：注重后期遮蔽
    population[3] = [5.0, 135.0, 1.0, 0.3, 3.0, 0.4, 5.0, 0.5]
    
    best_individual = None
    best_fitness = 0
    
    # 进化迭代
    for generation in range(generations):
        print(f"\n开始第 {generation+1}/{generations} 代进化")
        
        # 评估适应度
        fitness_results = []
        for individual in population:
            if not validate_params(individual):
                fitness_results.append(0)
                continue
            
            result = calculate_multi_smoke_effect(individual)
            fitness = result['effective_duration']
            fitness_results.append(fitness)
            
            # 更新全局最佳解
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
                print(f"第 {generation+1} 代: 找到更好的解，有效时长: {best_fitness:.4f}")
                print(f"参数: {best_individual}")
        
        # 如果已经到最后一代，跳出循环
        if generation == generations - 1:
            break
        
        # 选择精英
        elite_indices = np.argsort(fitness_results)[-elite_size:]
        elite = [population[i] for i in elite_indices]
        
        # 创建新一代
        new_population = elite.copy()
        
        # 通过选择、交叉和变异生成其余个体
        while len(new_population) < population_size:
            # 选择两个父代 (锦标赛选择)
            tournament_size = 3
            parent1_idx = max(random.sample(range(population_size), tournament_size), 
                             key=lambda i: fitness_results[i])
            parent2_idx = max(random.sample(range(population_size), tournament_size), 
                             key=lambda i: fitness_results[i])
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # 交叉
            if random.random() < 0.8:  # 80%概率进行交叉
                # 均匀交叉
                child = []
                for i in range(len(parent1)):
                    if random.random() < 0.5:
                        child.append(parent1[i])
                    else:
                        child.append(parent2[i])
            else:
                # 直接继承父代1
                child = parent1.copy()
            
            # 变异
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    # 根据不同参数使用不同的变异幅度
                    if i == 0:  # 方向
                        child[i] += random.uniform(-2, 2)
                    elif i == 1:  # 速度
                        child[i] += random.uniform(-5, 5)
                    else:  # 时间参数
                        child[i] += random.uniform(-0.3, 0.3)
            
            # 约束参数
            child = constrain_params(child)
            
            # 确保参数有效
            if validate_params(child):
                new_population.append(child)
        
        # 更新种群
        population = new_population
        
        # 显示本代最佳个体
        gen_best_idx = np.argmax(fitness_results)
        gen_best_fitness = fitness_results[gen_best_idx]
        gen_best_individual = population[gen_best_idx]
        
        print(f"第 {generation+1} 代最佳适应度: {gen_best_fitness:.4f}")
        print(f"参数: {gen_best_individual}")
    
    # 对最佳个体进行局部搜索微调
    print("\n对全局最佳解进行局部搜索微调...")
    best_result = adaptive_local_search(best_individual, iterations=20)
    
    return best_result

def particle_swarm_optimization(n_particles=30, iterations=40):
    """粒子群优化算法"""
    # 从问题2的结果初始化方向和速度范围
    direction_range = (1, 10)
    speed_range = (130, 140)
    
    # 参数范围定义
    param_ranges = [
        direction_range,           # 方向
        speed_range,               # 速度
        (0.1, 2.0),                # t_r1
        (0.1, 1.0),                # dt_d1
        (0.1, 4.0),                # t_r2
        (0.1, 1.0),                # dt_d2
        (0.1, 6.0),                # t_r3
        (0.1, 1.0)                 # dt_d3
    ]
    
    # 初始化粒子
    particles = []
    velocities = []
    for _ in range(n_particles):
        # 随机初始化粒子位置
        particle = []
        for i, (min_val, max_val) in enumerate(param_ranges):
            particle.append(random.uniform(min_val, max_val))
        
        # 确保参数合法
        particle = constrain_params(particle)
        particles.append(particle)
        
        # 初始化粒子速度
        velocity = []
        for i, (min_val, max_val) in enumerate(param_ranges):
            range_size = max_val - min_val
            velocity.append(random.uniform(-range_size * 0.1, range_size * 0.1))
        velocities.append(velocity)
    
    # 初始化粒子历史最佳位置和适应度
    best_positions = particles.copy()
    best_fitnesses = [0] * n_particles
    
    # 全局最佳位置和适应度
    global_best_position = None
    global_best_fitness = 0
    
    # 迭代优化
    for iteration in range(iterations):
        print(f"\n开始第 {iteration+1}/{iterations} 次迭代")
        
        # 评估每个粒子的适应度
        for i, particle in enumerate(particles):
            # 确保参数合法
            if not validate_params(particle):
                continue
            
            result = calculate_multi_smoke_effect(particle)
            fitness = result['effective_duration']
            
            # 更新个体历史最佳
            if fitness > best_fitnesses[i]:
                best_fitnesses[i] = fitness
                best_positions[i] = particle.copy()
                
                # 更新全局最佳
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle.copy()
                    print(f"第 {iteration+1} 次迭代: 找到更好的解，有效时长: {global_best_fitness:.4f}")
                    print(f"参数: {global_best_position}")
        
        # 更新粒子速度和位置
        w = 0.7  # 惯性权重
        c1 = 1.5  # 个体学习因子
        c2 = 1.5  # 社会学习因子
        
        for i in range(n_particles):
            for j in range(len(particles[i])):
                # 更新速度
                velocities[i][j] = (w * velocities[i][j] + 
                                  c1 * random.random() * (best_positions[i][j] - particles[i][j]) +
                                  c2 * random.random() * (global_best_position[j] - particles[i][j]))
                
                # 限制速度
                max_velocity = (param_ranges[j][1] - param_ranges[j][0]) * 0.1
                velocities[i][j] = max(-max_velocity, min(max_velocity, velocities[i][j]))
                
                # 更新位置
                particles[i][j] += velocities[i][j]
            
            # 约束参数
            particles[i] = constrain_params(particles[i])
        
        # 显示当前迭代最佳
        print(f"第 {iteration+1} 次迭代全局最佳适应度: {global_best_fitness:.4f}")
    
    # 对最佳粒子进行局部搜索微调
    print("\n对全局最佳解进行局部搜索微调...")
    best_result = adaptive_local_search(global_best_position, iterations=20)
    
    return best_result

def phased_optimization():
    """分阶段优化策略"""
    print("开始分阶段优化...")
    
    # 阶段一：基于问题2的结果优化第一枚烟幕弹
    direction = 5.0  # 问题2的最优方向
    speed = 136.5    # 问题2的最优速度
    
    # 阶段一：优化第一枚烟幕弹
    print("\n阶段一：优化第一枚烟幕弹")
    best_phase1 = None
    best_phase1_duration = 0
    
    for t_r1 in np.linspace(0.1, 1.0, 10):
        for dt_d1 in np.linspace(0.1, 1.0, 10):
            # 临时设置后两枚烟幕弹（暂不关注其效果）
            t_r2 = t_r1 + 2.0
            dt_d2 = 0.5
            t_r3 = t_r2 + 2.0
            dt_d3 = 0.5
            
            params = [direction, speed, t_r1, dt_d1, t_r2, dt_d2, t_r3, dt_d3]
            result = calculate_multi_smoke_effect(params)
            
            # 只关注第一枚烟幕弹的效果
            intervals1 = result['intervals'][0]
            duration1 = calculate_total_effective_time([intervals1])
            
            if duration1 > best_phase1_duration:
                best_phase1_duration = duration1
                best_phase1 = {
                    'params': params,
                    'duration': duration1,
                    'intervals': intervals1
                }
                print(f"阶段一: 找到更好的第一枚参数，有效时长: {duration1:.4f}")
                print(f"投放时间: {t_r1:.2f}, 起爆延迟: {dt_d1:.2f}")
    
    # 阶段二：基于第一枚优化第二枚
    print("\n阶段二：基于第一枚优化第二枚")
    best_phase2 = None
    best_phase2_duration = best_phase1_duration
    
    t_r1 = best_phase1['params'][2]
    dt_d1 = best_phase1['params'][3]
    intervals1 = best_phase1['intervals']
    
    # 找出第一枚效果结束的时间
    if intervals1:
        end_time1 = max([interval[1] for interval in intervals1])
        
        # 在第一枚效果结束前后优化第二枚
        for t_r2 in np.linspace(end_time1 - 2.0, end_time1 + 0.5, 10):
            t_r2 = max(t_r1 + 0.1, t_r2)  # 确保第二枚在第一枚之后投放
            
            for dt_d2 in np.linspace(0.1, 1.0, 10):
                # 临时设置第三枚烟幕弹
                t_r3 = t_r2 + 2.0
                dt_d3 = 0.5
                
                params = [direction, speed, t_r1, dt_d1, t_r2, dt_d2, t_r3, dt_d3]
                
                # 确保参数有效
                if not validate_params(params):
                    continue
                
                result = calculate_multi_smoke_effect(params)
                
                # 关注前两枚烟幕弹的总效果
                total_duration = calculate_total_effective_time([
                    result['intervals'][0], 
                    result['intervals'][1]
                ])
                
                if total_duration > best_phase2_duration:
                    best_phase2_duration = total_duration
                    best_phase2 = {
                        'params': params,
                        'duration': total_duration,
                        'intervals': result['intervals'][:2]
                    }
                    print(f"阶段二: 找到更好的前两枚参数，总有效时长: {total_duration:.4f}")
                    print(f"第二枚投放时间: {t_r2:.2f}, 起爆延迟: {dt_d2:.2f}")
    
    # 阶段三：基于前两枚优化第三枚
    print("\n阶段三：基于前两枚优化第三枚")
    best_phase3 = None
    best_phase3_duration = best_phase2_duration
    
    t_r1 = best_phase2['params'][2]
    dt_d1 = best_phase2['params'][3]
    t_r2 = best_phase2['params'][4]
    dt_d2 = best_phase2['params'][5]
    
    combined_intervals = best_phase2['intervals'][0] + best_phase2['intervals'][1]
    if combined_intervals:
        end_time2 = max([interval[1] for interval in combined_intervals])
        
        for t_r3 in np.linspace(end_time2 - 2.0, end_time2 + 0.5, 10):
            t_r3 = max(t_r2 + 0.1, t_r3)  # 确保第三枚在第二枚之后投放
            
            for dt_d3 in np.linspace(0.1, 1.0, 10):
                params = [direction, speed, t_r1, dt_d1, t_r2, dt_d2, t_r3, dt_d3]
                
                # 确保参数有效
                if not validate_params(params):
                    continue
                
                result = calculate_multi_smoke_effect(params)
                total_duration = result['effective_duration']
                
                if total_duration > best_phase3_duration:
                    best_phase3_duration = total_duration
                    best_phase3 = {
                        'params': params,
                        'duration': total_duration,
                        'result': result
                    }
                    print(f"阶段三: 找到更好的三枚参数，总有效时长: {total_duration:.4f}")
                    print(f"第三枚投放时间: {t_r3:.2f}, 起爆延迟: {dt_d3:.2f}")
    
    # 阶段四：局部联合优化
    print("\n阶段四：对最佳结果进行局部联合优化")
    if best_phase3:
        best_result = adaptive_local_search(
            best_phase3['params'], 
            iterations=50,
            initial_step_sizes=[0.5, 2.0, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05]
        )
    else:
        best_result = {'effective_duration': 0, 'params': [0]*8}
    
    print("\n分阶段优化完成")
    print(f"最终有效遮蔽时长: {best_result['effective_duration']:.4f}")
    print(f"最优参数: {best_result['params']}")
    
    return best_result

def hybrid_optimization():
    """混合优化策略：结合多种算法"""
    print("开始混合优化...")
    
    # 1. 分阶段优化
    print("\n===== 执行分阶段优化 =====")
    phased_result = phased_optimization()
    
    # 2. 遗传算法
    print("\n===== 执行遗传算法 =====")
    ga_result = genetic_algorithm()
    
    # 3. 粒子群算法
    print("\n===== 执行粒子群优化 =====")
    pso_result = particle_swarm_optimization()
    
    # 比较三种算法的结果
    results = [
        ("分阶段优化", phased_result),
        ("遗传算法", ga_result),
        ("粒子群优化", pso_result)
    ]
    
    best_method = max(results, key=lambda x: x[1]['effective_duration'])
    
    print("\n===== 优化结果比较 =====")
    for method, result in results:
        print(f"{method}有效遮蔽时长: {result['effective_duration']:.4f}")
    
    print(f"\n最佳方法: {best_method[0]}")
    print(f"最终有效遮蔽时长: {best_method[1]['effective_duration']:.4f}")
    print(f"最优参数: {best_method[1]['params']}")
    
    return best_method[1]

def visualize_multi_smoke_effect(result):
    """可视化多烟幕干扰弹效果"""
    params = result['params']
    direction, speed, t_r1, dt_d1, t_r2, dt_d2, t_r3, dt_d3 = params
    
    # 计算投放点和起爆点
    release_pos1 = drone_trajectory(DRONE_FY1_INIT, direction, speed, t_r1)
    detonation_time1 = t_r1 + dt_d1
    detonation_pos1 = smoke_trajectory(release_pos1, t_r1, direction, speed, detonation_time1)
    
    release_pos2 = drone_trajectory(DRONE_FY1_INIT, direction, speed, t_r2)
    detonation_time2 = t_r2 + dt_d2
    detonation_pos2 = smoke_trajectory(release_pos2, t_r2, direction, speed, detonation_time2)
    
    release_pos3 = drone_trajectory(DRONE_FY1_INIT, direction, speed, t_r3)
    detonation_time3 = t_r3 + dt_d3
    detonation_pos3 = smoke_trajectory(release_pos3, t_r3, direction, speed, detonation_time3)
    
    # 导弹到达假目标的时间
    missile_total_time = missile_time_to_target(MISSILE_M1_INIT, FAKE_TARGET, MISSILE_SPEED)
    
    # 创建3D图
    fig = plt.figure(figsize=(14, 10))
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
    drone_times = np.linspace(0, t_r3, 100)
    drone_positions = []
    for t in drone_times:
        drone_positions.append(drone_trajectory(DRONE_FY1_INIT, direction, speed, t))
    drone_positions = np.array(drone_positions)
    ax.plot(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2], 
            color='green', label='无人机FY1轨迹')
    ax.scatter(DRONE_FY1_INIT[0], DRONE_FY1_INIT[1], DRONE_FY1_INIT[2], 
               color='darkgreen', label='无人机FY1初始位置')
    
    # 绘制三个烟幕弹投放点和起爆点
    ax.scatter(release_pos1[0], release_pos1[1], release_pos1[2], 
               color='orange', s=80, label='烟幕弹1投放点')
    ax.scatter(detonation_pos1[0], detonation_pos1[1], detonation_pos1[2], 
               color='darkorange', s=80, label='烟幕弹1起爆点')
    
    ax.scatter(release_pos2[0], release_pos2[1], release_pos2[2], 
               color='purple', s=80, label='烟幕弹2投放点')
    ax.scatter(detonation_pos2[0], detonation_pos2[1], detonation_pos2[2], 
               color='darkviolet', s=80, label='烟幕弹2起爆点')
    
    ax.scatter(release_pos3[0], release_pos3[1], release_pos3[2], 
               color='cyan', s=80, label='烟幕弹3投放点')
    ax.scatter(detonation_pos3[0], detonation_pos3[1], detonation_pos3[2], 
               color='darkcyan', s=80, label='烟幕弹3起爆点')
    
    # 绘制有效遮蔽时间区间的烟幕云团
    colors = ['orange', 'purple', 'cyan']
    for i, (detonation_time, detonation_pos) in enumerate([
            (detonation_time1, detonation_pos1),
            (detonation_time2, detonation_pos2),
            (detonation_time3, detonation_pos3)
        ]):
        # 获取有效遮蔽时间区间
        intervals = result['intervals'][i] if 'intervals' in result else []
        
        # 对每个有效区间，绘制多个时间点的云团
        for interval_start, interval_end in intervals:
            for t in np.linspace(interval_start, interval_end, 3):
                cloud_pos = smoke_cloud_trajectory(detonation_pos, detonation_time, t)
                if cloud_pos is not None:
                    # 绘制烟幕云团
                    u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:15j]
                    x_sphere = cloud_pos[0] + SMOKE_EFFECTIVE_RADIUS * np.cos(u) * np.sin(v)
                    y_sphere = cloud_pos[1] + SMOKE_EFFECTIVE_RADIUS * np.sin(u) * np.sin(v)
                    z_sphere = cloud_pos[2] + SMOKE_EFFECTIVE_RADIUS * np.cos(v)
                    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color=colors[i], alpha=0.3, linewidth=0.5)
                    
                    # 绘制从导弹到真目标中心的线，表示遮蔽效果
                    missile_pos = missile_trajectory(MISSILE_M1_INIT, FAKE_TARGET, MISSILE_SPEED, t)
                    ax.plot([missile_pos[0], REAL_TARGET_CENTER[0]], 
                            [missile_pos[1], REAL_TARGET_CENTER[1]], 
                            [missile_pos[2], REAL_TARGET_CENTER[2]], 
                            'r--', alpha=0.3)
    
    # 添加遮蔽时长信息
    ax.text2D(0.05, 0.95, f"总有效遮蔽时长: {result['effective_duration']:.2f}秒", 
              transform=ax.transAxes, fontsize=12)
    
    # 添加每个烟幕弹的参数信息
    ax.text2D(0.05, 0.90, f"烟幕弹1: 投放时间={t_r1:.2f}s, 起爆延迟={dt_d1:.2f}s", 
              transform=ax.transAxes, fontsize=10, color='orange')
    ax.text2D(0.05, 0.87, f"烟幕弹2: 投放时间={t_r2:.2f}s, 起爆延迟={dt_d2:.2f}s", 
              transform=ax.transAxes, fontsize=10, color='purple')
    ax.text2D(0.05, 0.84, f"烟幕弹3: 投放时间={t_r3:.2f}s, 起爆延迟={dt_d3:.2f}s", 
              transform=ax.transAxes, fontsize=10, color='cyan')
    
    # 设置坐标轴范围和标签
    ax.set_xlim([0, 20000])
    ax.set_ylim([-500, 500])
    ax.set_zlim([0, 2500])
    ax.set_xlabel('X轴 (m)')
    ax.set_ylabel('Y轴 (m)')
    ax.set_zlabel('Z轴 (m)')
    ax.set_title('问题3: 三枚烟幕干扰弹投放策略')
    
    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels, loc='upper right')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/problem3_optimal_solution.png', dpi=300)
    plt.close()
    
    # 绘制时间线图，显示各烟幕弹的有效遮蔽区间
    plt.figure(figsize=(12, 6))
    
    colors = ['orange', 'purple', 'cyan']
    labels = ['烟幕弹1', '烟幕弹2', '烟幕弹3']
    
    for i, intervals in enumerate(result['intervals']):
        for start, end in intervals:
            plt.barh(i, end - start, left=start, height=0.6, color=colors[i], alpha=0.7)
            plt.text(start + (end - start) / 2, i, f"{end - start:.1f}s", 
                     ha='center', va='center', fontweight='bold')
    
    plt.yticks([0, 1, 2], labels)
    plt.xlabel('时间 (秒)')
    plt.title('三枚烟幕干扰弹的有效遮蔽时间区间')
    plt.grid(True, alpha=0.3)
    
    # 添加总有效遮蔽时长信息
    plt.figtext(0.5, 0.01, f"总有效遮蔽时长: {result['effective_duration']:.2f}秒", 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/problem3_shielding_timeline.png', dpi=300)
    plt.close()

def save_result_to_excel(result):
    """保存结果到Excel文件"""
    direction, speed, t_r1, dt_d1, t_r2, dt_d2, t_r3, dt_d3 = result['params']
    
    # 计算投放点和起爆点
    release_pos1 = drone_trajectory(DRONE_FY1_INIT, direction, speed, t_r1)
    detonation_time1 = t_r1 + dt_d1
    detonation_pos1 = smoke_trajectory(release_pos1, t_r1, direction, speed, detonation_time1)
    
    release_pos2 = drone_trajectory(DRONE_FY1_INIT, direction, speed, t_r2)
    detonation_time2 = t_r2 + dt_d2
    detonation_pos2 = smoke_trajectory(release_pos2, t_r2, direction, speed, detonation_time2)
    
    release_pos3 = drone_trajectory(DRONE_FY1_INIT, direction, speed, t_r3)
    detonation_time3 = t_r3 + dt_d3
    detonation_pos3 = smoke_trajectory(release_pos3, t_r3, direction, speed, detonation_time3)
    
    # 创建结果DataFrame
    result_data = {
        '参数': [
            '飞行方向', '飞行速度',
            '烟幕弹1投放时间', '烟幕弹1起爆延迟',
            '烟幕弹2投放时间', '烟幕弹2起爆延迟',
            '烟幕弹3投放时间', '烟幕弹3起爆延迟',
            '总有效遮蔽时长'
        ],
        '值': [
            f"{direction:.2f}°",
            f"{speed:.2f} m/s",
            f"{t_r1:.2f} s",
            f"{dt_d1:.2f} s",
            f"{t_r2:.2f} s",
            f"{dt_d2:.2f} s",
            f"{t_r3:.2f} s",
            f"{dt_d3:.2f} s",
            f"{result['effective_duration']:.2f} s"
        ]
    }
    
    df_params = pd.DataFrame(result_data)
    
    # 创建投放点和起爆点DataFrame
    positions_data = {
        '点位': [
            '烟幕弹1投放点', '烟幕弹1起爆点',
            '烟幕弹2投放点', '烟幕弹2起爆点',
            '烟幕弹3投放点', '烟幕弹3起爆点'
        ],
        'X坐标 (m)': [
            f"{release_pos1[0]:.2f}",
            f"{detonation_pos1[0]:.2f}",
            f"{release_pos2[0]:.2f}",
            f"{detonation_pos2[0]:.2f}",
            f"{release_pos3[0]:.2f}",
            f"{detonation_pos3[0]:.2f}"
        ],
        'Y坐标 (m)': [
            f"{release_pos1[1]:.2f}",
            f"{detonation_pos1[1]:.2f}",
            f"{release_pos2[1]:.2f}",
            f"{detonation_pos2[1]:.2f}",
            f"{release_pos3[1]:.2f}",
            f"{detonation_pos3[1]:.2f}"
        ],
        'Z坐标 (m)': [
            f"{release_pos1[2]:.2f}",
            f"{detonation_pos1[2]:.2f}",
            f"{release_pos2[2]:.2f}",
            f"{detonation_pos2[2]:.2f}",
            f"{release_pos3[2]:.2f}",
            f"{detonation_pos3[2]:.2f}"
        ]
    }
    
    df_positions = pd.DataFrame(positions_data)
    
    # 创建有效遮蔽时间区间DataFrame
    intervals_data = []
    for i, intervals in enumerate(result['intervals']):
        for j, (start, end) in enumerate(intervals):
            intervals_data.append({
                '烟幕弹': f"烟幕弹{i+1}",
                '区间序号': j+1,
                '开始时间 (s)': f"{start:.2f}",
                '结束时间 (s)': f"{end:.2f}",
                '时长 (s)': f"{end - start:.2f}"
            })
    
    df_intervals = pd.DataFrame(intervals_data)
    
    # 保存到Excel文件，不同表放在不同sheet
    with pd.ExcelWriter(f'{RESULTS_DIR}/problem3_optimal_solution.xlsx') as writer:
        df_params.to_excel(writer, sheet_name='最优参数', index=False)
        df_positions.to_excel(writer, sheet_name='投放点和起爆点', index=False)
        df_intervals.to_excel(writer, sheet_name='有效遮蔽时间区间', index=False)
    
    print(f"已保存结果到 {RESULTS_DIR}/problem3_optimal_solution.xlsx")

def main():
    """主函数"""
    start_time = time.time()
    print("开始优化问题3: 三枚烟幕干扰弹投放策略...")
    
    # 使用混合优化策略
    result = hybrid_optimization()
    
    # 可视化结果
    print("\n开始可视化最优结果...")
    visualize_multi_smoke_effect(result)
    
    # 保存结果到Excel
    save_result_to_excel(result)
    
    end_time = time.time()
    print(f"\n优化完成，总耗时: {end_time - start_time:.2f}秒")
    print(f"最终有效遮蔽时长: {result['effective_duration']:.4f}秒")
    
    direction, speed, t_r1, dt_d1, t_r2, dt_d2, t_r3, dt_d3 = result['params']
    print("\n最优烟幕干扰弹投放策略:")
    print(f"飞行方向: {direction:.2f}°")
    print(f"飞行速度: {speed:.2f} m/s")
    print(f"烟幕弹1: 投放时间={t_r1:.2f}s, 起爆延迟={dt_d1:.2f}s")
    print(f"烟幕弹2: 投放时间={t_r2:.2f}s, 起爆延迟={dt_d2:.2f}s")
    print(f"烟幕弹3: 投放时间={t_r3:.2f}s, 起爆延迟={dt_d3:.2f}s")
    
    return result

if __name__ == "__main__":
    main()