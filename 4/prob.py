import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import random
from joblib import Parallel, delayed

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

# 导弹初始位置
MISSILE_M1_INIT = np.array([20000, 0, 2000])
MISSILE_M2_INIT = np.array([19000, 600, 2100])
MISSILE_M3_INIT = np.array([18000, -600, 1900])

# 无人机初始位置
DRONE_FY1_INIT = np.array([17800, 0, 1800])
DRONE_FY2_INIT = np.array([12000, 1400, 1400])
DRONE_FY3_INIT = np.array([6000, -3000, 700])
DRONE_FY4_INIT = np.array([11000, 2000, 1800])
DRONE_FY5_INIT = np.array([13000, -2000, 1300])

# 为问题4使用的无人机
DRONES_PROBLEM4 = [DRONE_FY1_INIT, DRONE_FY2_INIT, DRONE_FY3_INIT]
DRONE_NAMES = ["FY1", "FY2", "FY3"]

# 创建结果目录
RESULTS_DIR = "optimization_results_q4"
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
    if missile_pos[0] - 10 < cloud_pos[0]:
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

def calculate_single_smoke_effect(drone_init_pos, drone_direction, drone_speed, release_time, detonation_delay, time_step=0.01):
    """计算单个烟幕干扰弹的效果，返回有效遮蔽的时间区间列表"""
    detonation_time = release_time + detonation_delay
    
    # 计算投放点
    release_pos = drone_trajectory(drone_init_pos, drone_direction, drone_speed, release_time)
    
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

def calculate_multi_drone_effect(params):
    """计算多无人机联合投放烟幕干扰弹的总效果
    
    参数格式:
    params = [
        dir1, spd1, t_r1, dt_d1,  # FY1的参数
        dir2, spd2, t_r2, dt_d2,  # FY2的参数
        dir3, spd3, t_r3, dt_d3   # FY3的参数
    ]
    """
    # 解析参数
    dir1, spd1, t_r1, dt_d1, dir2, spd2, t_r2, dt_d2, dir3, spd3, t_r3, dt_d3 = params
    
    # 计算三个无人机的烟幕弹效果
    intervals1 = calculate_single_smoke_effect(DRONE_FY1_INIT, dir1, spd1, t_r1, dt_d1)
    intervals2 = calculate_single_smoke_effect(DRONE_FY2_INIT, dir2, spd2, t_r2, dt_d2)
    intervals3 = calculate_single_smoke_effect(DRONE_FY3_INIT, dir3, spd3, t_r3, dt_d3)
    
    # 计算总有效遮蔽时长
    total_time = calculate_total_effective_time([intervals1, intervals2, intervals3])
    
    # 计算各投放点和起爆点
    release_pos1 = drone_trajectory(DRONE_FY1_INIT, dir1, spd1, t_r1)
    detonation_pos1 = smoke_trajectory(release_pos1, t_r1, dir1, spd1, t_r1 + dt_d1)
    
    release_pos2 = drone_trajectory(DRONE_FY2_INIT, dir2, spd2, t_r2)
    detonation_pos2 = smoke_trajectory(release_pos2, t_r2, dir2, spd2, t_r2 + dt_d2)
    
    release_pos3 = drone_trajectory(DRONE_FY3_INIT, dir3, spd3, t_r3)
    detonation_pos3 = smoke_trajectory(release_pos3, t_r3, dir3, spd3, t_r3 + dt_d3)
    
    return {
        'params': params,
        'effective_duration': total_time,
        'intervals': [intervals1, intervals2, intervals3],
        'release_positions': [release_pos1, release_pos2, release_pos3],
        'detonation_positions': [detonation_pos1, detonation_pos2, detonation_pos3],
        'detonation_times': [t_r1 + dt_d1, t_r2 + dt_d2, t_r3 + dt_d3]
    }

def validate_params(params):
    """验证参数是否合法"""
    dir1, spd1, t_r1, dt_d1, dir2, spd2, t_r2, dt_d2, dir3, spd3, t_r3, dt_d3 = params
    
    # 检查方向范围
    if not all(0 <= d < 360 for d in [dir1, dir2, dir3]):
        return False
    
    # 检查速度范围
    if not all(70 <= s <= 140 for s in [spd1, spd2, spd3]):
        return False
    
    # 检查投放时间和起爆延迟
    if not all(t >= 0 for t in [t_r1, t_r2, t_r3, dt_d1, dt_d2, dt_d3]):
        return False
    
    # 检查起爆延迟范围
    if not all(0.1 <= d <= 2.0 for d in [dt_d1, dt_d2, dt_d3]):
        return False
    
    # 检查总投放时间约束
    if not all(t_r + dt_d <= 15.0 for t_r, dt_d in [(t_r1, dt_d1), (t_r2, dt_d2), (t_r3, dt_d3)]):
        return False
    
    return True

def constrain_params(params):
    """约束参数在有效范围内"""
    dir1, spd1, t_r1, dt_d1, dir2, spd2, t_r2, dt_d2, dir3, spd3, t_r3, dt_d3 = params
    
    # 约束方向
    dir1 = dir1 % 360
    dir2 = dir2 % 360
    dir3 = dir3 % 360
    
    # 约束速度
    spd1 = max(70, min(140, spd1))
    spd2 = max(70, min(140, spd2))
    spd3 = max(70, min(140, spd3))
    
    # 约束投放时间和起爆延迟
    t_r1 = max(0, t_r1)
    t_r2 = max(0, t_r2)
    t_r3 = max(0, t_r3)
    
    dt_d1 = max(0.1, min(2.0, dt_d1))
    dt_d2 = max(0.1, min(2.0, dt_d2))
    dt_d3 = max(0.1, min(2.0, dt_d3))
    
    # 确保总投放时间不超过约束
    for i, (t_r, dt_d) in enumerate([(t_r1, dt_d1), (t_r2, dt_d2), (t_r3, dt_d3)]):
        if t_r + dt_d > 15.0:
            scale = 15.0 / (t_r + dt_d)
            if i == 0:
                t_r1 *= scale
                dt_d1 *= scale
            elif i == 1:
                t_r2 *= scale
                dt_d2 *= scale
            else:
                t_r3 *= scale
                dt_d3 *= scale
    
    return [dir1, spd1, t_r1, dt_d1, dir2, spd2, t_r2, dt_d2, dir3, spd3, t_r3, dt_d3]

def genetic_algorithm(population_size=100, generations=30, elite_size=10, mutation_rate=0.2):
    """遗传算法优化多无人机投放策略"""
    print("开始遗传算法优化...")
    
    # 创建初始种群
    population = []
    for _ in range(population_size):
        # 为每个无人机随机生成参数
        individual = []
        for _ in range(3):  # 三架无人机
            direction = random.uniform(0, 360)
            speed = random.uniform(70, 140)
            release_time = random.uniform(0, 4.0)
            detonation_delay = random.uniform(0.1, 2.0)
            individual.extend([direction, speed, release_time, detonation_delay])
        
        # 约束参数并添加到种群
        individual = constrain_params(individual)
        population.append(individual)
    
    # 添加基于问题2和问题3的知识的种子个体
    # FY1的种子参数从问题2得知
    population[0] = [
        5.0, 136.5, 0.3, 0.4,  # FY1参数
        30.0, 120.0, 1.0, 0.5,  # FY2参数
        350.0, 130.0, 2.0, 0.3  # FY3参数
    ]
    
    # 另一个变种种子
    population[1] = [
        6.0, 135.0, 0.5, 0.5,  # FY1参数
        25.0, 135.0, 1.5, 0.4,  # FY2参数
        355.0, 125.0, 2.5, 0.5  # FY3参数
    ]
    
    best_individual = None
    best_fitness = 0
    best_result = None
    
    # 记录每代最佳适应度
    generation_fitness = []
    
    # 进化迭代
    for generation in range(generations):
        print(f"开始第 {generation+1}/{generations} 代进化")
        
        # 评估适应度
        fitness_results = []
        for individual in population:
            if not validate_params(individual):
                fitness_results.append(0)
                continue
                
            result = calculate_multi_drone_effect(individual)
            fitness = result['effective_duration']
            fitness_results.append(fitness)
            
            # 更新全局最佳解
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
                best_result = result
                print(f"第 {generation+1} 代: 找到更好的解，有效时长: {best_fitness:.4f}")
        
        # 记录本代最佳适应度
        gen_best_idx = np.argmax(fitness_results)
        gen_best_fitness = fitness_results[gen_best_idx]
        generation_fitness.append(gen_best_fitness)
        
        print(f"第 {generation+1} 代最佳适应度: {gen_best_fitness:.4f}")
        
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
                    if i % 4 == 0:  # 方向
                        child[i] += random.uniform(-20, 20)
                    elif i % 4 == 1:  # 速度
                        child[i] += random.uniform(-10, 10)
                    else:  # 时间参数
                        child[i] += random.uniform(-0.5, 0.5)
            
            # 约束参数
            child = constrain_params(child)
            
            # 确保参数有效
            if validate_params(child):
                new_population.append(child)
        
        # 更新种群
        population = new_population
    
    # 绘制进化过程
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, generations+1), generation_fitness, 'b-', marker='o')
    plt.xlabel('代数')
    plt.ylabel('最佳适应度 (有效遮蔽时长)')
    plt.title('遗传算法进化过程')
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/ga_evolution.png", dpi=300)
    plt.close()
    
    return best_result

def particle_swarm_optimization(n_particles=50, iterations=30):
    """粒子群优化算法"""
    print("开始粒子群优化...")
    
    # 参数范围定义
    param_ranges = [
        (0, 360),    # dir1
        (70, 140),   # spd1
        (0, 4.0),    # t_r1
        (0.1, 2.0),  # dt_d1
        (0, 360),    # dir2
        (70, 140),   # spd2
        (0, 4.0),    # t_r2
        (0.1, 2.0),  # dt_d2
        (0, 360),    # dir3
        (70, 140),   # spd3
        (0, 4.0),    # t_r3
        (0.1, 2.0)   # dt_d3
    ]
    
    # 初始化粒子
    particles = []
    velocities = []
    for _ in range(n_particles):
        # 随机初始化粒子位置
        particle = []
        for min_val, max_val in param_ranges:
            particle.append(random.uniform(min_val, max_val))
        
        # 确保参数合法
        particle = constrain_params(particle)
        particles.append(particle)
        
        # 初始化粒子速度
        velocity = []
        for min_val, max_val in param_ranges:
            range_size = max_val - min_val
            velocity.append(random.uniform(-range_size * 0.1, range_size * 0.1))
        velocities.append(velocity)
    
    # 添加基于问题2和问题3的知识的种子粒子
    particles[0] = [
        5.0, 136.5, 0.3, 0.4,  # FY1参数
        30.0, 120.0, 1.0, 0.5,  # FY2参数
        350.0, 130.0, 2.0, 0.3  # FY3参数
    ]
    
    # 初始化粒子历史最佳位置和适应度
    best_positions = particles.copy()
    best_fitnesses = [0] * n_particles
    
    # 全局最佳位置和适应度
    global_best_position = None
    global_best_fitness = 0
    global_best_result = None
    
    # 记录每次迭代的最佳适应度
    iteration_fitness = []
    
    # 迭代优化
    for iteration in range(iterations):
        print(f"开始第 {iteration+1}/{iterations} 次迭代")
        
        # 评估每个粒子的适应度
        for i, particle in enumerate(particles):
            # 确保参数合法
            if not validate_params(particle):
                continue
            
            result = calculate_multi_drone_effect(particle)
            fitness = result['effective_duration']
            
            # 更新个体历史最佳
            if fitness > best_fitnesses[i]:
                best_fitnesses[i] = fitness
                best_positions[i] = particle.copy()
                
                # 更新全局最佳
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle.copy()
                    global_best_result = result
                    print(f"第 {iteration+1} 次迭代: 找到更好的解，有效时长: {global_best_fitness:.4f}")
        
        # 记录本次迭代的最佳适应度
        iteration_fitness.append(global_best_fitness)
        
        print(f"第 {iteration+1} 次迭代全局最佳适应度: {global_best_fitness:.4f}")
        
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
    
    # 绘制迭代过程
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, iterations+1), iteration_fitness, 'g-', marker='o')
    plt.xlabel('迭代次数')
    plt.ylabel('全局最佳适应度 (有效遮蔽时长)')
    plt.title('粒子群优化迭代过程')
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/pso_iteration.png", dpi=300)
    plt.close()
    
    return global_best_result

def hybrid_optimization():
    """混合优化策略：结合多种算法"""
    print("开始混合优化...")
    
    # 1. 遗传算法
    print("\n===== 执行遗传算法 =====")
    ga_result = genetic_algorithm()
    
    # 2. 粒子群优化
    print("\n===== 执行粒子群优化 =====")
    pso_result = particle_swarm_optimization()
    
    # 比较两种算法的结果
    results = [
        ("遗传算法", ga_result),
        ("粒子群优化", pso_result)
    ]
    
    best_method = max(results, key=lambda x: x[1]['effective_duration'])
    
    print("\n===== 优化结果比较 =====")
    for method, result in results:
        print(f"{method}有效遮蔽时长: {result['effective_duration']:.4f}")
    
    print(f"\n最佳方法: {best_method[0]}")
    print(f"最终有效遮蔽时长: {best_method[1]['effective_duration']:.4f}")
    
    return best_method[1]

def visualize_multi_drone_effect(result):
    """可视化多无人机投放烟幕干扰弹效果"""
    params = result['params']
    
    # 解析参数
    dir1, spd1, t_r1, dt_d1, dir2, spd2, t_r2, dt_d2, dir3, spd3, t_r3, dt_d3 = params
    
    # 导弹到达假目标的时间
    missile_total_time = missile_time_to_target(MISSILE_M1_INIT, FAKE_TARGET, MISSILE_SPEED)
    
    # 创建3D图
    fig = plt.figure(figsize=(15, 12))
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
    
    # 绘制无人机轨迹和投放点
    drone_params = [
        (DRONE_FY1_INIT, dir1, spd1, t_r1, "FY1", "green"),
        (DRONE_FY2_INIT, dir2, spd2, t_r2, "FY2", "purple"),
        (DRONE_FY3_INIT, dir3, spd3, t_r3, "FY3", "orange")
    ]
    
    release_positions = result['release_positions']
    detonation_positions = result['detonation_positions']
    detonation_times = result['detonation_times']
    
    for i, (drone_init, direction, speed, release_time, drone_name, color) in enumerate(drone_params):
        # 绘制无人机初始位置
        ax.scatter(drone_init[0], drone_init[1], drone_init[2], 
                   color=color, marker='^', s=100, label=f'{drone_name}初始位置')
        
        # 绘制无人机轨迹
        drone_times = np.linspace(0, release_time, 50)
        drone_positions = []
        for t in drone_times:
            drone_positions.append(drone_trajectory(drone_init, direction, speed, t))
        drone_positions = np.array(drone_positions)
        ax.plot(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2], 
                color=color, linestyle='-', label=f'{drone_name}轨迹')
        
        # 绘制投放点和起爆点
        ax.scatter(release_positions[i][0], release_positions[i][1], release_positions[i][2], 
                   color=color, s=80, marker='o', label=f'{drone_name}投放点')
        ax.scatter(detonation_positions[i][0], detonation_positions[i][1], detonation_positions[i][2], 
                   color=color, s=80, marker='*', label=f'{drone_name}起爆点')
    
    # 绘制有效遮蔽时间区间的烟幕云团
    colors = ['green', 'purple', 'orange']
    for i, (intervals, detonation_pos, detonation_time) in enumerate(zip(
            result['intervals'], detonation_positions, detonation_times)):
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
    
    # 添加每个无人机的参数信息
    drone_info = [
        (f"FY1: 方向={dir1:.1f}°, 速度={spd1:.1f}m/s, 投放={t_r1:.2f}s, 延迟={dt_d1:.2f}s", "green"),
        (f"FY2: 方向={dir2:.1f}°, 速度={spd2:.1f}m/s, 投放={t_r2:.2f}s, 延迟={dt_d2:.2f}s", "purple"),
        (f"FY3: 方向={dir3:.1f}°, 速度={spd3:.1f}m/s, 投放={t_r3:.2f}s, 延迟={dt_d3:.2f}s", "orange")
    ]
    
    for i, (info, color) in enumerate(drone_info):
        ax.text2D(0.05, 0.92 - i * 0.03, info, transform=ax.transAxes, fontsize=10, color=color)
    
    # 设置坐标轴范围和标签
    ax.set_xlim([-1000, 21000])
    ax.set_ylim([-3500, 3500])
    ax.set_zlim([0, 2500])
    ax.set_xlabel('X轴 (m)')
    ax.set_ylabel('Y轴 (m)')
    ax.set_zlabel('Z轴 (m)')
    ax.set_title('问题4: 多无人机协同投放烟幕干扰弹策略')
    
    # 优化图例显示
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/problem4_optimal_solution.png', dpi=300)
    plt.close()
    
    # 绘制时间线图，显示各烟幕弹的有效遮蔽区间
    plt.figure(figsize=(12, 6))
    
    colors = ['green', 'purple', 'orange']
    labels = ['FY1烟幕弹', 'FY2烟幕弹', 'FY3烟幕弹']
    
    for i, intervals in enumerate(result['intervals']):
        for start, end in intervals:
            plt.barh(i, end - start, left=start, height=0.6, color=colors[i], alpha=0.7)
            plt.text(start + (end - start) / 2, i, f"{end - start:.1f}s", 
                     ha='center', va='center', fontweight='bold')
    
    plt.yticks([0, 1, 2], labels)
    plt.xlabel('时间 (秒)')
    plt.title('三架无人机烟幕干扰弹的有效遮蔽时间区间')
    plt.grid(True, alpha=0.3)
    
    # 添加总有效遮蔽时长信息
    plt.figtext(0.5, 0.01, f"总有效遮蔽时长: {result['effective_duration']:.2f}秒", 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/problem4_shielding_timeline.png', dpi=300)
    plt.close()

def save_result_to_excel(result):
    """保存结果到Excel文件"""
    # 解析参数
    dir1, spd1, t_r1, dt_d1, dir2, spd2, t_r2, dt_d2, dir3, spd3, t_r3, dt_d3 = result['params']
    
    # 创建结果DataFrame
    drone_params = [
        ('FY1', dir1, spd1, t_r1, dt_d1),
        ('FY2', dir2, spd2, t_r2, dt_d2),
        ('FY3', dir3, spd3, t_r3, dt_d3)
    ]
    
    result_data = []
    for i, (drone, direction, speed, release_time, detonation_delay) in enumerate(drone_params):
        result_data.append({
            '无人机': drone,
            '飞行方向 (°)': round(direction, 2),
            '飞行速度 (m/s)': round(speed, 2),
            '投放时间 (s)': round(release_time, 2),
            '起爆延迟 (s)': round(detonation_delay, 2),
            '投放点X坐标 (m)': round(result['release_positions'][i][0], 2),
            '投放点Y坐标 (m)': round(result['release_positions'][i][1], 2),
            '投放点Z坐标 (m)': round(result['release_positions'][i][2], 2),
            '起爆点X坐标 (m)': round(result['detonation_positions'][i][0], 2),
            '起爆点Y坐标 (m)': round(result['detonation_positions'][i][1], 2),
            '起爆点Z坐标 (m)': round(result['detonation_positions'][i][2], 2)
        })
    
    df_results = pd.DataFrame(result_data)
    
    # 创建有效遮蔽时间区间DataFrame
    intervals_data = []
    for i, intervals in enumerate(result['intervals']):
        drone = drone_params[i][0]
        for j, (start, end) in enumerate(intervals):
            intervals_data.append({
                '无人机': drone,
                '区间序号': j+1,
                '开始时间 (s)': round(start, 2),
                '结束时间 (s)': round(end, 2),
                '时长 (s)': round(end - start, 2)
            })
    
    df_intervals = pd.DataFrame(intervals_data)
    
    # 创建总结信息
    summary_data = {
        '指标': ['总有效遮蔽时长 (s)', 'FY1有效遮蔽时长 (s)', 'FY2有效遮蔽时长 (s)', 'FY3有效遮蔽时长 (s)'],
        '值': [
            round(result['effective_duration'], 2),
            round(sum(end - start for start, end in result['intervals'][0]), 2),
            round(sum(end - start for start, end in result['intervals'][1]), 2),
            round(sum(end - start for start, end in result['intervals'][2]), 2)
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    
    # 保存到Excel文件
    with pd.ExcelWriter(f'{RESULTS_DIR}/problem4_optimal_solution.xlsx') as writer:
        df_results.to_excel(writer, sheet_name='投放参数', index=False)
        df_intervals.to_excel(writer, sheet_name='有效遮蔽时间区间', index=False)
        df_summary.to_excel(writer, sheet_name='总结', index=False)
    
    print(f"已保存结果到 {RESULTS_DIR}/problem4_optimal_solution.xlsx")

def main():
    """主函数"""
    start_time = time.time()
    print("开始优化问题4: 多无人机协同投放烟幕干扰弹策略...")
    
    # 使用混合优化策略
    result = hybrid_optimization()
    
    # 可视化结果
    print("\n开始可视化最优结果...")
    visualize_multi_drone_effect(result)
    
    # 保存结果到Excel
    save_result_to_excel(result)
    
    end_time = time.time()
    print(f"\n优化完成，总耗时: {end_time - start_time:.2f}秒")
    print(f"最终有效遮蔽时长: {result['effective_duration']:.4f}秒")
    
    # 输出最优参数
    dir1, spd1, t_r1, dt_d1, dir2, spd2, t_r2, dt_d2, dir3, spd3, t_r3, dt_d3 = result['params']
    print("\n最优烟幕干扰弹投放策略:")
    print(f"FY1: 方向={dir1:.2f}°, 速度={spd1:.2f}m/s, 投放时间={t_r1:.2f}s, 起爆延迟={dt_d1:.2f}s")
    print(f"FY2: 方向={dir2:.2f}°, 速度={spd2:.2f}m/s, 投放时间={t_r2:.2f}s, 起爆延迟={dt_d2:.2f}s")
    print(f"FY3: 方向={dir3:.2f}°, 速度={spd3:.2f}m/s, 投放时间={t_r3:.2f}s, 起爆延迟={dt_d3:.2f}s")
    
    return result

if __name__ == "__main__":
    main()