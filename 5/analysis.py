import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches


plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)
        
    def draw(self, renderer):
        FancyArrowPatch.draw(self, renderer)

    

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

# 基础函数
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
    bomb_intervals = {i: [] for i in range(len(sorted_defense))}  # 每个烟幕弹的有效区间
    in_cloud = False
    start_time_record = None
    current_effective_bomb = None
    
    # 为每个烟幕弹计算并缓存起爆点和有效时间窗口
    defense_cache = []
    for i, defense in enumerate(sorted_defense):
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
                'index': i,
                'drone_idx': drone_idx,
                'detonation_pos': detonation_pos,
                'detonation_time': detonation_time,
                'end_time': detonation_time + SMOKE_EFFECTIVE_TIME,
                'release_pos': release_pos,
                'release_time': release_time,
                'direction': direction,
                'speed': speed,
                'detonation_delay': detonation_delay
            })
    
    # 遍历导弹飞行的时间步长
    for t in np.arange(start_time, end_time, time_step):
        missile_pos = missile_trajectory(missile_init_pos, FAKE_TARGET, MISSILE_SPEED, t)
        
        # 检查所有烟幕云团对该导弹的遮蔽效果
        is_effective = 0
        effective_bomb_idx = None
        
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
                if effect > is_effective:
                    is_effective = effect
                    effective_bomb_idx = defense['index']
        
        # 记录有效遮蔽区间
        if is_effective and not in_cloud:
            in_cloud = True
            start_time_record = t
            current_effective_bomb = effective_bomb_idx
        elif not is_effective and in_cloud:
            in_cloud = False
            interval = (start_time_record, t)
            effective_intervals.append(interval)
            if current_effective_bomb is not None:
                bomb_intervals[current_effective_bomb].append(interval)
            current_effective_bomb = None
        elif is_effective and in_cloud and effective_bomb_idx != current_effective_bomb:
            # 如果有效烟幕弹发生变化，记录前一个的结束并开始新的
            interval = (start_time_record, t)
            effective_intervals.append(interval)
            if current_effective_bomb is not None:
                bomb_intervals[current_effective_bomb].append(interval)
            start_time_record = t
            current_effective_bomb = effective_bomb_idx
    
    # 检查如果结束时仍在云团中
    if in_cloud and start_time_record is not None:
        interval = (start_time_record, end_time)
        effective_intervals.append(interval)
        if current_effective_bomb is not None:
            bomb_intervals[current_effective_bomb].append(interval)
    
    # 将烟幕弹区间添加到防御计划中
    for i, defense in enumerate(sorted_defense):
        defense['intervals'] = bomb_intervals.get(i, [])
    
    return effective_intervals, sorted_defense

def calculate_total_effective_time(time_intervals):
    """计算多个时间区间的总有效时长，处理重叠问题"""
    if not time_intervals:
        return 0, []  # 修复：返回总时间和空列表
    
    sorted_intervals = sorted(time_intervals)
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        previous = merged[-1]
        if current[0] <= previous[1]:  # 重叠区间
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:  # 不重叠区间
            merged.append(current)
    
    total_time = sum(end - start for start, end in merged)
    return total_time, merged  # 修复：返回总时间和合并后的区间

def load_best_strategy():
    """加载最佳防御策略"""
    results_dir = "combined_defense_results"
    
    # 尝试从best_strategy_details.xlsx加载
    try:
        if os.path.exists(f"{results_dir}/best_strategy_details.xlsx"):
            df = pd.read_excel(f"{results_dir}/best_strategy_details.xlsx")
            
            # 构建防御策略
            drone_solutions = {}
            
            for _, row in df.iterrows():
                drone_idx = int(row['无人机'][2]) - 1  # 'FY1' -> 0
                
                if drone_idx not in drone_solutions:
                    drone_solutions[drone_idx] = {
                        'direction': row['飞行方向(°)'],
                        'speed': row['飞行速度(m/s)'],
                        'bombs': []
                    }
                
                missile_idx = int(row['目标导弹'][1]) - 1  # 'M1' -> 0
                
                drone_solutions[drone_idx]['bombs'].append({
                    'release_time': row['投放时间(s)'],
                    'detonation_delay': row['起爆延迟(s)'],
                    'target_missile': missile_idx
                })
            
            return drone_solutions
    except Exception as e:
        print(f"从best_strategy_details.xlsx加载失败: {e}")
    
    # 尝试从top100_defense_strategies.xlsx加载
    try:
        if os.path.exists(f"{results_dir}/top100_defense_strategies.xlsx"):
            df = pd.read_excel(f"{results_dir}/top100_defense_strategies.xlsx")
            if not df.empty:
                # 获取排名第一的防御策略
                best_strategy = df.iloc[0]
                
                # 构建防御策略
                drone_solutions = {}
                
                for drone_idx in range(5):
                    drone_solutions[drone_idx] = {
                        'direction': best_strategy[f'FY{drone_idx+1}_direction'],
                        'speed': best_strategy[f'FY{drone_idx+1}_speed'],
                        'bombs': []
                    }
                    
                    # 查找该无人机的所有烟幕弹
                    bomb_idx = 1
                    while f'FY{drone_idx+1}_bomb{bomb_idx}_missile' in best_strategy:
                        missile_name = best_strategy[f'FY{drone_idx+1}_bomb{bomb_idx}_missile']
                        missile_idx = int(missile_name[1]) - 1  # 'M1' -> 0
                        
                        drone_solutions[drone_idx]['bombs'].append({
                            'release_time': best_strategy[f'FY{drone_idx+1}_bomb{bomb_idx}_release_time'],
                            'detonation_delay': best_strategy[f'FY{drone_idx+1}_bomb{bomb_idx}_detonation_delay'],
                            'target_missile': missile_idx
                        })
                        
                        bomb_idx += 1
                
                return drone_solutions
    except Exception as e:
        print(f"从top100_defense_strategies.xlsx加载失败: {e}")
    
    # 如果上面都失败了，返回FY1的固定参数和一个空策略
    print("未找到已保存的最佳策略，使用FY1的固定参数和空策略")
    return {
        0: {
            'direction': 4.87985955499065,
            'speed': 140,
            'bombs': [
                {'release_time': 0, 'detonation_delay': 0, 'target_missile': 0},
                {'release_time': 1, 'detonation_delay': 0, 'target_missile': 0},
                {'release_time': 7.00425122294099, 'detonation_delay': 0.203560474384496, 'target_missile': 0}
            ]
        },
        1: {'direction': 0, 'speed': 100, 'bombs': []},
        2: {'direction': 0, 'speed': 100, 'bombs': []},
        3: {'direction': 0, 'speed': 100, 'bombs': []},
        4: {'direction': 0, 'speed': 100, 'bombs': []}
    }

def generate_defense_plans(drone_solutions):
    """根据每个无人机的解决方案生成导弹防御计划"""
    defense_plans = {0: [], 1: [], 2: []}
    
    for drone_idx, solution in drone_solutions.items():
        direction = solution['direction']
        speed = solution['speed']
        
        for bomb in solution['bombs']:
            target_missile = bomb['target_missile']
            release_time = bomb['release_time']
            detonation_delay = bomb['detonation_delay']
            
            defense_plans[target_missile].append({
                'drone_idx': drone_idx,
                'direction': direction,
                'speed': speed,
                'release_time': release_time,
                'detonation_delay': detonation_delay
            })
    
    return defense_plans

def analyze_best_strategy():
    """分析最佳防御策略，生成可视化和详细报告"""
    print("开始分析最佳防御策略...")
    
    # 创建结果目录
    results_dir = "strategy_visualization"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 加载最佳防御策略
    drone_solutions = load_best_strategy()
    
    # 生成导弹防御计划
    defense_plans = generate_defense_plans(drone_solutions)
    
    # 计算每个导弹的防御效果和烟幕弹的详细区间
    missile_results = {}
    for missile_idx in range(3):
        intervals, defense_with_intervals = calculate_missile_defense_effect(
            missile_idx, defense_plans[missile_idx]
        )
        total_time, merged_intervals = calculate_total_effective_time(intervals)
        
        missile_results[missile_idx] = {
            'total_time': total_time,
            'intervals': merged_intervals,
            'defense_with_intervals': defense_with_intervals
        }
    
    # 计算总有效遮蔽时长
    total_shielding_time = sum(result['total_time'] for result in missile_results.values())
    
    # 保存烟幕弹的详细信息
    bomb_details = []
    
    for missile_idx in range(3):
        for defense in missile_results[missile_idx]['defense_with_intervals']:
            drone_idx = defense['drone_idx']
            intervals = defense['intervals']
            
            for i, (start, end) in enumerate(intervals):
                bomb_details.append({
                    '无人机': f'FY{drone_idx+1}',
                    '目标导弹': f'M{missile_idx+1}',
                    '飞行方向(°)': defense['direction'],
                    '飞行速度(m/s)': defense['speed'],
                    '投放时间(s)': defense['release_time'],
                    '起爆延迟(s)': defense['detonation_delay'],
                    '遮蔽开始时间(s)': start,
                    '遮蔽结束时间(s)': end,
                    '遮蔽持续时间(s)': end - start
                })
    
    # 保存详细信息到Excel
    df_details = pd.DataFrame(bomb_details)
    df_details.to_excel(f"{results_dir}/bomb_shielding_details.xlsx", index=False)
    print(f"已保存烟幕弹遮蔽详细信息到 {results_dir}/bomb_shielding_details.xlsx")
    
    # 创建时间线可视化
    plt.figure(figsize=(14, 10))
    
    # 为每个导弹创建时间线
    missile_labels = ['M1', 'M2', 'M3']
    missile_colors = ['#FF5733', '#33A8FF', '#33FF57']
    
    for missile_idx in range(3):
        plt.subplot(3, 1, missile_idx+1)
        
        # 绘制所有烟幕弹的遮蔽区间
        for defense in missile_results[missile_idx]['defense_with_intervals']:
            drone_idx = defense['drone_idx']
            drone_label = f'FY{drone_idx+1}'
            
            # 不同无人机使用不同颜色和透明度
            alpha = 0.7
            drone_color = plt.cm.tab10(drone_idx)
            
            for start, end in defense['intervals']:
                plt.barh(drone_label, end-start, left=start, height=0.7, 
                         color=drone_color, alpha=alpha, edgecolor='black', linewidth=1)
                
                # 添加持续时间标签
                if end-start > 1.0:  # 只标注持续时间超过1秒的区间
                    plt.text(start + (end-start)/2, drone_label, f"{end-start:.1f}s", 
                             ha='center', va='center', fontsize=9, fontweight='bold')
        
        # 绘制合并后的总有效区间
        for i, (start, end) in enumerate(missile_results[missile_idx]['intervals']):
            plt.barh('总遮蔽', end-start, left=start, height=0.7,
                    color=missile_colors[missile_idx], alpha=0.9, edgecolor='black', linewidth=1)
            
            # 添加持续时间标签
            plt.text(start + (end-start)/2, '总遮蔽', f"{end-start:.1f}s", 
                     ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 设置坐标轴和标题
        plt.title(f'导弹{missile_labels[missile_idx]}的遮蔽时间线 (总遮蔽时长: {missile_results[missile_idx]["total_time"]:.2f}s)')
        plt.xlabel('时间 (秒)')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 调整y轴标签顺序
        drone_labels = [f'FY{i+1}' for i in range(5) if any(d['drone_idx'] == i for d in missile_results[missile_idx]['defense_with_intervals'])]
        if drone_labels:
            ticks = list(range(len(drone_labels)+1))  # 生成数值刻度 [0, 1, 2, 3, 4]

            plt.yticks(ticks=ticks,labels=drone_labels + ['总遮蔽'])
        else:
            plt.yticks(['总遮蔽'])
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/shielding_timeline.png", dpi=300)
    plt.savefig(f"{results_dir}/shielding_timeline.pdf")
    plt.close()
    
    # 创建3D轨迹可视化
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制假目标和真目标
    ax.scatter([0], [0], [0], color='red', s=100, marker='o', label='假目标')
    
    # 绘制真目标（圆柱体）
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, REAL_TARGET_HEIGHT, 10)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_cylinder = REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * np.cos(theta_grid)
    y_cylinder = REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * np.sin(theta_grid)
    ax.plot_surface(x_cylinder, y_cylinder, z_grid + REAL_TARGET_CENTER[2], alpha=0.5, color='blue')
    
    # 绘制导弹轨迹
    missile_colors = ['#FF5733', '#33A8FF', '#33FF57']
    for missile_idx in range(3):
        missile_init_pos = MISSILE_POSITIONS[missile_idx]
        distance = np.linalg.norm(FAKE_TARGET - missile_init_pos)
        missile_total_time = distance / MISSILE_SPEED
        
        # 计算轨迹点
        times = np.linspace(0, missile_total_time, 100)
        trajectory_points = []
        for t in times:
            pos = missile_trajectory(missile_init_pos, FAKE_TARGET, MISSILE_SPEED, t)
            trajectory_points.append(pos)
        
        trajectory_points = np.array(trajectory_points)
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
                color=missile_colors[missile_idx], linewidth=2, label=f'导弹M{missile_idx+1}轨迹')
        
        # 绘制导弹初始位置
        ax.scatter(missile_init_pos[0], missile_init_pos[1], missile_init_pos[2], 
                   color=missile_colors[missile_idx], s=80, marker='^', label=f'导弹M{missile_idx+1}初始位置')
        
        # 标记被遮蔽的区间
        for interval in missile_results[missile_idx]['intervals']:
            start_time, end_time = interval
            shielded_times = np.linspace(start_time, end_time, 20)
            shielded_points = []
            for t in shielded_times:
                pos = missile_trajectory(missile_init_pos, FAKE_TARGET, MISSILE_SPEED, t)
                shielded_points.append(pos)
            
            shielded_points = np.array(shielded_points)
            ax.plot(shielded_points[:, 0], shielded_points[:, 1], shielded_points[:, 2], 
                    color=missile_colors[missile_idx], linewidth=5, alpha=0.7)
    
    # 绘制无人机轨迹和烟幕弹
    drone_colors = ['#9933FF', '#FF33A8', '#A8FF33', '#33FFA8', '#FFA833']
    for drone_idx, solution in drone_solutions.items():
        drone_init_pos = DRONE_POSITIONS[drone_idx]
        direction = solution['direction']
        speed = solution['speed']
        
        # 找出该无人机的最晚投放时间
        max_time = 0
        if solution['bombs']:
            max_time = max(b['release_time'] + b['detonation_delay'] + 2 for b in solution['bombs'])
        
        # 计算轨迹点
        times = np.linspace(0, max(max_time, 10), 50)  # 至少绘制10秒的轨迹
        trajectory_points = []
        for t in times:
            pos = drone_trajectory(drone_init_pos, direction, speed, t)
            trajectory_points.append(pos)
        
        trajectory_points = np.array(trajectory_points)
        if len(trajectory_points) > 0:
            ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
                    color=drone_colors[drone_idx], linestyle='--', label=f'无人机FY{drone_idx+1}轨迹')
        
        # 绘制无人机初始位置
        ax.scatter(drone_init_pos[0], drone_init_pos[1], drone_init_pos[2], 
                   color=drone_colors[drone_idx], s=100, marker='o', label=f'无人机FY{drone_idx+1}初始位置')
        
        # 绘制无人机飞行方向箭头
        if max_time > 0:
            arrow_length = 1000  # 箭头长度
            direction_rad = np.radians(direction)
            end_x = drone_init_pos[0] + arrow_length * np.cos(direction_rad)
            end_y = drone_init_pos[1] + arrow_length * np.sin(direction_rad)
            arrow = Arrow3D([drone_init_pos[0], end_x], 
                            [drone_init_pos[1], end_y], 
                            [drone_init_pos[2], drone_init_pos[2]], 
                            mutation_scale=20, lw=3, arrowstyle="-|>", color=drone_colors[drone_idx])
            ax.add_artist(arrow)
        
        # 绘制烟幕弹投放和爆炸点
        for bomb_idx, bomb in enumerate(solution['bombs']):
            target_missile = bomb['target_missile']
            release_time = bomb['release_time']
            detonation_delay = bomb['detonation_delay']
            detonation_time = release_time + detonation_delay
            
            # 计算投放点
            release_pos = drone_trajectory(drone_init_pos, direction, speed, release_time)
            ax.scatter(release_pos[0], release_pos[1], release_pos[2], 
                       color=missile_colors[target_missile], s=80, marker='D', 
                       label=f'FY{drone_idx+1}投放点 #{bomb_idx+1}' if bomb_idx == 0 else "")
            
            # 计算爆炸点
            detonation_pos = smoke_trajectory(release_pos, release_time, direction, speed, detonation_time)
            if detonation_pos is not None:
                ax.scatter(detonation_pos[0], detonation_pos[1], detonation_pos[2], 
                           color=missile_colors[target_missile], s=100, marker='*', 
                           label=f'FY{drone_idx+1}爆炸点 #{bomb_idx+1}' if bomb_idx == 0 else "")
                
                # 绘制烟幕效果区域（简化为球体）
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = detonation_pos[0] + SMOKE_EFFECTIVE_RADIUS * np.cos(u) * np.sin(v)
                y = detonation_pos[1] + SMOKE_EFFECTIVE_RADIUS * np.sin(u) * np.sin(v)
                z = detonation_pos[2] + SMOKE_EFFECTIVE_RADIUS * np.cos(v)
                ax.plot_wireframe(x, y, z, color=missile_colors[target_missile], alpha=0.2)
    
    # 设置坐标轴范围和标签
    ax.set_xlim([-1000, 21000])
    ax.set_ylim([-3500, 3500])
    ax.set_zlim([0, 2500])
    ax.set_xlabel('X轴 (m)')
    ax.set_ylabel('Y轴 (m)')
    ax.set_zlabel('Z轴 (m)')
    ax.set_title(f'最佳防御策略3D可视化 (总有效遮蔽时长: {total_shielding_time:.2f}秒)')
    
    # 优化图例显示
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/defense_strategy_3d.png", dpi=300)
    plt.savefig(f"{results_dir}/defense_strategy_3d.pdf")
    plt.close()
    
    # 创建摘要报告
    summary = {
        '总有效遮蔽时长(s)': total_shielding_time,
        'M1有效遮蔽时长(s)': missile_results[0]['total_time'],
        'M2有效遮蔽时长(s)': missile_results[1]['total_time'],
        'M3有效遮蔽时长(s)': missile_results[2]['total_time'],
        '总烟幕弹数量': sum(len(solution['bombs']) for solution in drone_solutions.values())
    }
    
    pd.DataFrame([summary]).to_excel(f"{results_dir}/strategy_summary.xlsx", index=False)
    
    # 按无人机分组的报告
    drone_data = []
    for drone_idx, solution in drone_solutions.items():
        bomb_count = len(solution['bombs'])
        if bomb_count > 0:
            drone_data.append({
                '无人机': f'FY{drone_idx+1}',
                '飞行方向(°)': solution['direction'],
                '飞行速度(m/s)': solution['speed'],
                '烟幕弹数量': bomb_count,
                '目标导弹': ', '.join(f'M{b["target_missile"]+1}' for b in solution['bombs'])
            })
    
    pd.DataFrame(drone_data).to_excel(f"{results_dir}/drone_summary.xlsx", index=False)
    
    # 输出摘要信息
    print("\n防御策略分析完成!")
    print(f"总有效遮蔽时长: {total_shielding_time:.2f}秒")
    print(f"导弹M1有效遮蔽时长: {missile_results[0]['total_time']:.2f}秒")
    print(f"导弹M2有效遮蔽时长: {missile_results[1]['total_time']:.2f}秒")
    print(f"导弹M3有效遮蔽时长: {missile_results[2]['total_time']:.2f}秒")
    print(f"总烟幕弹数量: {sum(len(solution['bombs']) for solution in drone_solutions.values())}")
    
    print(f"\n可视化和详细报告已保存到 {results_dir} 目录")
    
    return total_shielding_time, missile_results

# 创建防御过程动画
def create_defense_animation(missile_results, drone_solutions, frame_count=200):
    """创建防御过程的动画"""
    print("开始创建防御过程动画...")
    
    # 创建结果目录
    results_dir = "strategy_visualization"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 计算导弹总飞行时间
    missile_total_times = []
    for missile_idx in range(3):
        missile_init_pos = MISSILE_POSITIONS[missile_idx]
        distance = np.linalg.norm(FAKE_TARGET - missile_init_pos)
        missile_total_times.append(distance / MISSILE_SPEED)
    
    max_missile_time = max(missile_total_times)
    
    # 计算最大需要展示的时间
    max_defense_time = 0
    for drone_idx, solution in drone_solutions.items():
        for bomb in solution['bombs']:
            release_time = bomb['release_time']
            detonation_delay = bomb['detonation_delay']
            max_defense_time = max(max_defense_time, release_time + detonation_delay + SMOKE_EFFECTIVE_TIME)
    
    max_time = min(max(max_defense_time, max_missile_time), max_missile_time)
    
    max_time = 35

    # 设置动画
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 颜色设置
    missile_colors = ['#FF5733', '#33A8FF', '#33FF57']
    drone_colors = ['#9933FF', '#FF33A8', '#A8FF33', '#33FFA8', '#FFA833']
    
    # 初始化所有对象的轨迹数据
    missile_trajectories = []
    for missile_idx in range(3):
        missile_init_pos = MISSILE_POSITIONS[missile_idx]
        times = np.linspace(0, missile_total_times[missile_idx], 1000)
        points = []
        for t in times:
            pos = missile_trajectory(missile_init_pos, FAKE_TARGET, MISSILE_SPEED, t)
            points.append(pos)
        missile_trajectories.append(np.array(points))
    
    drone_trajectories = []
    for drone_idx, solution in drone_solutions.items():
        drone_init_pos = DRONE_POSITIONS[drone_idx]
        direction = solution['direction']
        speed = solution['speed']
        
        max_drone_time = 0
        if solution['bombs']:
            max_drone_time = max(b['release_time'] for b in solution['bombs']) + 10
        
        times = np.linspace(0, max_drone_time, 1000)
        points = []
        for t in times:
            pos = drone_trajectory(drone_init_pos, direction, speed, t)
            points.append(pos)
        drone_trajectories.append(np.array(points))
    
    # 计算每个烟幕弹的轨迹和云团
    bomb_data = []
    for drone_idx, solution in drone_solutions.items():
        drone_init_pos = DRONE_POSITIONS[drone_idx]
        direction = solution['direction']
        speed = solution['speed']
        
        for bomb in solution['bombs']:
            target_missile = bomb['target_missile']
            release_time = bomb['release_time']
            detonation_delay = bomb['detonation_delay']
            detonation_time = release_time + detonation_delay
            
            # 计算投放点
            release_pos = drone_trajectory(drone_init_pos, direction, speed, release_time)
            
            # 计算烟幕弹轨迹
            bomb_times = np.linspace(release_time, detonation_time, 100)
            bomb_points = []
            for t in bomb_times:
                pos = smoke_trajectory(release_pos, release_time, direction, speed, t)
                if pos is not None:
                    bomb_points.append(pos)
            
            # 计算爆炸点
            detonation_pos = smoke_trajectory(release_pos, release_time, direction, speed, detonation_time)
            
            # 计算云团轨迹
            cloud_end_time = detonation_time + SMOKE_EFFECTIVE_TIME
            cloud_times = np.linspace(detonation_time, cloud_end_time, 100)
            cloud_points = []
            for t in cloud_times:
                pos = smoke_cloud_trajectory(detonation_pos, detonation_time, t)
                if pos is not None:
                    cloud_points.append(pos)
            
            bomb_data.append({
                'drone_idx': drone_idx,
                'target_missile': target_missile,
                'release_time': release_time,
                'release_pos': release_pos,
                'detonation_time': detonation_time,
                'detonation_pos': detonation_pos,
                'cloud_end_time': cloud_end_time,
                'bomb_trajectory': np.array(bomb_points) if bomb_points else None,
                'cloud_trajectory': np.array(cloud_points) if cloud_points else None
            })
    
    # 动画初始化函数
    def init():
        ax.clear()
        
        # 绘制假目标和真目标
        ax.scatter([0], [0], [0], color='red', s=100, marker='o')
        
        # 绘制真目标（圆柱体）
        theta = np.linspace(0, 2*np.pi, 20)
        z = np.linspace(0, REAL_TARGET_HEIGHT, 5)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_cylinder = REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * np.cos(theta_grid)
        y_cylinder = REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * np.sin(theta_grid)
        ax.plot_surface(x_cylinder, y_cylinder, z_grid + REAL_TARGET_CENTER[2], alpha=0.5, color='blue')
        
        # 绘制导弹初始位置
        for missile_idx in range(3):
            missile_init_pos = MISSILE_POSITIONS[missile_idx]
            ax.scatter(missile_init_pos[0], missile_init_pos[1], missile_init_pos[2], 
                       color=missile_colors[missile_idx], s=80, marker='^')
        
        # 绘制无人机初始位置
        for drone_idx in range(5):
            drone_init_pos = DRONE_POSITIONS[drone_idx]
            ax.scatter(drone_init_pos[0], drone_init_pos[1], drone_init_pos[2], 
                       color=drone_colors[drone_idx], s=100, marker='o')
        
        # 设置坐标轴范围和标签
        ax.set_xlim([-1000, 21000])
        ax.set_ylim([-3500, 3500])
        ax.set_zlim([0, 2500])
        ax.set_xlabel('X轴 (m)')
        ax.set_ylabel('Y轴 (m)')
        ax.set_zlabel('Z轴 (m)')
        
        return []
    
    # 动画更新函数
    def update(frame):
        ax.clear()
        
        # 计算当前时间
        current_time = frame * max_time / frame_count
        
        # 绘制假目标和真目标
        ax.scatter([0], [0], [0], color='red', s=100, marker='o')
        
        # 绘制真目标（圆柱体）
        theta = np.linspace(0, 2*np.pi, 20)
        z = np.linspace(0, REAL_TARGET_HEIGHT, 5)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_cylinder = REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * np.cos(theta_grid)
        y_cylinder = REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * np.sin(theta_grid)
        ax.plot_surface(x_cylinder, y_cylinder, z_grid + REAL_TARGET_CENTER[2], alpha=0.5, color='blue')
        
        # 绘制导弹轨迹和当前位置
        for missile_idx in range(3):
            missile_init_pos = MISSILE_POSITIONS[missile_idx]
            
            # 绘制轨迹
            if current_time <= missile_total_times[missile_idx]:
                # 找出当前时间点之前的轨迹
                traj_idx = int(current_time / missile_total_times[missile_idx] * 1000)
                traj_idx = min(traj_idx, len(missile_trajectories[missile_idx]) - 1)
                traj = missile_trajectories[missile_idx][:traj_idx+1]
                
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                        color=missile_colors[missile_idx], linewidth=2)
                
                # 绘制当前位置
                current_pos = missile_trajectory(missile_init_pos, FAKE_TARGET, MISSILE_SPEED, current_time)
                ax.scatter(current_pos[0], current_pos[1], current_pos[2], 
                           color=missile_colors[missile_idx], s=100, marker='>')
                
                # 检查是否处于有效遮蔽状态
                is_shielded = False
                for interval in missile_results[missile_idx]['intervals']:
                    if interval[0] <= current_time <= interval[1]:
                        is_shielded = True
                        break
                
                if is_shielded:
                    # 如果被遮蔽，在导弹周围绘制特效
                    u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]
                    x = current_pos[0] + 100 * np.cos(u) * np.sin(v)
                    y = current_pos[1] + 100 * np.sin(u) * np.sin(v)
                    z = current_pos[2] + 100 * np.cos(v)
                    ax.plot_wireframe(x, y, z, color=missile_colors[missile_idx], alpha=0.3)
        
        # 绘制无人机轨迹和当前位置
        for drone_idx, solution in drone_solutions.items():
            drone_init_pos = DRONE_POSITIONS[drone_idx]
            direction = solution['direction']
            speed = solution['speed']
            
            # 确定该无人机是否在当前时间活跃
            if not solution['bombs'] or current_time > max(b['release_time'] for b in solution['bombs']) + 10:
                continue
            
            # 绘制轨迹
            traj_idx = min(int(current_time * 100), len(drone_trajectories[drone_idx]) - 1)
            traj = drone_trajectories[drone_idx][:traj_idx+1]
            
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                    color=drone_colors[drone_idx], linestyle='--', linewidth=2)
            
            # 绘制当前位置
            current_pos = drone_trajectory(drone_init_pos, direction, speed, current_time)
            ax.scatter(current_pos[0], current_pos[1], current_pos[2], 
                       color=drone_colors[drone_idx], s=100, marker='o')
        
        # 绘制烟幕弹和云团
        for bomb in bomb_data:
            # 烟幕弹投放
            if current_time >= bomb['release_time']:
                # 如果已经投放，绘制投放点
                ax.scatter(bomb['release_pos'][0], bomb['release_pos'][1], bomb['release_pos'][2], 
                           color=missile_colors[bomb['target_missile']], s=80, marker='D')
                
                # 如果烟幕弹在飞行中
                if bomb['release_time'] <= current_time < bomb['detonation_time']:
                    # 绘制烟幕弹轨迹
                    traj_idx = int((current_time - bomb['release_time']) / 
                                   (bomb['detonation_time'] - bomb['release_time']) * 100)
                    traj_idx = min(traj_idx, len(bomb['bomb_trajectory']) - 1 if bomb['bomb_trajectory'] is not None else 0)
                    
                    if bomb['bomb_trajectory'] is not None:
                        traj = bomb['bomb_trajectory'][:traj_idx+1]
                        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                                color=missile_colors[bomb['target_missile']], linestyle='-.', linewidth=2)
                    
                    # 绘制当前位置
                    current_bomb_pos = smoke_trajectory(
                        bomb['release_pos'], bomb['release_time'], 
                        drone_solutions[bomb['drone_idx']]['direction'], 
                        drone_solutions[bomb['drone_idx']]['speed'], 
                        current_time
                    )
                    if current_bomb_pos is not None:
                        ax.scatter(current_bomb_pos[0], current_bomb_pos[1], current_bomb_pos[2], 
                                   color=missile_colors[bomb['target_missile']], s=80, marker='s')
                
                # 如果云团已经形成
                if bomb['detonation_time'] <= current_time <= bomb['cloud_end_time']:
                    # 绘制爆炸点
                    ax.scatter(bomb['detonation_pos'][0], bomb['detonation_pos'][1], bomb['detonation_pos'][2], 
                               color=missile_colors[bomb['target_missile']], s=100, marker='*')
                    
                    # 绘制云团
                    current_cloud_pos = smoke_cloud_trajectory(
                        bomb['detonation_pos'], bomb['detonation_time'], current_time
                    )
                    
                    if current_cloud_pos is not None:
                        u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:8j]
                        x = current_cloud_pos[0] + SMOKE_EFFECTIVE_RADIUS * np.cos(u) * np.sin(v)
                        y = current_cloud_pos[1] + SMOKE_EFFECTIVE_RADIUS * np.sin(u) * np.sin(v)
                        z = current_cloud_pos[2] + SMOKE_EFFECTIVE_RADIUS * np.cos(v)
                        ax.plot_wireframe(x, y, z, color=missile_colors[bomb['target_missile']], alpha=0.3)
        
        # 设置坐标轴范围和标签
        ax.set_xlim([-1000, 21000])
        ax.set_ylim([-3500, 3500])
        ax.set_zlim([0, 2500])
        ax.set_xlabel('X轴 (m)')
        ax.set_ylabel('Y轴 (m)')
        ax.set_zlabel('Z轴 (m)')
        ax.set_title(f'防御过程动画 (当前时间: {current_time:.2f}秒)')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color='red', marker='o', linestyle='None', 
                      markersize=10, label='假目标'),
            plt.Line2D([0], [0], color='blue', marker='s', linestyle='None', 
                      markersize=10, label='真目标')
        ]
        
        for missile_idx in range(3):
            legend_elements.append(
                plt.Line2D([0], [0], color=missile_colors[missile_idx], marker='>', linestyle='-', 
                          markersize=10, label=f'导弹M{missile_idx+1}')
            )
        
        for drone_idx in range(5):
            if drone_idx in drone_solutions and drone_solutions[drone_idx]['bombs']:
                legend_elements.append(
                    plt.Line2D([0], [0], color=drone_colors[drone_idx], marker='o', linestyle='--', 
                              markersize=10, label=f'无人机FY{drone_idx+1}')
                )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        return []
    
    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=frame_count, init_func=init, blit=True)
    
    # 保存动画
    ani.save(f"{results_dir}/defense_animation.mp4", writer='ffmpeg', fps=15, dpi=200)
    print(f"防御过程动画已保存到 {results_dir}/defense_animation.mp4")
    
    plt.close()

if __name__ == "__main__":
    # 分析最佳防御策略
    total_time, missile_results = analyze_best_strategy()
    
    # 加载最佳防御策略
    drone_solutions = load_best_strategy()
    
    # 创建防御过程动画
    create_defense_animation(missile_results, drone_solutions)
    
    print("全部分析和可视化完成！")