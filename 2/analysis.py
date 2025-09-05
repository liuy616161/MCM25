import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建结果目录
results_dir = "parameter_analysis_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 从Excel文件读取数据
def load_data_from_excel(file_path):
    """从Excel文件加载数据并过滤指定参数范围"""
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        print(f"原始数据: {len(df)}行 x {len(df.columns)}列")
        
        # 定义目标参数范围
        directions = list(range(0, 10, 1)) + list(range(175, 186, 1)) + list(range(350, 361, 1))
        speeds = np.linspace(70, 140, 21)
        release_times = np.linspace(0, 1.5, 11)
        detonation_delays = np.linspace(0, 1, 11)
        
        # 过滤数据 - 使用近似匹配
        # 对于direction和speed，我们可以直接使用isin
        df_filtered = df[df['direction'].isin(directions) & df['speed'].isin(speeds)]
        
        # 对于连续变量release_time和detonation_delay，使用近似匹配
        # 创建匹配函数
        def closest_match(value, target_list):
            return min(target_list, key=lambda x: abs(x - value))
        
        # 应用匹配函数
        df_filtered['release_time'] = df_filtered['release_time'].apply(lambda x: closest_match(x, release_times))
        df_filtered['detonation_delay'] = df_filtered['detonation_delay'].apply(lambda x: closest_match(x, detonation_delays))
        
        print(f"过滤后数据: {len(df_filtered)}行")
        
        # 如果过滤后数据太少，则使用原始数据
        if len(df_filtered) < 50:
            print("过滤后数据太少，使用原始数据进行分析")
            return df
        
        return df_filtered
    
    except Exception as e:
        print(f"加载Excel文件时出错: {e}")
        return None

# 参数空间可视化
def visualize_parameter_space():
    """可视化参数搜索空间"""
    directions = list(range(0, 10, 1)) + list(range(175, 186, 1)) + list(range(350, 361, 1))
    speeds = np.linspace(70, 140, 21)
    release_times = np.linspace(0, 1.5, 11)
    detonation_delays = np.linspace(0, 1, 11)
    
    # 创建参数空间可视化
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 方向参数分布
    plt.subplot(2, 2, 1)
    plt.hist(directions, bins=len(directions), color='skyblue', edgecolor='black')
    plt.title('方向参数空间分布')
    plt.xlabel('方向 (度)')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    # 2. 速度参数分布
    plt.subplot(2, 2, 2)
    plt.hist(speeds, bins=len(speeds), color='salmon', edgecolor='black')
    plt.title('速度参数空间分布')
    plt.xlabel('速度 (m/s)')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    # 3. 投放时间参数分布
    plt.subplot(2, 2, 3)
    plt.hist(release_times, bins=15, color='lightgreen', edgecolor='black')
    plt.title('投放时间参数空间分布')
    plt.xlabel('投放时间 (秒)')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    # 4. 起爆延迟参数分布
    plt.subplot(2, 2, 4)
    plt.hist(detonation_delays, bins=15, color='plum', edgecolor='black')
    plt.title('起爆延迟参数空间分布')
    plt.xlabel('起爆延迟 (秒)')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/parameter_space.png", dpi=300)
    plt.close()
    
    # 计算理论组合数
    total_combinations = len(directions) * len(speeds) * len(release_times) * len(detonation_delays)
    
    # 创建参数空间摘要文件
    with open(f"{results_dir}/parameter_space_summary.txt", 'w', encoding='utf-8') as f:
        f.write("参数空间摘要\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"方向 (direction): {len(directions)}个值\n")
        f.write(f"  范围: {min(directions)}-{max(directions)}度\n")
        f.write(f"  值: {directions}\n\n")
        
        f.write(f"速度 (speed): {len(speeds)}个值\n")
        f.write(f"  范围: {min(speeds)}-{max(speeds)}m/s\n")
        f.write(f"  值: {speeds}\n\n")
        
        f.write(f"投放时间 (release_time): {len(release_times)}个值\n")
        f.write(f"  范围: {min(release_times):.2f}-{max(release_times):.2f}秒\n\n")
        
        f.write(f"起爆延迟 (detonation_delay): {len(detonation_delays)}个值\n")
        f.write(f"  范围: {min(detonation_delays):.2f}-{max(detonation_delays):.2f}秒\n\n")
        
        f.write(f"理论组合总数: {total_combinations}个\n")

# 数据基本分析函数
def analyze_data(df):
    """对数据进行基本分析"""
    print("="*50)
    print("数据基本统计信息:")
    print(df.describe())
    
    print("\n="*50)
    print("各参数的唯一值数量:")
    for col in ['direction', 'speed', 'release_time', 'detonation_delay']:
        print(f"{col}: {df[col].nunique()} 个唯一值")
    
    print("\n="*50)
    print("有效遮蔽时长统计:")
    print(f"最大值: {df['effective_duration'].max():.2f} 秒")
    print(f"最小值: {df['effective_duration'].min():.2f} 秒")
    print(f"平均值: {df['effective_duration'].mean():.2f} 秒")
    print(f"中位数: {df['effective_duration'].median():.2f} 秒")
    
    # 找出最优参数组合
    best_idx = df['effective_duration'].idxmax()
    print("\n="*50)
    print("最优参数组合:")
    for col in ['direction', 'speed', 'release_time', 'detonation_delay', 'effective_duration']:
        print(f"{col}: {df.loc[best_idx, col]}")
    
    # 输出结果到文件
    with open(f"{results_dir}/basic_stats.txt", 'w', encoding='utf-8') as f:
        f.write("数据基本统计信息:\n")
        f.write(df.describe().to_string())
        
        f.write("\n\n各参数的唯一值数量:\n")
        for col in ['direction', 'speed', 'release_time', 'detonation_delay']:
            f.write(f"{col}: {df[col].nunique()} 个唯一值\n")
        
        f.write("\n有效遮蔽时长统计:\n")
        f.write(f"最大值: {df['effective_duration'].max():.2f} 秒\n")
        f.write(f"最小值: {df['effective_duration'].min():.2f} 秒\n")
        f.write(f"平均值: {df['effective_duration'].mean():.2f} 秒\n")
        f.write(f"中位数: {df['effective_duration'].median():.2f} 秒\n")
        
        f.write("\n最优参数组合:\n")
        for col in ['direction', 'speed', 'release_time', 'detonation_delay', 'effective_duration']:
            f.write(f"{col}: {df.loc[best_idx, col]}\n")

# 单变量分析函数 - 分组优化版
def analyze_single_variables(df):
    """分析单个变量对有效遮蔽时长的影响，针对特定参数组进行优化"""
    plt.figure(figsize=(20, 15))
    
    # 1. 方向对有效时长的影响 - 分组分析
    plt.subplot(2, 2, 1)
    # 将方向分为三组: 0-9, 175-185, 350-360
    df['direction_group'] = pd.cut(
        df['direction'], 
        bins=[-1, 10, 174, 186, 349, 361],
        labels=['0-9°', '10-174°', '175-185°', '186-349°', '350-360°']
    )
    
    direction_effect = df.groupby('direction')['effective_duration'].agg(['mean', 'max', 'min']).reset_index()
    
    # 分组绘制
    for group in ['0-9°', '175-185°', '350-360°']:
        group_data = df[df['direction_group'] == group]
        if not group_data.empty:
            direction_values = sorted(group_data['direction'].unique())
            max_durations = []
            for d in direction_values:
                max_durations.append(group_data[group_data['direction'] == d]['effective_duration'].max())
            
            plt.plot(direction_values, max_durations, marker='o', linestyle='-', label=f'方向组 {group}')
    
    plt.xlabel('方向 (度)')
    plt.ylabel('最大有效遮蔽时长 (秒)')
    plt.title('方向对遮蔽时长的影响')
    plt.grid(True)
    plt.legend()
    
    # 2. 速度对有效时长的影响
    plt.subplot(2, 2, 2)
    speed_effect = df.groupby('speed')['effective_duration'].agg(['mean', 'max', 'min']).reset_index()
    plt.plot(speed_effect['speed'], speed_effect['mean'], 'b-', marker='o', label='平均值')
    plt.plot(speed_effect['speed'], speed_effect['max'], 'g-', marker='s', label='最大值')
    plt.plot(speed_effect['speed'], speed_effect['min'], 'r-', marker='^', label='最小值')
    plt.fill_between(speed_effect['speed'], 
                     speed_effect['min'], 
                     speed_effect['max'], 
                     alpha=0.2, color='green')
    plt.xlabel('速度 (m/s)')
    plt.ylabel('有效遮蔽时长 (秒)')
    plt.title('速度对遮蔽时长的影响')
    plt.grid(True)
    plt.legend()
    
    # 3. 投放时间对有效时长的影响
    plt.subplot(2, 2, 3)
    # 使用分箱
    df['release_time_bin'] = pd.cut(df['release_time'], bins=10)
    release_time_effect = df.groupby('release_time_bin')['effective_duration'].agg(['mean', 'max', 'min']).reset_index()
    bin_centers = [(interval.left + interval.right)/2 for interval in release_time_effect['release_time_bin']]
    plt.plot(bin_centers, release_time_effect['mean'], 'b-', marker='o', label='平均值')
    plt.plot(bin_centers, release_time_effect['max'], 'g-', marker='s', label='最大值')
    plt.plot(bin_centers, release_time_effect['min'], 'r-', marker='^', label='最小值')
    plt.fill_between(bin_centers, 
                     release_time_effect['min'], 
                     release_time_effect['max'], 
                     alpha=0.2, color='green')
    plt.xlabel('投放时间 (秒)')
    plt.ylabel('有效遮蔽时长 (秒)')
    plt.title('投放时间对遮蔽时长的影响')
    plt.grid(True)
    plt.legend()
    
    # 4. 起爆延迟对有效时长的影响
    plt.subplot(2, 2, 4)
    # 使用分箱
    df['detonation_delay_bin'] = pd.cut(df['detonation_delay'], bins=10)
    detonation_delay_effect = df.groupby('detonation_delay_bin')['effective_duration'].agg(['mean', 'max', 'min']).reset_index()
    bin_centers = [(interval.left + interval.right)/2 for interval in detonation_delay_effect['detonation_delay_bin']]
    plt.plot(bin_centers, detonation_delay_effect['mean'], 'b-', marker='o', label='平均值')
    plt.plot(bin_centers, detonation_delay_effect['max'], 'g-', marker='s', label='最大值')
    plt.plot(bin_centers, detonation_delay_effect['min'], 'r-', marker='^', label='最小值')
    plt.fill_between(bin_centers, 
                     release_time_effect['min'], 
                     release_time_effect['max'], 
                     alpha=0.2, color='green')
    plt.xlabel('起爆延迟 (秒)')
    plt.ylabel('有效遮蔽时长 (秒)')
    plt.title('起爆延迟对遮蔽时长的影响')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/single_variable_effects.png", dpi=300)
    plt.close()

    # 保存各参数对应的最佳值
    best_values = {
        '方向': direction_effect.loc[direction_effect['max'].idxmax(), 'direction'],
        '速度': speed_effect.loc[speed_effect['max'].idxmax(), 'speed'],
        '投放时间': bin_centers[release_time_effect['max'].idxmax()],
        '起爆延迟': bin_centers[detonation_delay_effect['max'].idxmax()]
    }
    
    with open(f"{results_dir}/best_single_values.txt", 'w', encoding='utf-8') as f:
        f.write("各参数的最佳单独值:\n")
        for param, value in best_values.items():
            f.write(f"{param}: {value}\n")

# 高级双变量交互分析
def analyze_advanced_interactions(df):
    """高级双变量交互分析，针对特定参数组优化"""
    # 参数组合分析
    
    # 1. 方向与速度的交互影响
    plt.figure(figsize=(12, 8))
    
    # 将方向分为三组
    direction_groups = {
        '0-9°': (0, 10),
        '175-185°': (175, 186),
        '350-360°': (350, 361)
    }
    
    # 为每个方向组绘制速度影响曲线
    for group_name, (min_dir, max_dir) in direction_groups.items():
        group_data = df[(df['direction'] >= min_dir) & (df['direction'] < max_dir)]
        if not group_data.empty:
            speed_effect = group_data.groupby('speed')['effective_duration'].max().reset_index()
            plt.plot(speed_effect['speed'], speed_effect['effective_duration'], 
                    marker='o', linestyle='-', label=f'方向组 {group_name}')
    
    plt.xlabel('速度 (m/s)')
    plt.ylabel('最大有效遮蔽时长 (秒)')
    plt.title('不同方向组下速度对遮蔽时长的影响')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{results_dir}/direction_speed_interaction.png", dpi=300)
    plt.close()
    
    # 2. 方向、速度与遮蔽时长的三维关系 - 仅针对特定方向组
    for group_name, (min_dir, max_dir) in direction_groups.items():
        group_data = df[(df['direction'] >= min_dir) & (df['direction'] < max_dir)]
        if len(group_data) > 10:  # 确保有足够数据点
            plt.figure(figsize=(10, 8))
            ax = plt.axes(projection='3d')
            
            scatter = ax.scatter3D(group_data['direction'], group_data['speed'], 
                                 group_data['effective_duration'],
                                 c=group_data['effective_duration'], cmap='viridis',
                                 s=50, alpha=0.7)
            
            plt.colorbar(scatter, label='有效遮蔽时长 (秒)')
            ax.set_xlabel('方向 (度)')
            ax.set_ylabel('速度 (m/s)')
            ax.set_zlabel('有效遮蔽时长 (秒)')
            ax.set_title(f'方向组 {group_name} 中方向与速度的三维关系')
            plt.savefig(f"{results_dir}/3d_{group_name}_direction_speed.png", dpi=300)
            plt.close()
    
    # 3. 投放时间与起爆延迟的交互热力图
    plt.figure(figsize=(12, 10))
    
    # 创建透视表
    pivot = df.pivot_table(
        index='release_time_bin',
        columns='detonation_delay_bin',
        values='effective_duration',
        aggfunc='max'
    )
    
    # 绘制热力图
    sns.heatmap(pivot, cmap='viridis', annot=False, fmt='.2f', cbar=True)
    plt.title('投放时间与起爆延迟对遮蔽时长的交互影响')
    plt.savefig(f"{results_dir}/release_detonation_heatmap.png", dpi=300)
    plt.close()
    
    # 4. 最优参数组合的详细分析
    best_idx = df['effective_duration'].idxmax()
    best_params = df.loc[best_idx]
    
    # 找出方向组
    for group_name, (min_dir, max_dir) in direction_groups.items():
        if min_dir <= best_params['direction'] < max_dir:
            best_direction_group = group_name
            break
    else:
        best_direction_group = "其他"
    
    # 在最优方向组和速度下，分析投放时间和起爆延迟
    optimal_group = df[
        (df['direction'] >= direction_groups[best_direction_group][0]) & 
        (df['direction'] < direction_groups[best_direction_group][1]) &
        (df['speed'] == best_params['speed'])
    ]
    
    if len(optimal_group) > 10:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(optimal_group['release_time'], optimal_group['detonation_delay'],
                            c=optimal_group['effective_duration'], cmap='viridis',
                            s=80, alpha=0.7)
        
        # 标记最优点
        plt.scatter([best_params['release_time']], [best_params['detonation_delay']], 
                   color='red', s=150, marker='*', edgecolors='black', label='最优组合')
        
        plt.colorbar(scatter, label='有效遮蔽时长 (秒)')
        plt.xlabel('投放时间 (秒)')
        plt.ylabel('起爆延迟 (秒)')
        plt.title(f'方向组 {best_direction_group}, 速度 {best_params["speed"]}m/s 下的最优投放组合')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{results_dir}/optimal_release_detonation.png", dpi=300)
        plt.close()

# 参数重要性分析函数
def analyze_parameter_importance(df):
    """使用随机森林分析参数重要性"""
    # 准备数据
    X = df[['direction', 'speed', 'release_time', 'detonation_delay']]
    y = df['effective_duration']
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练随机森林模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # 特征重要性
    importances = model.feature_importances_
    feature_names = ['方向', '速度', '投放时间', '起爆延迟']
    
    # 绘制特征重要性
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    plt.title('参数重要性分析')
    bars = plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices])
    plt.xlim([-1, X.shape[1]])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/parameter_importance.png", dpi=300)
    plt.close()
    
    # 部分依赖图
    fig, ax = plt.subplots(figsize=(12, 10))
    features = [0, 1, 2, 3]  # 对应方向、速度、投放时间、起爆延迟
    PartialDependenceDisplay.from_estimator(model, X, features, ax=ax, feature_names=feature_names)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/partial_dependence.png", dpi=300)
    plt.close()
    
    # 保存参数重要性数据
    importance_df = pd.DataFrame({
        'Parameter': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    importance_df.to_excel(f"{results_dir}/parameter_importance.xlsx", index=False)
    
    return model, X, y, feature_names

# 优化建议生成函数
def generate_optimization_recommendations(df, model):
    """基于分析结果生成优化建议"""
    # 获取最优参数组合
    best_idx = df['effective_duration'].idxmax()
    best_params = df.loc[best_idx]
    
    # 计算方向的可能优化
    directions = list(range(0, 10, 1)) + list(range(175, 186, 1)) + list(range(350, 361, 1))
    optimal_directions = []
    
    for direction in directions:
        direction_data = df[df['direction'] == direction]
        if not direction_data.empty:
            max_duration = direction_data['effective_duration'].max()
            if max_duration >= 0.95 * best_params['effective_duration']:  # 达到最优值的95%以上
                optimal_directions.append((direction, max_duration))
    
    # 计算速度的可能优化
    speeds = np.linspace(70, 140, 21)
    optimal_speeds = []
    
    for speed in speeds:
        speed_data = df[df['speed'] == speed]
        if not speed_data.empty:
            max_duration = speed_data['effective_duration'].max()
            if max_duration >= 0.95 * best_params['effective_duration']:  # 达到最优值的95%以上
                optimal_speeds.append((speed, max_duration))
    
    # 分析投放时间和起爆延迟
    # 获取最佳区间
    df['total_time'] = df['release_time'] + df['detonation_delay']
    total_time_effect = df.groupby('total_time')['effective_duration'].max().reset_index()
    best_total_times = total_time_effect.sort_values('effective_duration', ascending=False).head(3)
    
    # 生成建议报告
    with open(f"{results_dir}/optimization_recommendations.txt", 'w', encoding='utf-8') as f:
        f.write("烟幕干扰弹参数优化建议\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. 最优参数组合:\n")
        f.write(f"   方向: {best_params['direction']:.2f}°\n")
        f.write(f"   速度: {best_params['speed']:.2f} m/s\n")
        f.write(f"   投放时间: {best_params['release_time']:.2f} 秒\n")
        f.write(f"   起爆延迟: {best_params['detonation_delay']:.2f} 秒\n")
        f.write(f"   有效遮蔽时长: {best_params['effective_duration']:.2f} 秒\n\n")
        
        f.write("2. 推荐方向区间:\n")
        for direction, duration in sorted(optimal_directions):
            percent = duration / best_params['effective_duration'] * 100
            f.write(f"   {direction}° (达到最优值的 {percent:.1f}%, 遮蔽时长: {duration:.2f}秒)\n")
        
        f.write("\n3. 推荐速度区间:\n")
        for speed, duration in sorted(optimal_speeds):
            percent = duration / best_params['effective_duration'] * 100
            f.write(f"   {speed} m/s (达到最优值的 {percent:.1f}%, 遮蔽时长: {duration:.2f}秒)\n")
        
        f.write("\n4. 投放时间与起爆延迟组合建议:\n")
        f.write("   最优组合应满足以下关系:\n")
        for _, row in best_total_times.iterrows():
            f.write(f"   投放时间 + 起爆延迟 ≈ {row['total_time']:.2f}秒 (可达遮蔽时长: {row['effective_duration']:.2f}秒)\n")
        
        f.write("\n5. 综合优化建议:\n")
        f.write(f"   a) 方向: 优先选择 {best_params['direction']}° 附近，或者属于以下三个组的方向: 0-9°, 175-185°, 350-360°\n")
        f.write(f"   b) 速度: 建议使用 {best_params['speed']} m/s")
        if best_params['speed'] < 140:
            f.write("，如果条件允许，可以适当提高速度\n")
        else:
            f.write("\n")
        f.write(f"   c) 投放时间和起爆延迟: 总时间应接近 {best_params['release_time'] + best_params['detonation_delay']:.2f}秒，")
        f.write(f"其中投放时间约 {best_params['release_time']:.2f}秒，起爆延迟约 {best_params['detonation_delay']:.2f}秒\n")
        
        # 特殊情况建议
        f.write("\n6. 特殊情况建议:\n")
        f.write("   a) 如果无法实现精确的方向控制，优先保证速度和投放时机的准确性\n")
        f.write("   b) 如果环境因素限制速度，可以通过调整投放时间和起爆延迟来补偿\n")
        f.write("   c) 在实际操作中，推荐进行小范围的实验验证，以适应具体环境条件\n")

# 结果综合分析函数
def analyze_results_summary(df, model, X, y, feature_names):
    """综合分析结果并生成报告"""
    # 最优参数组合
    best_idx = df['effective_duration'].idxmax()
    best_params = df.loc[best_idx, ['direction', 'speed', 'release_time', 'detonation_delay', 'effective_duration']]
    
    # 计算参数重要性
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Parameter': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # 相关性分析
    corr = df[['direction', 'speed', 'release_time', 'detonation_delay', 'effective_duration']].corr()
    
    # 生成结果摘要图
    plt.figure(figsize=(15, 12))
    
    # 1. 最优参数组合
    plt.subplot(2, 2, 1)
    params = ['方向', '速度', '投放时间', '起爆延迟']
    values = best_params[['direction', 'speed', 'release_time', 'detonation_delay']].values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = plt.bar(params, values, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title(f'最优参数组合 (遮蔽时长: {best_params["effective_duration"]:.2f}秒)')
    plt.grid(True)
    
    # 2. 参数重要性
    plt.subplot(2, 2, 2)
    bars = plt.bar(importance_df['Parameter'], importance_df['Importance'], color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('参数重要性')
    plt.grid(True)
    
    # 3. 相关性热图
    plt.subplot(2, 2, 3)
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('参数相关性分析')
    
    # 4. 参数分布与遮蔽时长关系
    plt.subplot(2, 2, 4)
    
    # 创建散点图，用颜色表示遮蔽时长
    scatter = plt.scatter(df['release_time'], df['detonation_delay'], 
                        c=df['effective_duration'], cmap='viridis', 
                        s=df['speed']/2, alpha=0.7)
    
    plt.colorbar(scatter, label='有效遮蔽时长 (秒)')
    plt.xlabel('投放时间 (秒)')
    plt.ylabel('起爆延迟 (秒)')
    plt.title('参数组合与遮蔽时长关系')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/results_summary.png", dpi=300)
    plt.close()
    
    # 生成文本摘要
    with open(f"{results_dir}/analysis_summary.txt", 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("烟幕干扰弹参数优化分析摘要\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. 最优参数组合:\n")
        f.write(f"   方向: {best_params['direction']:.2f}°\n")
        f.write(f"   速度: {best_params['speed']:.2f} m/s\n")
        f.write(f"   投放时间: {best_params['release_time']:.2f} 秒\n")
        f.write(f"   起爆延迟: {best_params['detonation_delay']:.2f} 秒\n")
        f.write(f"   有效遮蔽时长: {best_params['effective_duration']:.2f} 秒\n\n")
        
        f.write("2. 参数重要性排序:\n")
        for i, row in importance_df.iterrows():
            f.write(f"   {row['Parameter']}: {row['Importance']:.4f}\n")
        f.write("\n")
        
        f.write("3. 参数相关性:\n")
        f.write("   与有效遮蔽时长的相关系数:\n")
        for param in ['direction', 'speed', 'release_time', 'detonation_delay']:
            f.write(f"   {param}: {corr.loc[param, 'effective_duration']:.4f}\n")
        f.write("\n")
        
        f.write("4. 参数交互影响摘要:\n")
        f.write("   - 方向与速度: 在特定方向下，较高的速度往往能获得更好的遮蔽效果\n")
        f.write("   - 投放时间与起爆延迟: 这两个参数需要精确配合，以确保烟幕云团在导弹和目标之间形成有效遮蔽\n")
        f.write("   - 速度与投放时间: 速度影响无人机到达最佳投放位置的时间，与投放时间紧密相关\n")
        f.write("   - 方向与起爆延迟: 不同方向下可能需要不同的起爆延迟以获得最佳效果\n")

# 主函数
def main():
    # 显示参数空间
    visualize_parameter_space()
    
    # 提示用户输入Excel文件路径
    file_path = "./all_solutions.xlsx"
    
    # 加载数据
    df = load_data_from_excel(file_path)
    
    if df is None or df.empty:
        print("无法加载数据，请检查文件路径或内容。")
        return
    
    # 基本分析
    analyze_data(df)
    
    # 单变量分析
    analyze_single_variables(df)
    
    # 高级交互分析
    analyze_advanced_interactions(df)
    
    # 参数重要性分析
    model, X, y, feature_names = analyze_parameter_importance(df)
    
    # 生成优化建议
    generate_optimization_recommendations(df, model)
    
    # 结果综合分析
    analyze_results_summary(df, model, X, y, feature_names)
    
    print(f"\n分析完成，结果已保存至 {results_dir} 目录")

if __name__ == "__main__":
    main()