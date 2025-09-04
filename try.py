import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 系统参数
missile_speed = 300  # m/s
drone_speed = 120    # m/s
smoke_radius = 10    # m
smoke_sink_speed = 3 # m/s
smoke_duration = 20  # s
target_radius = 7    # m
target_height = 10   # m

# 初始位置
M1_initial = np.array([20000, 0, 2000])
FY1_initial = np.array([17800, 0, 1800])
fake_target = np.array([0, 0, 0])
real_target_center = np.array([0, 200, 0])

# 时间参数
task_start_time = 0
deploy_time = 1.5  # 投放时间
explosion_delay = 3.6  # 起爆延迟
explosion_time = deploy_time + explosion_delay  # 5.1s

def calculate_missile_position(t):
    """计算导弹在t时刻的位置"""
    # 导弹朝向假目标飞行
    direction = fake_target - M1_initial
    direction = direction / np.linalg.norm(direction)
    return M1_initial + missile_speed * t * direction

def calculate_drone_position(t):
    """计算无人机在t时刻的位置（受领任务后）"""
    # 无人机朝向假目标飞行
    direction = fake_target - FY1_initial
    direction = direction / np.linalg.norm(direction)
    return FY1_initial + drone_speed * t * direction

def calculate_smoke_position(t_after_deploy):
    """计算烟幕弹在投放后t时刻的位置（自由落体）"""
    drone_pos_at_deploy = calculate_drone_position(deploy_time)
    # 自由落体运动
    fall_distance = 0.5 * 9.8 * t_after_deploy**2
    smoke_pos = drone_pos_at_deploy.copy()
    smoke_pos[2] -= fall_distance
    return smoke_pos

def calculate_smoke_center_after_explosion(t_after_explosion):
    """计算起爆后烟幕中心的位置"""
    smoke_pos_at_explosion = calculate_smoke_position(explosion_delay)
    # 烟幕下沉
    smoke_center = smoke_pos_at_explosion.copy()
    smoke_center[2] -= smoke_sink_speed * t_after_explosion
    return smoke_center

def is_line_intersect_sphere(p1, p2, center, radius):
    """判断线段是否与球相交"""
    # 线段从p1到p2，球心center，半径radius
    d = p2 - p1
    f = p1 - center
    
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius**2
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return False
    
    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2*a)
    t2 = (-b + discriminant) / (2*a)
    
    # 检查交点是否在线段上
    if (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1):
        return True
    return False

def check_obstruction(t):
    """检查在时刻t，烟幕是否遮挡了导弹对真目标的视线"""
    if t < explosion_time:
        return False
    
    t_after_explosion = t - explosion_time
    if t_after_explosion > smoke_duration:
        return False
    
    # 获取当前位置
    missile_pos = calculate_missile_position(t)
    smoke_center = calculate_smoke_center_after_explosion(t_after_explosion)
    
    # 检查导弹到真目标的视线是否被烟幕遮挡
    # 真目标是一个圆柱，检查多个点
    obstruction_count = 0
    total_points = 0
    
    # 检查真目标圆柱体上的多个点
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        for h in [0, target_height/2, target_height]:
            target_point = real_target_center + np.array([
                target_radius * np.cos(angle),
                target_radius * np.sin(angle),
                h
            ])
            total_points += 1
            
            if is_line_intersect_sphere(missile_pos, target_point, smoke_center, smoke_radius):
                obstruction_count += 1
    
    # 如果大部分视线被遮挡，认为有效遮蔽
    return obstruction_count > total_points * 0.5

# 计算有效遮蔽时长
time_step = 0.01
max_time = 30
obstruction_times = []

for t in np.arange(0, max_time, time_step):
    if check_obstruction(t):
        obstruction_times.append(t)

# 计算连续遮蔽时段
if obstruction_times:
    start_time = obstruction_times[0]
    end_time = obstruction_times[-1]
    total_obstruction_time = end_time - start_time
else:
    total_obstruction_time = 0
    start_time = 0
    end_time = 0

# 输出结果
print("="*60)
print("问题1 计算结果")
print("="*60)
print(f"无人机FY1初始位置: {FY1_initial}")
print(f"无人机飞行速度: {drone_speed} m/s")
print(f"无人机飞行方向: 朝向假目标")
print(f"投放时间: {deploy_time} s")
print(f"起爆时间: {explosion_time} s")
print(f"投放点位置: {calculate_drone_position(deploy_time)}")
print(f"起爆点位置: {calculate_smoke_position(explosion_delay)}")
print("-"*60)
if total_obstruction_time > 0:
    print(f"有效遮蔽开始时间: {start_time:.2f} s")
    print(f"有效遮蔽结束时间: {end_time:.2f} s")
    print(f"有效遮蔽时长: {total_obstruction_time:.2f} s")
else:
    print("未形成有效遮蔽")

# 可视化轨迹
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制时间序列的位置
time_points = np.linspace(0, 25, 100)
missile_trajectory = np.array([calculate_missile_position(t) for t in time_points])
drone_trajectory = np.array([calculate_drone_position(t) for t in time_points if t <= deploy_time])

# 绘制轨迹
ax.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], missile_trajectory[:, 2], 
        'r-', label='导弹M1轨迹', linewidth=2)
ax.plot(drone_trajectory[:, 0], drone_trajectory[:, 1], drone_trajectory[:, 2], 
        'b-', label='无人机FY1轨迹', linewidth=2)

# 标记关键点
ax.scatter(*M1_initial, color='red', s=100, marker='^', label='M1初始位置')
ax.scatter(*FY1_initial, color='blue', s=100, marker='s', label='FY1初始位置')
ax.scatter(*fake_target, color='black', s=200, marker='x', label='假目标')
ax.scatter(*real_target_center, color='green', s=200, marker='o', label='真目标中心')
ax.scatter(*calculate_drone_position(deploy_time), color='orange', s=150, 
           marker='v', label='投放点')
ax.scatter(*calculate_smoke_position(explosion_delay), color='purple', s=150, 
           marker='*', label='起爆点')

# 绘制烟幕云团（在某个时刻）
if obstruction_times:
    t_sample = (start_time + end_time) / 2
    smoke_center_sample = calculate_smoke_center_after_explosion(t_sample - explosion_time)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = smoke_radius * np.outer(np.cos(u), np.sin(v)) + smoke_center_sample[0]
    y = smoke_radius * np.outer(np.sin(u), np.sin(v)) + smoke_center_sample[1]
    z = smoke_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + smoke_center_sample[2]
    ax.plot_surface(x, y, z, alpha=0.3, color='gray')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('烟幕干扰弹投放策略示意图')
ax.legend()
plt.show()

# 更详细的分析
print("\n" + "="*60)
print("详细分析")
print("="*60)

# 计算关键时刻的状态
for t in [0, deploy_time, explosion_time, explosion_time + 5, explosion_time + 10]:
    missile_pos = calculate_missile_position(t)
    missile_dist_to_fake = np.linalg.norm(missile_pos - fake_target)
    missile_dist_to_real = np.linalg.norm(missile_pos - real_target_center)
    
    print(f"\n时刻 t = {t:.1f} s:")
    print(f"  导弹位置: ({missile_pos[0]:.1f}, {missile_pos[1]:.1f}, {missile_pos[2]:.1f})")
    print(f"  导弹距假目标: {missile_dist_to_fake:.1f} m")
    print(f"  导弹距真目标: {missile_dist_to_real:.1f} m")
    
    if t >= explosion_time and t <= explosion_time + smoke_duration:
        smoke_center = calculate_smoke_center_after_explosion(t - explosion_time)
        print(f"  烟幕中心位置: ({smoke_center[0]:.1f}, {smoke_center[1]:.1f}, {smoke_center[2]:.1f})")
        if check_obstruction(t):
            print("  ✓ 有效遮蔽")
        else:
            print("  ✗ 未形成有效遮蔽")