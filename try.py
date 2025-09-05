import numpy as np
import time

# --- 1. 定义常量 ---
# 导弹M1
V_M1 = 300
P0_M1 = np.array([20000, 0, 2000])
TARGET_FALSE = np.array([0, 0, 0])

# 无人机初始位置
UAV_INITIAL_POSITIONS = {
    'FY1': np.array([17800, 0, 1800]),
    'FY2': np.array([12000, 1400, 1400]),
    'FY3': np.array([6000, -3000, 700])
}
V_UAV_MIN, V_UAV_MAX = 70, 140

# 真目标
TARGET_TRUE_CENTER_BASE = np.array([0, 200, 0])
TARGET_TRUE_RADIUS = 7
TARGET_TRUE_HEIGHT = 10

# 烟幕
SMOKE_CLOUD_RADIUS = 10
SMOKE_CLOUD_SINK_V = 3
SMOKE_DURATION = 20

# 物理常量
G = 9.8

# --- 2. 运动学模型 (修改以支持多无人机) ---
direction_m1 = (TARGET_FALSE - P0_M1) / np.linalg.norm(TARGET_FALSE - P0_M1)
def missile_position(t):
    return P0_M1 + V_M1 * t * direction_m1

def uav_position(uav_id, v_f, theta, t):
    p0_uav = UAV_INITIAL_POSITIONS[uav_id]
    vx = v_f * np.cos(theta)
    vy = v_f * np.sin(theta)
    return p0_uav + t * np.array([vx, vy, 0])

def grenade_position(uav_id, v_f, theta, t_d, t):
    if t < t_d: return uav_position(uav_id, v_f, theta, t)
    p0_grenade = uav_position(uav_id, v_f, theta, t_d)
    v0_grenade = np.array([v_f * np.cos(theta), v_f * np.sin(theta), 0])
    dt = t - t_d
    pos = p0_grenade + v0_grenade * dt + 0.5 * np.array([0, 0, -G]) * dt**2
    return pos

def cloud_center_position(uav_id, v_f, theta, t_d, t_b, t):
    if t < t_b: return None
    p_detonation = grenade_position(uav_id, v_f, theta, t_d, t_b)
    dt = t - t_b
    return p_detonation + dt * np.array([0, 0, -SMOKE_CLOUD_SINK_V])

# --- 3. 几何遮蔽判断 ---
target_key_points = []
for h in [0, TARGET_TRUE_HEIGHT]:
    center = TARGET_TRUE_CENTER_BASE + np.array([0, 0, h])
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        target_key_points.append(center + np.array([TARGET_TRUE_RADIUS * np.cos(angle), TARGET_TRUE_RADIUS * np.sin(angle), 0]))
target_key_points.append(TARGET_TRUE_CENTER_BASE + np.array([0, 0, TARGET_TRUE_HEIGHT/2]))

def is_line_segment_intercepted_by_sphere(p1, p2, sphere_center, sphere_radius):
    v = p2 - p1
    a = np.dot(v, v)
    if a == 0: return False
    b = 2 * np.dot(v, p1 - sphere_center)
    c = np.dot(p1 - sphere_center, p1 - sphere_center) - sphere_radius**2
    discriminant = b**2 - 4*a*c
    if discriminant < 0: return False
    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)
    if (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1*t2 < 0): return True
    return False

def is_shielded(m_pos, c_pos):
    if c_pos is None or c_pos[2] < -SMOKE_CLOUD_RADIUS: return False
    for point in target_key_points:
        if not is_line_segment_intercepted_by_sphere(m_pos, point, c_pos, SMOKE_CLOUD_RADIUS):
            return False
    return True

# --- 4. 适应度函数 ---
def calculate_fitness_q4(params):
    v1, th1, td1, tb1, v2, th2, td2, tb2, v3, th3, td3, tb3 = params

    if not (tb1 > td1 and tb2 > td2 and tb3 > td3):
        return 0.0

    total_shielding_time = 0
    time_step = 0.1
    start_time = min(tb1, tb2, tb3)
    end_time = max(tb1, tb2, tb3) + SMOKE_DURATION

    for t in np.arange(start_time, end_time, time_step):
        m_pos = missile_position(t)
        if m_pos[0] <= TARGET_TRUE_CENTER_BASE[0]: break

        c1_pos = cloud_center_position('FY1', v1, th1, td1, tb1, t)
        c2_pos = cloud_center_position('FY2', v2, th2, td2, tb2, t)
        c3_pos = cloud_center_position('FY3', v3, th3, td3, tb3, t)
        
        if is_shielded(m_pos, c1_pos) or \
           is_shielded(m_pos, c2_pos) or \
           is_shielded(m_pos, c3_pos):
            total_shielding_time += time_step
    
    return total_shielding_time

# --- 辅助函数: 计算单弹遮蔽时间与时间段 (已修改) ---
def calculate_single_shield_details(uav_id, uav_params):
    v_f, theta, t_d, t_b = uav_params
    if not (t_b > t_d): return 0.0, -1.0, -1.0
    
    single_shield_time = 0.0
    start_shield = -1.0
    end_shield = -1.0
    time_step = 0.1
    
    for t in np.arange(t_b, t_b + SMOKE_DURATION, time_step):
        m_pos = missile_position(t)
        if m_pos[0] <= TARGET_TRUE_CENTER_BASE[0]: break
        c_pos = cloud_center_position(uav_id, v_f, theta, t_d, t_b, t)
        if is_shielded(m_pos, c_pos):
            if start_shield < 0: # 首次探测到遮蔽
                start_shield = t
            end_shield = t # 持续更新最后遮蔽时间
            single_shield_time += time_step
            
    return single_shield_time, start_shield, end_shield

# --- 5. PSO 主算法 ---
def pso_optimizer(n_particles, n_iterations, initial_solution=None):
    bounds = [
        (V_UAV_MIN, V_UAV_MAX), (0, 2 * np.pi), (1, 30), (2, 40),
        (V_UAV_MIN, V_UAV_MAX), (0, 2 * np.pi), (1, 30), (2, 40),
        (V_UAV_MIN, V_UAV_MAX), (0, 2 * np.pi), (1, 30), (2, 40),
    ]
    n_dim = len(bounds)
    
    particles_pos = np.random.rand(n_particles, n_dim)
    for i in range(n_dim):
        particles_pos[:, i] = particles_pos[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    if initial_solution is not None:
        particles_pos[0] = initial_solution

    particles_vel = np.random.randn(n_particles, n_dim) * 0.1
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([calculate_fitness_q4(p) for p in pbest_pos])
    
    gbest_idx = np.argmax(pbest_fitness)
    gbest_pos = np.copy(pbest_pos[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]
    
    w, c1, c2 = 0.7, 1.5, 1.5
    print("\n--- 开始优化(问题4: 3机协同) ---")
    start_time = time.time()
    
    for it in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(2)
            cognitive_vel = c1 * r1 * (pbest_pos[i] - particles_pos[i])
            social_vel = c2 * r2 * (gbest_pos - particles_pos[i])
            particles_vel[i] = w * particles_vel[i] + cognitive_vel + social_vel
            particles_pos[i] += particles_vel[i]
            for j in range(n_dim):
                particles_pos[i, j] = np.clip(particles_pos[i, j], bounds[j][0], bounds[j][1])

            current_fitness = calculate_fitness_q4(particles_pos[i])
            
            if current_fitness > pbest_fitness[i]:
                pbest_fitness[i] = current_fitness
                pbest_pos[i] = np.copy(particles_pos[i])
                if current_fitness > gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_pos = np.copy(particles_pos[i])
        
        # <-- 修改点：恢复过程打印 -->
        if (it + 1) % 5 == 0:
            print(f"迭代次数: {it + 1}/{n_iterations}, 当前最优遮蔽时间: {gbest_fitness:.3f} s")
            
    end_time = time.time()
    print(f"优化完成! 总耗时: {end_time - start_time:.2f} s")
    return gbest_pos, gbest_fitness

# --- 6. 执行与结果 ---
if __name__ == '__main__':
    NUM_PARTICLES = 100
    NUM_ITERATIONS = 200
    
    print("\n--- 正在评估您指定的初始策略 ---")
    
    # 策略 1 (for FY1)
    v1_initial = 70.0
    theta1_initial = np.deg2rad(3.97)
    td1_initial = 1.5
    tb1_initial = 1.6
    
    # 策略 3 (for FY3)
    v3_initial = 98.15
    theta3_initial = np.deg2rad(93.30)
    td3_initial = 26.757
    tb3_initial = 31.445
    
    # 为FY2设置一个合理的默认初始值 (朝向假目标)
    v2_initial = 120.0
    td2_initial = 1.5
    tb2_initial = 5.1
    direction_vector_fy2 = TARGET_FALSE[0:2] - UAV_INITIAL_POSITIONS['FY2'][0:2]
    theta2_initial = np.arctan2(direction_vector_fy2[1], direction_vector_fy2[0])
    
    initial_solution = np.array([
        v1_initial, theta1_initial, td1_initial, tb1_initial,
        v2_initial, theta2_initial, td2_initial, tb2_initial,
        v3_initial, theta3_initial, td3_initial, tb3_initial
    ])
    
    initial_fitness = calculate_fitness_q4(initial_solution)
    print(f"初始策略的综合遮蔽时间为: {initial_fitness:.3f} s")
    
    best_solution, max_time = pso_optimizer(
        NUM_PARTICLES, 
        NUM_ITERATIONS,
        initial_solution=initial_solution
    )
    
    # 记录最终最优参数
    v1, th1, td1, tb1, v2, th2, td2, tb2, v3, th3, td3, tb3 = best_solution

    # 计算各无人机的投放点、起爆点、遮蔽时间等
    drop_point1 = uav_position('FY1', v1, th1, td1)
    detonation_point1 = grenade_position('FY1', v1, th1, td1, tb1)
    shield_time1, start1, end1 = calculate_single_shield_details('FY1', (v1, th1, td1, tb1))

    drop_point2 = uav_position('FY2', v2, th2, td2)
    detonation_point2 = grenade_position('FY2', v2, th2, td2, tb2)
    shield_time2, start2, end2 = calculate_single_shield_details('FY2', (v2, th2, td2, tb2))

    drop_point3 = uav_position('FY3', v3, th3, td3)
    detonation_point3 = grenade_position('FY3', v3, th3, td3, tb3)
    shield_time3, start3, end3 = calculate_single_shield_details('FY3', (v3, th3, td3, tb3))

    # --- 打印FY1的策略 ---
    print(f"无人机 FY1:")
    print(f"  - 飞行速度: {v1:.2f} m/s")
    print(f"  - 飞行方向: {np.rad2deg(th1):.2f} 度")
    print(f"  - 投放时间: {td1:.3f} s")
    print(f"  - 投放点坐标: ({drop_point1[0]:.2f}, {drop_point1[1]:.2f}, {drop_point1[2]:.2f})")
    print(f"  - 起爆时间: {tb1:.3f} s")
    print(f"  - 起爆点坐标: ({detonation_point1[0]:.2f}, {detonation_point1[1]:.2f}, {detonation_point1[2]:.2f})")
    print(f"  - 单弹有效遮蔽: {shield_time1:.3f} s")
    if start1 > 0:
        print(f"  - 遮蔽时间段: {start1:.3f}s - {end1:.3f}s")
    else:
        print(f"  - 遮蔽时间段: 无")
    print("-"*65)

    # --- 打印FY2的策略 ---
    print(f"无人机 FY2:")
    print(f"  - 飞行速度: {v2:.2f} m/s")
    print(f"  - 飞行方向: {np.rad2deg(th2):.2f} 度")
    print(f"  - 投放时间: {td2:.3f} s")
    print(f"  - 投放点坐标: ({drop_point2[0]:.2f}, {drop_point2[1]:.2f}, {drop_point2[2]:.2f})")
    print(f"  - 起爆时间: {tb2:.3f} s")
    print(f"  - 起爆点坐标: ({detonation_point2[0]:.2f}, {detonation_point2[1]:.2f}, {detonation_point2[2]:.2f})")
    print(f"  - 单弹有效遮蔽: {shield_time2:.3f} s")
    if start2 > 0:
        print(f"  - 遮蔽时间段: {start2:.3f}s - {end2:.3f}s")
    else:
        print(f"  - 遮蔽时间段: 无")
    print("-"*65)
    
    # --- 打印FY3的策略 ---
    print(f"无人机 FY3:")
    print(f"  - 飞行速度: {v3:.2f} m/s")
    print(f"  - 飞行方向: {np.rad2deg(th3):.2f} 度")
    print(f"  - 投放时间: {td3:.3f} s")
    print(f"  - 投放点坐标: ({drop_point3[0]:.2f}, {drop_point3[1]:.2f}, {drop_point3[2]:.2f})")
    print(f"  - 起爆时间: {tb3:.3f} s")
    print(f"  - 起爆点坐标: ({detonation_point3[0]:.2f}, {detonation_point3[1]:.2f}, {detonation_point3[2]:.2f})")
    print(f"  - 单弹有效遮蔽: {shield_time3:.3f} s")
    if start3 > 0:
        print(f"  - 遮蔽时间段: {start3:.3f}s - {end3:.3f}s")
    else:
        print(f"  - 遮蔽时间段: 无")
    
    print("="*65)
    print(f"最大有效综合遮蔽时间: {max_time:.3f} s")
    print("="*65)

# 无人机 FY1:
#   - 飞行速度: 70.00 m/s
#   - 飞行方向: 3.97 度
#   - 投放时间: 1.500 s
#   - 投放点坐标: (17904.75, 7.27, 1800.00)
#   - 起爆时间: 1.600 s
#   - 起爆点坐标: (17911.73, 7.75, 1799.95)
#   - 单弹有效遮蔽: 4.600 s
#   - 遮蔽时间段: 2.300s - 6.800s
# -----------------------------------------------------------------
#   无人机 FY2:
#
#   - 飞行速度: 81.13 m/s
#   - 飞行方向: 360.00 度
#   - 投放时间: 1.000 s
#   - 投放点坐标: (12081.13, 1400.00, 1400.00)
#   - 起爆时间: 16.447 s
#   - 起爆点坐标: (13334.35, 1400.00, 230.78)

# -----------------------------------------------------------------
# 无人机 FY3:
#   - 飞行速度: 98.15 m/s
#   - 飞行方向: 93.30 度
#   - 投放时间: 26.757 s
#   - 投放点坐标: (5848.83, -378.16, 700.00)
#   - 起爆时间: 31.445 s
#   - 起爆点坐标: (5822.34, 81.21, 592.31)
#   - 单弹有效遮蔽: 2.800 s
#   - 遮蔽时间段: 32.845s - 35.545s