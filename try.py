import numpy as np
import time

# --- 1. 定义常量 ---
# 导弹M1
V_M1 = 300
P0_M1 = np.array([20000, 0, 2000])
TARGET_FALSE = np.array([0, 0, 0])

# 无人机初始位置 (关键修改：定义一个字典来存储多架无人机信息)
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

# 传入无人机编号 uav_id 来获取其初始位置
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

# --- 3. 几何遮蔽判断 (无需修改) ---
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

# --- 4. 适应度函数 (修改以处理12个决策变量) ---
def calculate_fitness_q4(params):
    # 解析12个参数
    v1, th1, td1, tb1, \
    v2, th2, td2, tb2, \
    v3, th3, td3, tb3 = params

    # 检查基本约束
    if not (tb1 > td1 and tb2 > td2 and tb3 > td3):
        return 0.0

    total_shielding_time = 0
    time_step = 0.1 # 保持较粗的步长以在合理时间内完成优化
    start_time = min(tb1, tb2, tb3)
    end_time = max(tb1, tb2, tb3) + SMOKE_DURATION

    for t in np.arange(start_time, end_time, time_step):
        m_pos = missile_position(t)
        if m_pos[0] <= TARGET_TRUE_CENTER_BASE[0]: break

        # 计算3个独立的烟幕云位置
        c1_pos = cloud_center_position('FY1', v1, th1, td1, tb1, t)
        c2_pos = cloud_center_position('FY2', v2, th2, td2, tb2, t)
        c3_pos = cloud_center_position('FY3', v3, th3, td3, tb3, t)
        
        # 判断联合遮蔽效果
        if is_shielded(m_pos, c1_pos) or \
           is_shielded(m_pos, c2_pos) or \
           is_shielded(m_pos, c3_pos):
            total_shielding_time += time_step
    
    return total_shielding_time

# --- 5. PSO 主算法 ---
def pso_optimizer(n_particles, n_iterations):
    # 12维决策变量的边界
    bounds = [
        # FY1's params
        (V_UAV_MIN, V_UAV_MAX), (0, 2 * np.pi), (1, 30), (2, 40),
        # FY2's params
        (V_UAV_MIN, V_UAV_MAX), (0, 2 * np.pi), (1, 30), (2, 40),
        # FY3's params
        (V_UAV_MIN, V_UAV_MAX), (0, 2 * np.pi), (1, 30), (2, 40),
    ]
    n_dim = len(bounds)
    
    particles_pos = np.random.rand(n_particles, n_dim)
    for i in range(n_dim):
        particles_pos[:, i] = particles_pos[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    particles_vel = np.random.randn(n_particles, n_dim) * 0.1
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([calculate_fitness_q4(p) for p in pbest_pos])
    
    gbest_idx = np.argmax(pbest_fitness)
    gbest_pos = pbest_pos[gbest_idx]
    gbest_fitness = pbest_fitness[gbest_idx]
    
    w, c1, c2 = 0.9, 2.0, 2.0
    print("\n--- 开始优化(问题4: 3机协同) ---")
    start_time = time.time()
    
    for it in range(n_iterations):
        for i in range(n_particles):
            # (PSO核心更新逻辑与之前相同)
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
                pbest_pos[i] = particles_pos[i]
                if current_fitness > gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_pos = particles_pos[i]
        
        if (it + 1) % 5 == 0: # 减少打印频率
            print(f"迭代次数: {it + 1}/{n_iterations}, 当前最优遮蔽时间: {gbest_fitness:.3f} s")
            
    end_time = time.time()
    print(f"优化完成! 总耗时: {end_time - start_time:.2f} s")
    return gbest_pos, gbest_fitness

# --- 6. 执行与结果 ---
if __name__ == '__main__':
    # 注意：由于维度增加，计算量巨大。这里使用较少的粒子和迭代次数进行演示。
    # 为获得更好结果，应大幅增加这两个值。
    NUM_PARTICLES = 100
    NUM_ITERATIONS = 200
    
    best_solution, max_time = pso_optimizer(NUM_PARTICLES, NUM_ITERATIONS)
    
    # 解析最优解
    v1, th1, td1, tb1, v2, th2, td2, tb2, v3, th3, td3, tb3 = best_solution
    
    # 准备结果
    uavs_params = [
        ('FY1', v1, th1, td1, tb1),
        ('FY2', v2, th2, td2, tb2),
        ('FY3', v3, th3, td3, tb3),
    ]

    # <-- 修改点：采用顺序打印，而不是表格 -->
    print("\n" + "="*65)
    print(" " * 20 + "问题4 最优协同投放策略")
    print("="*65)

    for i, params in enumerate(uavs_params):
        uav_id, v, th, td, tb = params
        drop_point = uav_position(uav_id, v, th, td)
        detonation_point = grenade_position(uav_id, v, th, td, tb)
        
        print(f"无人机 {uav_id}:")
        print(f"  - 飞行速度: {v:.2f} m/s")
        print(f"  - 飞行方向: {np.rad2deg(th):.2f} 度")
        print(f"  - 投放时间: {td:.3f} s")
        print(f"  - 投放点坐标: ({drop_point[0]:.2f}, {drop_point[1]:.2f}, {drop_point[2]:.2f})")
        print(f"  - 起爆时间: {tb:.3f} s")
        print(f"  - 起爆点坐标: ({detonation_point[0]:.2f}, {detonation_point[1]:.2f}, {detonation_point[2]:.2f})")
        
        if i < len(uavs_params) - 1:
            print("-"*65)
    
    print("="*65)
    print(f"最大有效遮蔽时间: {max_time:.3f} s")
    print("="*65)

