import numpy as np
import time

# --- 1. 定义常量 (已根据题目核对) ---
# 导弹M1
V_M1 = 300  # 速度 (m/s)
P0_M1 = np.array([20000, 0, 2000])  # 初始位置 (m)
TARGET_FALSE = np.array([0, 0, 0])   # 假目标位置 (m)

# 无人机 FY1
P0_FY1 = np.array([17800, 0, 1800]) # 初始位置 (m)
V_FY1_MIN, V_FY1_MAX = 70, 140       # 速度范围 (m/s)

# 真目标 (圆柱体)
TARGET_TRUE_CENTER_BASE = np.array([0, 200, 0]) # 底面中心
TARGET_TRUE_RADIUS = 7    # 半径 (m)
TARGET_TRUE_HEIGHT = 10   # 高度 (m)

# 烟幕
SMOKE_CLOUD_RADIUS = 10   # 烟幕云半径 (m)
SMOKE_CLOUD_SINK_V = 3    # 烟幕云下沉速度 (m/s)
SMOKE_DURATION = 20       # 烟幕有效持续时间 (s)

# 物理常量
G = 9.8  # 重力加速度 (m/s^2)

# --- 2. 运动学模型 ---
direction_m1 = (TARGET_FALSE - P0_M1) / np.linalg.norm(TARGET_FALSE - P0_M1)
def missile_position(t):
    return P0_M1 + V_M1 * t * direction_m1

def uav_position(v_f, theta, t):
    vx = v_f * np.cos(theta)
    vy = v_f * np.sin(theta)
    return P0_FY1 + t * np.array([vx, vy, 0])

def grenade_position(v_f, theta, t_d, t):
    if t < t_d: return uav_position(v_f, theta, t)
    p0_grenade = uav_position(v_f, theta, t_d)
    v0_grenade = np.array([v_f * np.cos(theta), v_f * np.sin(theta), 0])
    dt = t - t_d
    pos = p0_grenade + v0_grenade * dt + 0.5 * np.array([0, 0, -G]) * dt**2
    return pos

def cloud_center_position(v_f, theta, t_d, t_b, t):
    if t < t_b: return None
    p_detonation = grenade_position(v_f, theta, t_d, t_b)
    dt = t - t_b
    return p_detonation + dt * np.array([0, 0, -SMOKE_CLOUD_SINK_V])

# --- 3. 几何遮蔽判断 ---
target_key_points = []
for h in [0, TARGET_TRUE_HEIGHT]:
    center = TARGET_TRUE_CENTER_BASE + np.array([0, 0, h])
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False): # 增加关键点以提高精度
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
    if c_pos is None: return False
    for point in target_key_points:
        if not is_line_segment_intercepted_by_sphere(m_pos, point, c_pos, SMOKE_CLOUD_RADIUS):
            return False
    return True

# --- 4. 适应度函数 ---
def calculate_fitness(params):
    v_f, theta, t_d, t_b = params
    if t_d < 1.5 or t_b <= t_d: return 0.0
    total_shielding_time = 0
    time_step = 0.01 # 使用更小的时间步长以提高精度
    for t in np.arange(t_b, t_b + SMOKE_DURATION, time_step):
        m_pos = missile_position(t)
        if m_pos[0] <= TARGET_TRUE_CENTER_BASE[0]: break
        c_pos = cloud_center_position(v_f, theta, t_d, t_b, t)
        if c_pos is not None and c_pos[2] < -SMOKE_CLOUD_RADIUS: break
        if is_shielded(m_pos, c_pos):
            total_shielding_time += time_step
    return total_shielding_time

# --- 5. PSO 主算法 ---
def pso_optimizer(n_particles, n_iterations, initial_solution=None):
    bounds = [(V_FY1_MIN, V_FY1_MAX), (0, 2 * np.pi), (1.5, 40), (2, 50)]
    particles_pos = np.random.rand(n_particles, 4)
    for i in range(4):
        particles_pos[:, i] = particles_pos[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    if initial_solution is not None:
        particles_pos[0] = initial_solution
    particles_vel = np.random.randn(n_particles, 4) * 0.1
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([calculate_fitness(p) for p in pbest_pos])
    gbest_idx = np.argmax(pbest_fitness)
    gbest_pos = pbest_pos[gbest_idx]
    gbest_fitness = pbest_fitness[gbest_idx]
    w, c1, c2 = 0.7, 1.5, 1.5
    print("\n--- 开始优化 ---")
    start_time = time.time()
    for it in range(n_iterations):
        for i in range(n_particles):
            current_fitness = calculate_fitness(particles_pos[i])
            if current_fitness > pbest_fitness[i]:
                pbest_fitness[i] = current_fitness
                pbest_pos[i] = particles_pos[i]
                if current_fitness > gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_pos = particles_pos[i]
        for i in range(n_particles):
            r1, r2 = np.random.rand(2)
            cognitive_vel = c1 * r1 * (pbest_pos[i] - particles_pos[i])
            social_vel = c2 * r2 * (gbest_pos - particles_pos[i])
            particles_vel[i] = w * particles_vel[i] + cognitive_vel + social_vel
            particles_pos[i] += particles_vel[i]
            for j in range(4):
                particles_pos[i, j] = np.clip(particles_pos[i, j], bounds[j][0], bounds[j][1])
        if (it + 1) % 10 == 0:
            print(f"迭代次数: {it + 1}/{n_iterations}, 当前最优遮蔽时间: {gbest_fitness:.3f} s")
    end_time = time.time()
    print(f"优化完成! 总耗时: {end_time - start_time:.2f} s")
    return gbest_pos, gbest_fitness

# --- 6. 执行与结果 ---
if __name__ == '__main__':
    # --- 第1步: 验证问题1的条件 ---
    print("--- 正在验证问题1的条件 ---")
    v_f_initial = 120.0
    t_d_initial = 1.5
    t_b_initial = 1.5 + 3.6
    direction_vector_xy = TARGET_FALSE[0:2] - P0_FY1[0:2]
    theta_initial = np.arctan2(direction_vector_xy[1], direction_vector_xy[0])
    initial_params = np.array([v_f_initial, theta_initial, t_d_initial, t_b_initial])
    shielding_time = calculate_fitness(initial_params)
    print(f"计算得到的初始遮蔽时间为: {shielding_time:.3f} s")

    # --- 第2步: 以问题1为起点，优化求解问题2 ---
    NUM_PARTICLES = 50
    NUM_ITERATIONS = 50
    best_solution, max_time = pso_optimizer(
        NUM_PARTICLES, 
        NUM_ITERATIONS, 
        initial_solution=initial_params
    )
    
    v_opt, theta_opt, td_opt, tb_opt = best_solution
    drop_point_opt = uav_position(v_opt, theta_opt, td_opt)
    detonation_point_opt = grenade_position(v_opt, theta_opt, td_opt, tb_opt)
    
    print("\n--- 问题2 最优策略 ---")
    print(f"无人机飞行速度 (v_f): {v_opt:.2f} m/s")
    print(f"无人机飞行角度 (θ): {np.rad2deg(theta_opt):.2f} 度")
    print(f"烟幕弹投放点坐标: ({drop_point_opt[0]:.2f}, {drop_point_opt[1]:.2f}, {drop_point_opt[2]:.2f})")
    print(f"烟幕弹起爆点坐标: ({detonation_point_opt[0]:.2f}, {detonation_point_opt[1]:.2f}, {detonation_point_opt[2]:.2f})")
    print(f"烟幕弹爆炸时间: {tb_opt:.3f} s")
    print("--------------------")
    print(f"最大有效遮蔽时间: {max_time:.3f} s")

# --- 问题2 最优策略 ---
# 无人机飞行速度 (v_f): 130.62 m/s
# 无人机飞行角度 (θ): 179.33 度
# 烟幕弹投放点坐标: (17604.09, 2.28, 1800.00)
# 烟幕弹起爆点坐标: (17063.57, 8.59, 1716.08)
# 烟幕弹爆炸时间: 5.638 s
# --------------------
# 最大有效遮蔽时间: 4.180 s