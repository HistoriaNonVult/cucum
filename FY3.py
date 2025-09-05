import numpy as np
import time

# --- 1. 定义常量 (聚焦于FY3) ---
# 导弹M1
V_M1 = 300
P0_M1 = np.array([20000, 0, 2000])
TARGET_FALSE = np.array([0, 0, 0])

# 无人机 FY3 (关键修改：只保留FY3的初始位置)
P0_UAV = np.array([6000, -3000, 700])
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

# --- 2. 运动学模型 (已简化，不再需要uav_id) ---
direction_m1 = (TARGET_FALSE - P0_M1) / np.linalg.norm(TARGET_FALSE - P0_M1)
def missile_position(t):
    return P0_M1 + V_M1 * t * direction_m1

def uav_position(v_f, theta, t):
    vx = v_f * np.cos(theta)
    vy = v_f * np.sin(theta)
    return P0_UAV + t * np.array([vx, vy, 0])

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

# --- 4. 适应度函数 (已简化为单弹计算) ---
def calculate_fitness(params):
    v_f, theta, t_d, t_b = params
    if not (t_b > t_d): return 0.0

    total_shielding_time = 0
    time_step = 0.1
    for t in np.arange(t_b, t_b + SMOKE_DURATION, time_step):
        m_pos = missile_position(t)
        if m_pos[0] <= TARGET_TRUE_CENTER_BASE[0]: break
        c_pos = cloud_center_position(v_f, theta, t_d, t_b, t)
        if is_shielded(m_pos, c_pos):
            total_shielding_time += time_step
    return total_shielding_time

# --- 新增辅助函数: 计算遮蔽时间段 ---
def calculate_shield_details(params):
    v_f, theta, t_d, t_b = params
    if not (t_b > t_d): return 0.0, -1.0, -1.0
    
    total_shielding_time = 0.0
    start_shield = -1.0
    end_shield = -1.0
    time_step = 0.1
    
    for t in np.arange(t_b, t_b + SMOKE_DURATION, time_step):
        m_pos = missile_position(t)
        if m_pos[0] <= TARGET_TRUE_CENTER_BASE[0]: break
        c_pos = cloud_center_position(v_f, theta, t_d, t_b, t)
        if is_shielded(m_pos, c_pos):
            if start_shield < 0:
                start_shield = t
            end_shield = t
            total_shielding_time += time_step
            
    return total_shielding_time, start_shield, end_shield

# --- 5. PSO 主算法 (已简化为4维优化) ---
def pso_optimizer(n_particles, n_iterations):
    bounds = [(V_UAV_MIN, V_UAV_MAX), (0, 2 * np.pi), (1, 30), (2, 40)]
    n_dim = len(bounds)
    
    particles_pos = np.random.rand(n_particles, n_dim)
    for i in range(n_dim):
        particles_pos[:, i] = particles_pos[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    particles_vel = np.random.randn(n_particles, n_dim) * 0.1
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([calculate_fitness(p) for p in pbest_pos])
    
    gbest_idx = np.argmax(pbest_fitness)
    gbest_pos = pbest_pos[gbest_idx]
    gbest_fitness = pbest_fitness[gbest_idx]
    
    w, c1, c2 = 0.9, 2.0, 2.8
    print("\n--- 开始优化(单机FY3) ---")
    start_time = time.time()
    
    for it in range(n_iterations):
        for i in range(n_particles):
            # (PSO核心更新逻辑)
            r1, r2 = np.random.rand(2)
            cognitive_vel = c1 * r1 * (pbest_pos[i] - particles_pos[i])
            social_vel = c2 * r2 * (gbest_pos - particles_pos[i])
            particles_vel[i] = w * particles_vel[i] + cognitive_vel + social_vel
            particles_pos[i] += particles_vel[i]
            for j in range(n_dim):
                particles_pos[i, j] = np.clip(particles_pos[i, j], bounds[j][0], bounds[j][1])

            current_fitness = calculate_fitness(particles_pos[i])
            
            if current_fitness > pbest_fitness[i]:
                pbest_fitness[i] = current_fitness
                pbest_pos[i] = particles_pos[i]
                if current_fitness > gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_pos = particles_pos[i]
        
        if (it + 1) % 10 == 0:
            print(f"迭代次数: {it + 1}/{n_iterations}, 当前最优遮蔽时间: {gbest_fitness:.3f} s")
            
    end_time = time.time()
    print(f"优化完成! 总耗时: {end_time - start_time:.2f} s")
    return gbest_pos, gbest_fitness

# --- 6. 执行与结果 ---
if __name__ == '__main__':
    NUM_PARTICLES = 1000
    NUM_ITERATIONS = 200
    
    best_solution, max_time = pso_optimizer(NUM_PARTICLES, NUM_ITERATIONS)
    
    v_opt, theta_opt, td_opt, tb_opt = best_solution
    
    drop_point = uav_position(v_opt, theta_opt, td_opt)
    detonation_point = grenade_position(v_opt, theta_opt, td_opt, tb_opt)
    
    _, start_t, end_t = calculate_shield_details(best_solution)

    print("\n" + "="*65)
    print(" " * 18 + "无人机 FY3 单机最优投放策略")
    print("="*65)
    print(f"  - 飞行速度: {v_opt:.2f} m/s")
    print(f"  - 飞行方向: {np.rad2deg(theta_opt):.2f} 度")
    print(f"  - 投放时间: {td_opt:.3f} s")
    print(f"  - 投放点坐标: ({drop_point[0]:.2f}, {drop_point[1]:.2f}, {drop_point[2]:.2f})")
    print(f"  - 起爆时间: {tb_opt:.3f} s")
    print(f"  - 起爆点坐标: ({detonation_point[0]:.2f}, {detonation_point[1]:.2f}, {detonation_point[2]:.2f})")
    if start_t > 0:
        print(f"  - 遮蔽时间段: {start_t:.3f}s - {end_t:.3f}s")
    else:
        print(f"  - 遮蔽时间段: 无")
    print("="*65)
    print(f"最大有效遮蔽时间: {max_time:.3f} s")
    print("="*65)
