from iLQR import *
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow

N_state = 3
N_control = 2
x_0 = np.array([[0.0], [0.0], [0.0]])
x_f = np.array([[2.0], [7.0], [math.pi * 0.8]])

T_s = 0.1
T = 10
T_N = int(T / T_s)
max_iters = 50

sys = System(N_state, N_control, x_f, T_s, T_N)

x_state_init = np.zeros((N_state * (T_N + 1), 1))
u_control_init = np.zeros((N_control * T_N, 1))
v_init = 0.7
w_init = 0.3
x_state_init[0:3] = x_0
u_control_init[N_control * (T_N - 1) : N_control * T_N] = np.array([[v_init], [w_init]])
init_cost = 0.0
for i in range(T_N):
    u_control_init[i * N_control: (i + 1) * N_control] = np.array([[v_init], [w_init]])
    x_state_init[(i + 1) * N_state: (i + 2) * N_state] = sys.Calc_next_state(
        x_state_init[i * N_state: (i + 1) * N_state], u_control_init[i * N_control: (i + 1) * N_control])
    init_cost += sys.GetProgressCost(u_control_init[i * N_control: (i + 1) * N_control])

init_cost += sys.GetTerminalCost(x_state_init[-3: ])

ilqr_planner = iLQR()

rlt_state = ilqr_planner.Slove(max_iters, sys, x_state_init, u_control_init, init_cost)

fig, ax = plt.subplots()
length = 0.5  # 箭头的长度

# 绘制箭头
arrow1 = Arrow(x_0[0], x_0[1], math.cos(x_0[2]) * length, math.sin(x_0[2]) * length, width=1.0, color='r')
ax.add_patch(arrow1)
arrow2 = Arrow(x_f[0], x_f[1], math.cos(x_f[2]) * length, math.sin(x_f[2]) * length, width=1.0, color='r')
ax.add_patch(arrow2)

for i in range(T_N + 1):
    x = x_state_init[i * N_state + 0]
    y = x_state_init[i * N_state + 1]
    heading = x_state_init[i * N_state + 2]
    arrow_init = Arrow(x, y, math.cos(heading) * length, math.sin(heading) * length, width=0.1, color='y')
    ax.add_patch(arrow_init)

for i in range(T_N + 1):
    x = rlt_state[i * N_state + 0]
    y = rlt_state[i * N_state + 1]
    heading = rlt_state[i * N_state + 2]
    arrow_init = Arrow(x, y, math.cos(heading) * length, math.sin(heading) * length, width=0.1, color='b')
    ax.add_patch(arrow_init)

# 设置坐标轴范围和比例
ax.set_xlim(-1, 10)
ax.set_ylim(-1, 10)
ax.set_aspect('equal')  # 确保x轴和y轴比例相同
# 显示图形
plt.show()
