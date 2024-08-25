import numpy as np
import math


# J = 1/2 * x_N_gain * (x_N - x_f)^2) + 1/2 * u_i_gain * sum_0_N-1(v_k^2 + w_k^2)
# 约束为 x_i[0][0] < 3.0
# 处理之后的cost函数为：J = 1/2(1000 * (x_N - x_f)^2) + 1/2 sum_0_N-1((v_k^2 + w_k^2) -1/t * log(-(x_k[0][0] - 3.0)))
class System:

    def __init__(self, N_state, N_control, x_f, T_s, T_N, constrained, t, x_max, x_N_gain, u_i_gain):
        self.N_state = N_state
        self.N_control = N_control
        self.x_f = x_f
        self.T_s = T_s
        self.T_N = T_N
        self.constrained = constrained
        self.t = t
        self.x_max = x_max
        self.x_N_gain = x_N_gain
        self.u_i_gain = u_i_gain

    def GetTerminalCost(self, x_N):
        tmp = (0.5 * self.x_N_gain * (x_N - self.x_f).T @ (x_N - self.x_f))
        return tmp

    def GetProgressCost(self, u_i, x_i):
        if self.constrained:
            return (0.5 * self.u_i_gain *  (self.T_s * u_i).T @ (self.T_s * u_i)) - 1.0 / self.t * math.log(self.x_max - x_i[0][0])
        else:
            return (0.5 * self.u_i_gain * (self.T_s * u_i).T @ (self.T_s * u_i))

    def GetTerminalCostToGoJacobian(self, x_N):
        return self.x_N_gain * (x_N - self.x_f)

    def GetTerminalCostToGoHession(self, x_N):
        return self.x_N_gain * np.eye(self.N_state)

    # x, u
    def GetCostToGoJacobian(self, x_i, u_i):
        if self.constrained:
            return np.array([[1.0 / self.t / (self.x_max - x_i[0][0])],
                             [0.0],
                             [0.0],
                             [self.u_i_gain * self.T_s * u_i[0][0]],
                             [self.u_i_gain * self.T_s * u_i[1][0]]
                             ])
        else:
            return np.array([[0.0],
                             [0.0],
                             [0.0],
                             [self.u_i_gain * self.T_s * u_i[0][0]],
                             [self.u_i_gain * self.T_s * u_i[1][0]]
                             ])

    def GetCostToGoHession(self, x_i, u_i):
        hession = np.zeros((self.N_state + self.N_control, self.N_state + self.N_control))
        if self.constrained:
            hession[0, 0] = 1.0 / self.t / ((self.x_max - x_i[0][0])**2)
            hession[-2, -2] = self.u_i_gain * self.T_s
            hession[-1, -1] = self.u_i_gain * self.T_s
        else:
            hession[-2, -2] = self.u_i_gain * self.T_s
            hession[-1, -1] = self.u_i_gain * self.T_s
        return hession

    def Get_A(self, x_i, u_i):
        A = np.eye(3)
        # A[0][2] = -math.sin(x_i[-1]) * u_i[0] * self.T_s
        # A[1][2] = math.cos(x_i[-1]) * u_i[0] * self.T_s
        return A

    def Get_B(self, x_i, u_i):
        tmp_1 = math.cos(x_i[-1][0])
        tmp_2 = math.sin(x_i[-1][0])
        return self.T_s * np.array([[tmp_1, 0.0], [tmp_2, 0.0], [0.0, 1.0]])

    def Get_A_inter(self, x_i, u_i):
        A = np.eye(3)
        return A

    def Get_B_inter(self, x_i, u_i):
        tmp_1 = math.cos(x_i[-1][0])
        tmp_2 = math.sin(x_i[-1][0])
        return self.T_s * np.array([[tmp_1, 0.0], [tmp_2, 0.0], [0.0, 1.0]])

    def Calc_next_state(self, x_i, u_i):
        A = self.Get_A_inter(x_i, u_i)
        B = self.Get_B_inter(x_i, u_i)
        tmp = A @ x_i + B @ u_i
        return tmp


class iLQR:

    def Slove(self, max_iters, system, x_init, u_init, cost_init):
        cost_old = cost_init
        x_state = x_init
        u_control = u_init
        for it in range(max_iters):
            K, d, delta_J_hat = self.BackwardPass(system, x_state, u_control)
            if it > 3 and np.abs(delta_J_hat) < 1e-5: break
            x_state_new, u_control_new, cost_new = self.ForwardPass(system, x_state, u_control, K, d)
            if cost_old - cost_new > 0:
                cost_old = cost_new
                x_state = x_state_new
                u_control = u_control_new
        return x_state, u_control, cost_old

    def BackwardPass(self, system, x_state, u_control):
        K_i = np.zeros((system.N_control, system.N_state))
        d_i = np.zeros((system.N_control, 1))
        delta_J_hat = 0

        K = np.zeros((system.N_control * system.T_N, system.N_state))
        d = np.zeros((system.N_control * system.T_N, 1))

        p_jacobian_x = system.GetTerminalCostToGoJacobian(x_state[-3:])
        P_hssion_x = system.GetTerminalCostToGoHession(x_state[-3:])
        for i in range(system.T_N - 1, -1, -1):
            x_i = x_state[i * system.N_state: (i + 1) * system.N_state]
            u_i = u_control[i * system.N_control: (i + 1) * system.N_control]
            A_i = system.Get_A(x_i, u_i)
            B_i = system.Get_B(x_i, u_i)

            l_jacobian = system.GetCostToGoJacobian(x_i, u_i)
            l_x = l_jacobian[0: system.N_state]
            l_u = l_jacobian[system.N_state:]
            l_hession = system.GetCostToGoHession(x_i, u_i)
            l_xx = l_hession[0: system.N_state, 0: system.N_state]
            l_xu = l_hession[0: system.N_state, system.N_state:]
            l_ux = l_hession[system.N_state:, 0: system.N_state]
            l_uu = l_hession[system.N_state:, system.N_state:]

            Q_x_i = l_x + A_i.T @ p_jacobian_x
            Q_u_i = l_u + B_i.T @ p_jacobian_x
            Q_x_i_x_i = l_xx + A_i.T @ P_hssion_x @ A_i
            Q_x_i_u_i = l_xu + A_i.T @ P_hssion_x @ B_i
            Q_u_i_x_i = l_ux + B_i.T @ P_hssion_x @ A_i
            Q_u_i_u_i = l_uu + B_i.T @ P_hssion_x @ B_i

            Q_u_i_u_i_inv = np.linalg.inv(Q_u_i_u_i)
            K_i = -Q_u_i_u_i_inv @ Q_u_i_x_i
            d_i = -Q_u_i_u_i_inv @ Q_u_i

            p_jacobian_x = Q_x_i + K_i.T @ Q_u_i_u_i @ d_i + Q_x_i_u_i @ d_i + K_i.T @ Q_u_i
            P_hssion_x = Q_x_i_x_i + K_i.T @ Q_u_i_u_i @ K_i + Q_x_i_u_i @ K_i + K_i.T @ Q_u_i_x_i

            K[i * system.N_control: (i + 1) * system.N_control][:] = K_i
            d[i * system.N_control: (i + 1) * system.N_control][:] = d_i
            delta_J_hat += 0.5 * d_i.T @ Q_u_i_u_i @ d_i + Q_u_i.T @ d_i

        return K, d, delta_J_hat

    def ForwardPass(self, system, x_state, u_control, K, d):
        x_state_new = np.zeros((system.N_state * (system.T_N + 1), 1))
        cost_new = 0.0
        x_state_new[0: 3] = x_state[0: 3]
        u_control_new = np.zeros((system.N_control * system.T_N, 1))

        for i in range(0, system.T_N):
            u_last = u_control[i * system.N_control: (i + 1) * system.N_control]
            delta_x = x_state_new[i * system.N_state: (i + 1) * system.N_state] - x_state[i * system.N_state: (
                                                                                                                          i + 1) * system.N_state]
            u_control_new[i * system.N_control: (i + 1) * system.N_control] = u_last + K[i * system.N_control: (
                                                                                                                       i + 1) * system.N_control] @ delta_x + d[
                                                                                                                                                              i * system.N_control:(
                                                                                                                                                                                           i + 1) * system.N_control]
            x_state_new[(i + 1) * system.N_state: (i + 2) * system.N_state] = system.Calc_next_state(
                x_state_new[i * system.N_state: (i + 1) * system.N_state],
                u_control_new[i * system.N_control: (i + 1) * system.N_control])
            cost_new += system.GetProgressCost(u_control_new[i * system.N_control: (i + 1) * system.N_control],
                                               x_state_new[i * system.N_state: (i + 1) * system.N_state])

        cost_new += system.GetTerminalCost(x_state_new[-3:])

        return x_state_new, u_control_new, cost_new
