import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def generate_plots(P, x, rho, V, T, M, title): 
    plt.figure(figsize=(8, 6))
    plt.plot(x, rho, color='m')
    plt.title(f'{title} - Density')
    plt.xlabel('Nozzle X-Direction')
    plt.ylabel('rho/rho0')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(x, T, color='c')
    plt.title(f'{title} - Temperature')
    plt.xlabel('Nozzle X-Direction')
    plt.ylabel('T/T0')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(x, P, color='g')
    plt.title(f'{title} - Pressure')
    plt.xlabel('Nozzle X-Direction')
    plt.ylabel('P/P0')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(x, M, color='y')
    plt.title(f'{title} - Mach Number')
    plt.xlabel('Nozzle X-Direction')
    plt.ylabel('Mach No.')
    plt.show()

def calc_flux(Q1, Q2, Q3, gamma):
    F1 = Q2
    F2 = (Q2 ** 2 / Q1) + (gamma - 1) / gamma * (Q3 - (gamma / 2) * Q2 ** 2 / Q1)
    F3 = (gamma * Q2 * Q3 / Q1) - (gamma * (gamma - 1) / 2) * (Q2 ** 3 / Q1 ** 2)
    return F1, F2, F3

time_step = lambda dx,CFL,T,v: np.min(CFL * dx / (np.sqrt(T) + np.abs(v)))


def mac_cormac(rho, A, v, T, Mach_no, P, Q1, Q2, Q3, gamma, CFL, Nx, dx, f):
    S = np.zeros(Nx)
    Q1_p = np.zeros(Nx)
    Q2_p = np.zeros(Nx)
    Q3_p = np.zeros(Nx)
    Q1_c = np.zeros(Nx)
    Q2_c = np.zeros(Nx)
    Q3_c = np.zeros(Nx)
    
    for z in range(1000):
        Q1_old = Q1.copy()
        Q2_old = Q2.copy()
        Q3_old = Q3.copy()

        dt = time_step(dx, CFL, T, v)

        # Predictor step
        f()
        F1, F2, F3 = calc_flux(Q1, Q2, Q3, gamma)
        S[1:-1] = (1 / gamma) * rho[1:-1] * T[1:-1] * (A[2:] - A[1:-1]) / dx
        Q1_p[1:-1] = -(F1[2:] - F1[1:-1]) / dx

        Q2_p[1:-1] = -(F2[2:] - F2[1:-1]) / dx + S[1:-1]

        Q3_p[1:-1] = -(F3[2:] - F3[1:-1]) / dx
        Q1[1:-1] += Q1_p[1:-1] * dt
        Q2[1:-1] += Q2_p[1:-1] * dt
        Q3[1:-1] += Q3_p[1:-1] * dt

        
        # Corrector step
        F1, F2, F3 = calc_flux(Q1, Q2, Q3, gamma)
        S[1:-1] = (1 / gamma) * rho[1:-1] * T[1:-1] * (A[1:-1] - A[:-2]) / dx

        Q1_c[1:-1] = -(F1[1:-1] - F1[:-2]) / dx
        Q2_c[1:-1] = -(F2[1:-1] - F2[:-2]) / dx + S[1:-1]
        Q3_c[1:-1] = -(F3[1:-1] - F3[:-2]) / dx

        Q1[1:-1] = Q1_old[1:-1] + 0.5 * (Q1_p[1:-1] + Q1_c[1:-1]) * dt
        Q2[1:-1] = Q2_old[1:-1] + 0.5 * (Q2_p[1:-1] + Q2_c[1:-1]) * dt
        Q3[1:-1] = Q3_old[1:-1] + 0.5 * (Q3_p[1:-1] + Q3_c[1:-1]) * dt

        f()
        # Calculate primitive variables for next time step
        rho = Q1 / A
        v = Q2 / Q1
        T = (gamma - 1) * (Q3 / Q1 - (gamma / 2) * v ** 2)
        P = rho * T
        Mach_no = v / np.sqrt(T)
    return P, rho, v, T, Mach_no
