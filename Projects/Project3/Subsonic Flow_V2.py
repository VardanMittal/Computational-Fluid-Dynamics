import numpy as np
import matplotlib.pyplot as plt

L = 3
dx = 0.05
N = int(L / dx) + 1
gamma = 1.4
R = 287

x = np.linspace(0, L, N)

dens = np.zeros(N)
dens_temp = np.zeros(N)
pres = np.zeros(N)
pres_temp = np.zeros(N)
vel = np.zeros(N)
temp = np.zeros(N)
vel_temp = np.zeros(N)
temp_temp = np.zeros(N)
eng = temp[:]

U1 = np.zeros(N)
U1_temp = np.zeros(N)
U2 = np.zeros(N)
U2_temp = np.zeros(N)
U3 = np.zeros(N)
U3_temp = np.zeros(N)
F1 = np.zeros(N)
F2 = np.zeros(N)
F3 = np.zeros(N)
J2 = np.zeros(N)

diff_U1 = np.zeros(N)
diff_U1_n1 = np.zeros(N)
diff_U2 = np.zeros(N)
diff_U2_n1 = np.zeros(N)
diff_U3 = np.zeros(N)
diff_U3_n1 = np.zeros(N)
Ap = np.zeros(N)
diff_Apf = np.zeros(N)
diff_Apb = np.zeros(N)


# Initial profile setup
for i in range(N):
    dens[i]=1-0.0003*i*dx
    temp[i]=1-0.00009333*i*dx
    vel[i]=0.05+0.11*i*dx
    

for i in range(N):
    Ap[i] = 1 + 2.2 * ((i * dx - 1.5) ** 2)

for i in range(N):
    diff_Apf[i] = ((1 + 2.2 * (((i + 1) * dx - 1.5) ** 2)) - (Ap[i])) / dx
    diff_Apb[i] = ((Ap[i]) - (1 + 2.2 * (((i - 1) * dx - 1.5) ** 2))) / dx

pend = dens[N-1]*temp[N-1] * R

U1 = U1_temp = dens * Ap
U2 = U2_temp = U1 * vel
U3 = U1_temp = dens * ((temp / (gamma - 1)) + (gamma / 2) * vel * vel) * Ap

dens = U1 / Ap
pres = dens * temp
# Time iteration
for time in range(2000):
    # Calculating flux terms with minimal correction for division by zero
    F1 = U2
    F2 = ((U2 * U2 / (U1)) + ((gamma - 1) / gamma) * (U3 - (gamma / 2) * (U2 * U2 / (U1))))
    F3 = (gamma * U2 * U3 / (U1)) - ((gamma * (gamma - 1) / 2) * (U2 * U2 * U2 / (U1) ** 2))
    J2 = ((gamma - 1) / gamma) * (U3 - (gamma / 2) * (U2 * U2 / (U1))) * diff_Apf / Ap

    for i in range(1, N - 1):
        diff_U1[i] = -(F1[i + 1] - F1[i]) / dx
        diff_U2[i] = (-(F2[i + 1] - F2[i]) / dx) + J2[i]
        diff_U3[i] = -(F3[i + 1] - F3[i]) / dx

    dt_all = np.zeros(N - 2)
    for i in range(1, N - 1):
        dt_all[i - 1] = 0.5 * dx / (vel[i] + (temp[i] + 1e-6) ** 0.5)
    dt = min(dt_all)

    # Update temporary variables
    U1_temp = U1 + dt * diff_U1
    U2_temp = U2 + dt * diff_U2
    U3_temp = U3 + dt * diff_U3

    # Boundary conditions
    U1_temp[N - 1] = 2 * U1_temp[N - 2] - U1_temp[N - 3]
    U2_temp[0] = 2 * U2[1] - U2_temp[2]
    U2_temp[N - 1] = 2 * U2_temp[N - 2] - U2_temp[N - 3]
    vel_temp = U2_temp / (U1_temp + 1e-6)
    U3_temp[0] = U1_temp[0] * ((1 / (gamma - 1)) + (gamma / 2) * vel_temp[0] * vel_temp[0])
    U3_temp[N - 1] = (pend * Ap[N - 1] / (gamma - 1)) + ((gamma / 2) * U2_temp[N - 1] * vel_temp[N - 1])

    temp_temp = (gamma - 1) * ((U3_temp / (U1_temp + 1e-6)) - (gamma / 2) * vel_temp * vel_temp)
    dens_temp = U1_temp / Ap
    pres_temp = dens_temp * temp_temp

    F1 = U2_temp
    F2 = ((U2_temp * U2_temp / (U1_temp + 1e-6)) + ((gamma - 1) / gamma) * (U3_temp - (gamma / 2) * (U2_temp * U2_temp / (U1_temp + 1e-6))))
    F3 = (gamma * U2_temp * U3_temp / (U1_temp + 1e-6)) - ((gamma * (gamma - 1) / 2) * (U2_temp * U2_temp * U2_temp / (U1_temp + 1e-6) ** 2))
    J2 = ((gamma - 1) / gamma) * (U3_temp - (gamma / 2) * (U2_temp * U2_temp / (U1_temp + 1e-6))) * diff_Apb / Ap

    for i in range(1, N - 1):
        diff_U1_n1[i] = -(F1[i] - F1[i - 1]) / dx
        diff_U2_n1[i] = (-(F2[i] - F2[i - 1]) / dx) + J2[i]
        diff_U3_n1[i] = -(F3[i] - F3[i - 1]) / dx

    # Update main variables with corrected values
    for i in range(1, N - 1):
        U1[i] += 0.5 * dt * (diff_U1[i] + diff_U1_n1[i])
        U2[i] += 0.5 * dt * (diff_U2[i] + diff_U2_n1[i])
        U3[i] += 0.5 * dt * (diff_U3[i] + diff_U3_n1[i])

    # Apply boundary conditions to the updated variables
    U1[N - 1] = 2 * U1[N - 2] - U1[N - 3]
    U2[0] = 2 * U2[1] - U2[2]
    U2[N - 1] = 2 * U2[N - 2] - U2[N - 3]
    vel = U2 / (U1 + 1e-6)
    U3[0] = U1[0] * ((1 / (gamma - 1)) + (gamma / 2) * vel[0] * vel[0])
    U3[N - 1] = (pend * Ap[N - 1] / (gamma - 1)) + ((gamma / 2) * U2[N - 1] * vel[N - 1])
    temp = (gamma - 1) * ((U3 / (U1 + 1e-6)) - (gamma / 2) * vel * vel)

# Final calculations for plotting
dens = U1 / Ap
pres = dens * temp
Mach_no = vel / np.sqrt(temp + 1e-6)

print(pres)


vect = np.arange(0, N * dx, dx)
plt.figure(1, figsize=(10, 5))
plt.plot(vect, pres)  # Pressure distribution
plt.xlabel("Nozzle Length")
plt.ylabel("Pressure")
plt.title("Pressure Variation")
plt.show()

plt.figure(2, figsize=(10, 5))
plt.plot(vect, Mach_no)  # Mach number distribution
plt.xlabel("Nozzle Length")
plt.ylabel("Mach Number")
plt.title("Mach Number")
plt.show()

plt.figure(3, figsize=(10, 5))
plt.plot(vect, temp)  # Temperature distribution
plt.xlabel("Nozzle Length")
plt.ylabel("Temperature")
plt.title("Temperature Variation")
plt.show()

plt.figure(3, figsize=(10, 5))
plt.plot(vect, dens)  # Temperature distribution
plt.xlabel("Nozzle Length")
plt.ylabel("Density")
plt.title("Density Variation")
plt.show()