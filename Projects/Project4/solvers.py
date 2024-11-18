import numpy as np

def TDMA(d,l,u,b):
    """Function that solves TDMA
    
    Keyword arguments:
    d - np.array(): diagonal vector
    b - np.array(): known vector
    l - np.array(): lower diagonal vector
    u - np.array(): upper diagonal vector
    Return: x -> solution vector
    """
    n = len(d)
    for i in range(1, n):
        factor = l[i-1] / d[i-1]
        d[i] = d[i] - factor * u[i-1]
        b[i] = b[i] - factor * b[i-1]
    x = np.zeros(n)
    x[-1] = b[-1] / d[-1]

    for i in reversed(range(n-1)):
        x[i] = (b[i] - u[i] * x[i+1]) / d[i]

    return x

def x_momentum_solver(Nx,Ny,dx,dy, u, v, velocity, p):
    u_star = np.zeros((Nx+1, Ny))
    du = np.zeros((Nx+1,Ny))

    De = dy / dx
    Dw = dy / dx
    Dn = dx / dy
    Ds = dx / dy

    for j in range(1, Ny-1):
        a = b = c = d = np.zeros(Nx+1)
        for i in range(1,Nx):
            Fe = 0.5 * dy * (u[i + 1, j] + u[i, j])
            Fw = 0.5 * dy * (u[i - 1, j] + u[i, j])
            Fn = 0.5 * dx * (v[i, j + 1] + v[i - 1, j + 1])
            Fs = 0.5 * dx * (v[i, j] + v[i - 1, j])

            aE = De * Fe + max(-Fe, 0)
            aW = Dw * Fw + max(Fw, 0)
            aN = Dn * Fn + max(-Fn, 0)
            aS = Ds * Fs + max(Fs, 0)
            aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)

            pressure_term = (p[i - 1, j] - p[i, j]) * dy

            a[i] = -aW
            b[i] = aP
            c[i] = -aE
            d[i] = (aN * u[i, j + 1] + aS * u[i, j - 1] + pressure_term) + u[i, j]

        u_star[1:-1,j] = TDMA(a[1:-1], b[1:-1], c[1:-1], d[1:-1])

    for i in range(1, Nx):
        a = b = c = d = np.zeros(Nx+1)
        for j in range(1, Ny-1):
            Fe = 0.5 *  dy * (u[i + 1, j] + u[i, j])
            Fw = 0.5 *  dy * (u[i - 1, j] + u[i, j])
            Fn = 0.5 *  dx * (v[i, j + 1] + v[i - 1, j + 1])
            Fs = 0.5 * dx * (v[i, j] + v[i - 1, j])

            aE = De * Fe + max(-Fe, 0)
            aW = Dw * Fw + max(Fw, 0)
            aN = Dn * Fn + max(-Fn, 0)
            aS = Ds * Fs + max(Fs, 0)
            aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)

            pressure_term = (p[i - 1, j] - p[i, j]) * dy

            a[j] = -aS
            b[j] = aP
            c[j] = -aN
            d[j] = (aE * u_star[i + 1, j] + aW * u_star[i - 1, j] + pressure_term) + u_star[i, j]
        u_star[i, 1:-1] = TDMA(a[1:-1], b[1:-1], c[1:-1], d[1:-1])
    u_star[0, :] = -u_star[1, :]  # Left wall
    u_star[-1, :] = -u_star[-2, :]  # Right wall
    u_star[:, 0] = 0.0  # Bottom wall
    u_star[:, -1] = velocity  # Top

    return u_star, du

def y_momentum_solver(Nx,Ny,dx,dy, u, v, velocity, p):
    v_star = np.zeros((Nx, Ny + 1))
    d_v = np.zeros((Nx, Ny + 1))

    De =  dy / dx  # Convective coefficients
    Dw =dy / dx
    Dn =  dx / dy
    Ds = dx / dy


    # ADI Method
    # First pass: y-direction
    for i in range(1, Nx):
        # Thomas algorithm for tridiagonal matrix
        a = np.zeros(Ny + 1)
        b = np.zeros(Ny + 1)
        c = np.zeros(Ny + 1)
        d = np.zeros(Ny + 1)

        for j in range(1, Ny):
            Fe = 0.5 * dy * (u[i + 1, j] + u[i + 1, j - 1])
            Fw = 0.5 * dy * (u[i, j] + u[i, j - 1])
            Fn = 0.5 * dx * (v[i, j + 1] + v[i - 1, j + 1])
            Fs = 0.5 * dx * (v[i, j] + v[i - 1, j])

            aE = De * Fe + max(-Fe, 0)
            aW = Dw * Fw + max(Fw, 0)
            aN = Dn * Fn + max(-Fn, 0)
            aS = Ds * Fs + max(Fs, 0)
            aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)

            pressure_term = (p[i, j - 1] - p[i, j]) * dx

            a[j] = -aS
            b[j] = aP
            c[j] = -aN
            d[j] = (aE * v[i + 1, j] + aW * v[i - 1, j] + pressure_term) + v[i, j]

        # Solve tridiagonal system
        v_star[i, 1:-1] = TDMA(a[1:-1], b[1:-1], c[1:-1], d[1:-1])

    # Second pass: x-direction
    for j in range(1, Ny):
        # Thomas algorithm for tridiagonal matrix
        a = np.zeros(Nx + 1)
        b = np.zeros(Nx + 1)
        c = np.zeros(Nx + 1)
        d = np.zeros(Nx + 1)

        for i in range(1, Nx - 1):
            Fe = 0.5 * dy * (u[i + 1, j] + u[i, j])
            Fw = 0.5 * dy * (u[i - 1, j] + u[i, j])
            Fn = 0.5  * dx * (v[i, j + 1] + v[i, j + 1])
            Fs = 0.5 * dx * (v[i, j] + v[i, j - 1])

            aE = De * Fe + max(-Fe, 0)
            aW = Dw * Fw + max(Fw, 0)
            aN = Dn * Fn + max(-Fn, 0)
            aS = Ds * Fs  + max(Fs, 0)
            aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)

            pressure_term = (p[i - 1, j] - p[i, j]) * dy

            a[i] = -aW
            b[i] = aP
            c[i] = -aE
            d[i] = (aN * v_star[i, j + 1] + aS * v_star[i, j - 1] + pressure_term) + v_star[i, j]

        # Solve tridiagonal system
        v_star[1:-1, j] = TDMA(a[1:-1], b[1:-1], c[1:-1], d[1:-1])

    # Apply boundary conditions
    v_star[0, :] = 0.0  # Left wall
    v_star[Nx, :] = 0.0  # Right wall
    v_star[:, 0] = -v_star[:, 1]  # Bottom wall
    v_star[:, -1] = -v_star[:, -2]  # Top wall

    return v_star, d_v

import numpy as np

def pressure_correction(Nx, Ny, dx, dy, u_star, v_star, p, max_iter = 100):
    # Initialize the pressure correction array
    p_prime = np.zeros((Nx, Ny))
    
    # Initialize the matrix for storing results
    A_x = np.zeros((Nx, Ny))
    A_y = np.zeros((Nx, Ny))
    
    # Set up the right-hand side vector for the pressure correction
    b_x = np.zeros((Nx, Ny))
    b_y = np.zeros((Nx, Ny))
    
    # Precompute the right-hand side based on velocity components
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            b_x[i, j] = (u_star[i, j] - u_star[i-1, j]) * dy
            b_y[i, j] = (v_star[i, j] - v_star[i, j-1]) * dx
    
    # Main iteration loop for ADI
    for it in range(max_iter):
        # Solve in the x-direction (implicit in x)
        # Build the A_x matrix for the x-direction equation
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                A_x[i, j] = - dy / dx  # Coefficients in the x-direction
                A_x[i, j] += dy / dx  # Adjust based on the actual discretization
        
        # Solve the system in the x-direction
        p_prime_x = np.linalg.solve(A_x, b_x.flatten())
        p_prime[1:Nx-1, 1:Ny-1] = p_prime_x.reshape((Nx-2, Ny-2))
        
        # Solve in the y-direction (implicit in y)
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                A_y[i, j] = dx / dy  # Coefficients in the y-direction
                A_y[i, j] += dx / dy
        
        # Solve the system in the y-direction
        p_prime_y = np.linalg.solve(A_y, b_y.flatten())
        p_prime[1:Nx-1, 1:Ny-1] = p_prime_y.reshape((Nx-2, Ny-2))

        # Update the pressure
        p[1:Nx-1, 1:Ny-1] += p_prime[1:Nx-1, 1:Ny-1]
        
        # Apply boundary conditions (assuming zero pressure at the boundaries)
        p[0, :] = p[-1, :] = p[:, 0] = p[:, -1] = 0
        
        # Check for convergence
        if np.linalg.norm(p_prime) < 1e-5:
            print(f"Converged after {it+1} iterations.")
            break
    
    return p, p_prime

