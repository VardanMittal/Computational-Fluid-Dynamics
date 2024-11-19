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
    du = np.zeros((Nx+1, Ny))

    De = dy / dx
    Dw = dy / dx
    Dn = dx / dy
    Ds = dx / dy

    for j in range(1, Ny-1):
        a = b = c = d = np.zeros(Nx + 1)
        for i in range(1,Nx - 1):
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

    for i in range(1, Nx-1):
        a = b = c = d = np.zeros(Nx + 1)
        for j in range(1, Ny - 1):
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
        u_star[i, :-1] = TDMA(a[1:-1], b[1:-1], c[1:-1], d[1:-1])
    u_star[0, :] = -u_star[1, :]  # Left wall
    u_star[-1, :] = -u_star[-2, :]  # Right wall
    u_star[:, 0] = 0.0  # Bottom wall
    u_star[:, -1] = velocity  # Top

    return u_star, du

def y_momentum_solver(Nx,Ny,dx,dy, u, v, velocity, p):
    v_star = np.zeros((Nx, Ny+1))
    dv = np.zeros((Nx,Ny+1))

    De =  dy / dx  # Convective coefficients
    Dw =dy / dx
    Dn =  dx / dy
    Ds = dx / dy


    # ADI Method
    # First pass: y-direction
    for i in range(1, Nx - 1):
        # Thomas algorithm for tridiagonal matrix
        a = b = c = d = np.zeros(Ny + 1)

        for j in range(1, Ny - 1):
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
    for j in range(1, Ny - 1):
        # Thomas algorithm for tridiagonal matrix
        a = b =c = d = np.zeros(Ny + 1)
 
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
        v_star[:-1, j] = TDMA(a[1:-1], b[1:-1], c[1:-1], d[1:-1])

    # Apply boundary conditions
    v_star[0, :] = 0.0  # Left wall
    v_star[1, :] = 0.0  # Right wall
    v_star[:, 0] = -v_star[:, 1]  # Bottom wall
    v_star[:, -1] = -v_star[:, -2]  # Top wall

    return v_star, dv

def pressure_correction(Nx, Ny, dx, dy, u_star, v_star, p, max_iter=100):
    """
    Pressure correction step using ADI method with TDMA for the lid-driven cavity problem.
    """
    p_prime = np.zeros((Nx, Ny))
    b = np.zeros((Nx, Ny))  # Right-hand side array for the Poisson equation
    
    # Compute the RHS for the Poisson equation
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            b[i, j] = (u_star[i, j] - u_star[i-1, j]) * dy + (v_star[i, j] - v_star[i, j-1]) * dx
    
    for it in range(max_iter):
        # Step 1: Sweep in the x-direction
        for j in range(1, Ny-1):
            # Coefficients for the TDMA in x-direction
            a = np.full(Nx-2, -dy / dx**2)  # Sub-diagonal
            b_diag = np.full(Nx-2, 2 * (dy / dx**2 + dx / dy**2))  # Main diagonal
            c = np.full(Nx-2, -dy / dx**2)  # Super-diagonal
            d = np.zeros(Nx-2)  # RHS
            
            # Fill RHS
            for i in range(1, Nx-1):
                d[i-1] = b[i, j] - (dx / dy**2) * (p_prime[i, j-1] + p_prime[i, j+1])
            
            # Solve using TDMA
            p_prime[1:Nx-1, j] = TDMA(a, b_diag, c, d)
        
        # Step 2: Sweep in the y-direction
        for i in range(1, Nx-1):
            # Coefficients for the TDMA in y-direction
            a = np.full(Ny-2, -dx / dy**2)  # Sub-diagonal
            b_diag = np.full(Ny-2, 2 * (dy / dx**2 + dx / dy**2))  # Main diagonal
            c = np.full(Ny-2, -dx / dy**2)  # Super-diagonal
            d = np.zeros(Ny-2)  # RHS
            
            # Fill RHS
            for j in range(1, Ny-1):
                d[j-1] = b[i, j] - (dy / dx**2) * (p_prime[i-1, j] + p_prime[i+1, j])
            
            # Solve using TDMA
            p_prime[i, 1:Ny-1] = TDMA(a, b_diag, c, d)
        
        # Update pressure
        p[1:Nx-1, 1:Ny-1] += p_prime[1:Nx-1, 1:Ny-1]
        
        # Apply boundary conditions (zero pressure at boundaries for simplicity)
        p[0, :] = p[-1, :] = p[:, 0] = p[:, -1] = 0
        
        # Check for convergence
        if np.linalg.norm(p_prime) < 1e-5:
            print(f"Converged after {it+1} iterations.")
            break
    
    return p, p_prime


import numpy as np

def update_velocity(Nx, Ny, u_star, v_star, p_prime, d_u, d_v, velocity):

    v = np.zeros((Nx, Ny + 1))
    u = np.zeros((Nx + 1, Ny))
    
    # Update interior nodes of u
    for i in range(1, Nx):
        for j in range(1, Ny - 1):
            u[i, j] = u_star[i, j] + d_u[i, j] * (p_prime[i - 1, j] - p_prime[i, j])
    
    # Update interior nodes of v
    for i in range(1, Nx - 1):
        for j in range(1, Ny):
            v[i, j] = v_star[i, j] + d_v[i, j] * (p_prime[i, j - 1] - p_prime[i, j])
    
    # Apply boundary conditions for v
    v[0, :] = 0.0  # Left wall
    v[Nx - 1, :] = 0.0  # Right wall
    v[:, 0] = -v[:, 1]  # Bottom wall
    v[:, Ny] = -v[:, Ny - 1]  # Top wall
    
    # Apply boundary conditions for u
    u[0, :] = -u[1, :]  # Left wall
    u[Nx, :] = -u[Nx - 1, :]  # Right wall
    u[:, 0] = 0.0  # Bottom wall
    u[:, Ny - 1] = velocity  # Top wall
    
    return u, v
