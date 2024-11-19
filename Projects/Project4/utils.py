import numpy as np
def mesh_linking(A_p,A_e,A_s,A_n,A_w, u_star,v_star, u_face,v_face,p, source_x, source_y, alpha_uv, dy, dx, Re, velocity):
    D_e = dy / (dx * Re)
    D_w = dy / (dx * Re)
    D_n = dx / (dy * Re)
    D_s = dx / (dy * Re)

    ## For internal nodes
    for i in range(2, Ny):
        for j in range(2, Nx):
            n = (i-1) * Nx + (j - 1)
            
            Fe = dy * u_face[i,j]
            Fw = dy * u_face[i, j-1]
            Fn = dx * v_face[i-1, j]
            Fs = dx * v_face[i,j]

            A_e[i,j] = D_e + max(0.0, -Fe)
            A_w[i,j] = D_w + max(0.0, Fw)
            A_n[i,j] = D_n + max(0.0, -Fn)
            A_s[i,j] = D_s + max(0.0, Fs)
            A_p[i,j] = D_e + D_w + D_n + D_s + max(0.0, Fe) + max(0.0, -Fw) +  max(0.0, Fn) + max(0.0, -Fs)
            source_x[n] = 0.5 * alpha_uv(p[i, j-1] - p[i, j+1]) * dy + (1-alpha_uv) * A_p[i,j]*u_star[i,j]
            source_y[n] = 0.5 * alpha_uv(p[i+1, j] - p[i-1, j]) * dy + (1-alpha_uv) * A_p[i,j]*v_star[i,j]
    ## Left wall
    j = 1
    for i in range(2, Ny):
        n = (i - 1)* Nx
        Fe = dy * u_face[i,j]
        Fw = dy * u_face[i, j-1]
        Fn = dx * v_face[i-1, j]
        Fs = dx * v_face[i,j]

        A_e[i,j] = D_e + max(0.0, -Fe)
        A_n[i,j] = D_n + max(0.0, -Fn)
        A_s[i,j] = D_s + max(0.0, Fs)
        A_p[i,j] = D_e + 2 * D_w + D_n + D_s + max(0.0, Fe) + max(0.0, -Fw) +  max(0.0, Fn) + max(0.0, -Fs)
        source_x[n] = 0.5 * alpha_uv(p[i, j-1] - p[i, j+1]) * dy + (1-alpha_uv) * A_p[i,j]*u_star[i,j]
        source_y[n] = 0.5 * alpha_uv(p[i+1, j] - p[i-1, j]) * dy + (1-alpha_uv) * A_p[i,j]*v_star[i,j]

    ## for bottom wall
    i = Ny
    for j in range(2, Nx):
        n = (Ny - 1)*Nx + (j - 1)
        Fe = dy * u_face[i,j]
        Fw = dy * u_face[i, j-1]
        Fn = dx * v_face[i-1, j]
        Fs = dx * v_face[i,j]

        A_e[i,j] = D_e + max(0.0, -Fe)
        A_w[i,j] = D_w + max(0.0, Fw)
        A_n[i,j] = D_n + max(0.0, -Fn)

        A_p[i,j] = D_e + D_w + D_n + 2 * D_s + max(0.0, Fe) + max(0.0, -Fw) +  max(0.0, Fn) + max(0.0, -Fs)
        source_x[n] = 0.5 * alpha_uv(p[i, j-1] - p[i, j+1]) * dy + (1-alpha_uv) * A_p[i,j]*u_star[i,j]
        source_y[n] = 0.5 * alpha_uv(p[i+1, j] - p[i-1, j]) * dy + (1-alpha_uv) * A_p[i,j]*v_star[i,j]
    
    ## right wall
    j = Nx
    for i in range(2, Ny):
        n = i*Nx - 1
        Fe = dy * u_face[i,j]
        Fw = dy * u_face[i, j-1]
        Fn = dx * v_face[i-1, j]
        Fs = dx * v_face[i,j]

        A_w[i,j] = D_w + max(0.0, Fw)
        A_n[i,j] = D_n + max(0.0, -Fn)
        A_s[i,j] = D_s + max(0.0, Fs)
        A_p[i,j] = 2* D_e + D_w + D_n + D_s + max(0.0, Fe) + max(0.0, -Fw) +  max(0.0, Fn) + max(0.0, -Fs)
        source_x[n] = 0.5 * alpha_uv(p[i, j-1] - p[i, j+1]) * dy + (1-alpha_uv) * A_p[i,j]*u_star[i,j]
        source_y[n] = 0.5 * alpha_uv(p[i+1, j] - p[i-1, j]) * dy + (1-alpha_uv) * A_p[i,j]*v_star[i,j]

    ## top wall
    i = 1
    for j in range(2, Nx):
        n = j - 1
        Fe = dy * u_face[i,j]
        Fw = dy * u_face[i, j-1]
        Fn = dx * v_face[i-1, j]
        Fs = dx * v_face[i,j]

        A_e[i,j] = D_e + max(0.0, -Fe)
        A_w[i,j] = D_w + max(0.0, Fw)
        A_s[i,j] = D_s + max(0.0, Fs)
        A_p[i,j] = D_e + D_w + 2*D_n + D_s + max(0.0, Fe) + max(0.0, -Fw) +  max(0.0, Fn) + max(0.0, -Fs)
        source_x[n] = 0.5 * alpha_uv(p[i, j-1] - p[i, j+1]) * dy + (1-alpha_uv) * A_p[i,j]*u_star[i,j]
        source_y[n] = 0.5 * alpha_uv(p[i+1, j] - p[i-1, j]) * dy + (1-alpha_uv) * A_p[i,j]*v_star[i,j]
    
    ## top left corner
    i = 1, j = 1, n = 0
    Fe = dy * u_face[i,j]
    Fw = dy * u_face[i, j-1]
    Fn = dx * v_face[i-1, j]
    Fs = dx * v_face[i,j]

    A_e[i,j] = D_e + max(0.0, -Fe)
    A_s[i,j] = D_s + max(0.0, Fs)
    A_p[i,j] = D_e + 2 * D_w + 2 * D_n + D_s + max(0.0, Fe) + max(0.0, -Fw) +  max(0.0, Fn) + max(0.0, -Fs)
    source_x[n] = 0.5 * alpha_uv(p[i, j-1] - p[i, j+1]) * dy + (1-alpha_uv) * A_p[i,j]*u_star[i,j] + velocity  * (2 * D_n + max(0.0, -Fn))
    source_y[n] = 0.5 * alpha_uv(p[i+1, j] - p[i-1, j]) * dy + (1-alpha_uv) * A_p[i,j]*v_star[i,j]

    ## top right corner
    i = 1, j = Nx, n = Nx - 1

    Fe = dy * u_face[i,j]
    Fw = dy * u_face[i, j-1]
    Fn = dx * v_face[i-1, j]
    Fs = dx * v_face[i,j]

    A_w[i,j] = D_w + max(0.0, Fw)
    A_s[i,j] = D_s + max(0.0, Fs)
    A_p[i,j] = 2 * D_e + D_w + 2 * D_n + D_s + max(0.0, Fe) + max(0.0, -Fw) +  max(0.0, Fn) + max(0.0, -Fs)
    source_x[n] = 0.5 * alpha_uv(p[i, j-1] - p[i, j+1]) * dy + (1-alpha_uv) * A_p[i,j]*u_star[i,j] + velocity * (2 * D_n + max(0.0, Fn))
    source_y[n] = 0.5 * alpha_uv(p[i+1, j] - p[i-1, j]) * dy + (1-alpha_uv) * A_p[i,j]*v_star[i,j]

    ## Bottom left corner

    i = Ny, j = 1, n = (Ny - 1) * Nx
    Fe = dy * u_face[i,j]
    Fw = dy * u_face[i, j-1]
    Fn = dx * v_face[i-1, j]
    Fs = dx * v_face[i,j]

    A_e[i,j] = D_e + max(0.0, -Fe)
    A_n[i,j] = D_n + max(0.0, -Fn)
    A_p[i,j] = D_e + 2*D_w + D_n + 2 *D_s + max(0.0, Fe) + max(0.0, -Fw) +  max(0.0, Fn) + max(0.0, -Fs)
    source_x[n] = 0.5 * alpha_uv(p[i, j-1] - p[i, j+1]) * dy + (1-alpha_uv) * A_p[i,j]*u_star[i,j]
    source_y[n] = 0.5 * alpha_uv(p[i+1, j] - p[i-1, j]) * dy + (1-alpha_uv) * A_p[i,j]*v_star[i,j]

    ## Bottom right corner
    i = Ny, j = Nx, n = Nx * Ny - 1
    Fe = dy * u_face[i,j]
    Fw = dy * u_face[i, j-1]
    Fn = dx * v_face[i-1, j]
    Fs = dx * v_face[i,j]

    A_w[i,j] = D_w + max(0.0, Fw)
    A_n[i,j] = D_n + max(0.0, -Fn)
    A_p[i,j] = 2 * D_e + D_w + D_n + 2*D_s + max(0.0, Fe) + max(0.0, -Fw) +  max(0.0, Fn) + max(0.0, -Fs)
    source_x[n] = 0.5 * alpha_uv(p[i, j-1] - p[i, j+1]) * dy + (1-alpha_uv) * A_p[i,j]*u_star[i,j]
    source_y[n] = 0.5 * alpha_uv(p[i+1, j] - p[i-1, j]) * dy + (1-alpha_uv) * A_p[i,j]*v_star[i,j]
    A_e *= alpha_uv
    A_w *= alpha_uv
    A_n *= alpha_uv
    A_s *= alpha_uv