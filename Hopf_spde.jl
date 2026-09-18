using GLMakie
using LinearAlgebra
using Random
using LaTeXStrings

# Parameters
L = 10.0       # Domain length
Nx = 500       # Total number of grid points including boundaries
T = 50.0       # Final time
dt = 0.0002    # Time step
D = 1.0        # Diffusion coefficient
alpha = 10.0   # Bifurcation parameter
sigma = 1.0    # Noise intensity
N_traj = 5     # Number of trajectories

x = range(0, L, length=Nx)
dx = step(x)

# Number of interior points
N_int = Nx - 2

# Step 1: Set up Crank-Nicolson for Diffusion
diag_A = fill(-2.0, N_int)
offdiag_A = fill(1.0, N_int - 1)
A = Tridiagonal(offdiag_A, diag_A, offdiag_A) .* (1.0 / dx^2)

I_mat = I(N_int)
M_left = I_mat - (D * dt / 4) * A
M_right = I_mat + (D * dt / 4) * A

M_left_fact = factorize(M_left)

# Step 2: SDE Implicit Midpoint Solver
function solve_midpoint!(u, v, u_old, v_old, dt, alpha, dW1, dW2)
    @inbounds for i in 1:N_int
        x_val = u_old[i]
        y_val = v_old[i]
        
        for iter in 1:10
            f1 = alpha*x_val - y_val - x_val*(x_val^2 + y_val^2)
            f2 = alpha*y_val + x_val - y_val*(x_val^2 + y_val^2)
            
            R1 = 2*(x_val - u_old[i]) - f1*dt - dW1
            R2 = 2*(y_val - v_old[i]) - f2*dt - dW2
            
            if abs(R1) < 1e-10 && abs(R2) < 1e-10
                break
            end
            
            J11 = 2 - dt*(alpha - 3*x_val^2 - y_val^2)
            J12 = dt*(1 + 2*x_val*y_val)
            J21 = dt*(1 - 2*x_val*y_val)
            J22 = 2 - dt*(alpha - x_val^2 - 3*y_val^2)
            
            detJ = J11*J22 - J12*J21
            dx_step = -(R1*J22 - R2*J12) / detJ
            dy_step = -(R2*J11 - R1*J21) / detJ
            
            x_val += dx_step
            y_val += dy_step
        end
        
        u[i] = 2*x_val - u_old[i]
        v[i] = 2*y_val - v_old[i]
    end
end

# Step 3: Initial Conditions
U = zeros(Nx, N_traj)
V = zeros(Nx, N_traj)

# Angles \theta spanning [0, 2\pi) across the trajectories
thetas = range(0, 2pi * (1 - 1/N_traj), length=N_traj) # The last angle is below 2\pi

for k in 1:N_traj
    U[2:end-1, k] .= cos(thetas[k]) .* sin.(pi .* x[2:end-1])
    V[2:end-1, k] .= sin(thetas[k]) .* sin.(pi .* x[2:end-1])
end

# Step 4: Visualization Setup
U_obs = Observable(copy(U))
time_obs = Observable(0.0)

fig = Figure(size = (900, 550))

ax = Axis(fig[1, 1], 
          xlabel = L"x", 
          ylabel = L"u_i(t, x)", 
          title = @lift("t = $(round($time_obs, digits=2))"),
          limits = (0, L, -5, 5))

# Generate colors from viridis colormap
cmap = Makie.Categorical(GLMakie.cgrad(:viridis, N_traj, categorical=true))

# Plot initial lines
for k in 1:N_traj
    u_k = @lift($U_obs[:, k])
    lines!(ax, x, u_k, color = cmap[k], linewidth = 1.2)
end

# Step 5: Animation Loop
frames = round(Int, T / dt)

# Pre-allocate interior work vectors
u_int = zeros(N_int)
v_int = zeros(N_int)

noise_scale = sigma * sqrt(dt)

println("Simulating and recording videos...")

record(fig, "with_noise.mp4", 1:frames; framerate = Int(1/dt)) do frame
    dW1 = noise_scale * randn()
    dW2 = noise_scale * randn()
    
    for k in 1:N_traj
        u_int .= U[2:end-1, k]
        v_int .= V[2:end-1, k]
        
        u_half_int = M_left_fact \ (M_right * u_int)
        v_half_int = M_left_fact \ (M_right * v_int)
        
        solve_midpoint!(u_int, v_int, u_half_int, v_half_int, dt, alpha, dW1, dW2)
        
        U[2:end-1, k] .= M_left_fact \ (M_right * u_int)
        V[2:end-1, k] .= M_left_fact \ (M_right * v_int)
    end
    
    # Update observable triggering updates for all lines
    U_obs[] = copy(U)
    time_obs[] = frame * dt
end

println("Animation complete!")