#=
    Apply flatness-based control on heat equation
=#

L = 0.2; # Length of 1D rod

# Material
λ = 237;  # Thermal conductivity
ρ = 2700; # Density
c = 900;  # Specific heat capacity
α = λ / (ρ * c) # Diffusivity
γ = L^2 / α



# Construct flatness-based control signal 
η(L,α,i) = BigFloat(L)^(2i+1) / (BigFloat(α)^(i+1) * factorial(big(2i+1)))
idx_grid = 0:20;
eta = zeros(BigFloat, length(idx_grid))

for (n, iter) in enumerate(idx_grid)
    eta[n] = η(L,α,iter)
end

w =  2.0;
using FastGaussQuadrature
const Tf = 1000.0; # Final simulation time
bump(t) = exp(-1 / (t/Tf - (t/Tf)^2)^w)
t_gq, weights_gq = FastGaussQuadrature.gausslegendre(1000)
p = Tf/2;
Ω_int = p *FastGaussQuadrature.dot( weights_gq ,bump.(p*t_gq .+ p))

diff_ref = 200; # (y_f - y_0) = 200 Kelvin


function reference_signal(t)
    if t <= 0
        return 0
    elseif t>= Tf
        return 1
    end

    p1 = t/2
    numer = p1 *FastGaussQuadrature.dot( weights_gq ,bump.(p1*t_gq .+ p1))

    return numer / Ω_int
end


using DelimitedFiles
path_2_file = string("results/h_results/h_results_T_", round(Int,Tf), "_w_", round(Int,w*10), ".txt")
h_data = readdlm(path_2_file, '\t', BigFloat, '\n')

# Simulation
u_data = λ*(diff_ref/Ω_int) * (h_data * eta)
u_data = vcat(0, u_data, 0)

# Flatness-based input signal
function input_signal(t,dt)
    if t <= 0
        return u_data[1]
    elseif t >= Tf
        return u_data[end]
    end
    τ = t/dt + 1
    t0 = floor(Int, τ)
    t1 = t0 + 1;

    u0 = u_data[t0]
    u1 = u_data[t1]

    a = u1-u0;
    b = u0 - a*t0

    return a*τ + b;
end


# Diffusion: 1-dimensional
function diffusion_x!(dx,x,Nx, Δx) # in-place
    
    for i in 2 : Nx-1
      dx[i] =  (x[i-1] - 2*x[i] + x[i+1])/Δx^2
    end
    i1 = 1      # West
    i2 = Nx     # East
    dx[i1] = (-2*x[i1] + 2*x[i1+1])/Δx^2
    dx[i2] = (2*x[i2-1] - 2*x[i2])/Δx^2
    nothing 
end


# 1D heat equation with zero Neumann boundary conditions
function heat_eq!(dx,x,p,t)       
    u = max(input_signal(t,ts),0)   # Restriction for positive input signals
    
    diffusion_x!(dx,x,Nx,Δx)
  
    dx .= α * dx
    dx[1] = dx[1] + 2α/(λ * Δx) * u
end


# Discretization  
const Nx = 101;     # Number of elements x-direction
const Δx = L/(Nx-1) # Spatial sampling
const ts = Tf / (length(u_data)-1) # 1.0;     # Time step width

# Simulation of heat equation with flatness-based input signals
using OrdinaryDiffEq

θ0 = 300;
x0 = θ0 * ones(Nx) # Intial values
tspan =  (0.0, Tf)   # Time span

prob = ODEProblem(heat_eq!,x0,tspan)
alg = KenCarp4()    # Numerical integrator
sol = solve(prob,alg, saveat = ts)

tgrid = sol.t;
u_in(t) = max(input_signal(t,ts),0)
u_in_data = u_in.(tgrid)


using DelimitedFiles
data_ref = θ0 .+ diff_ref*reference_signal.(tgrid)
data_yout = sol[end,:]
data_uin = u_in_data
data_final_temp = sol[:,end]

path_time= string("results/data/","timegrid","_al",".txt")
path_ref = string("results/data/","reference_fbc","_al",".txt")
path_uin = string("results/data/","input_fbc","_al",".txt")
path_yout = string("results/data/","output_fbc","_al",".txt")


open(path_time, "w") do io
    writedlm(io, tgrid)
end

open(path_ref, "w") do io
    writedlm(io, data_ref)
end

open(path_uin, "w") do io
    writedlm(io, data_uin)
end

open(path_yout, "w") do io
    writedlm(io, data_yout)
end

