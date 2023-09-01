#=
    Optimization-based control computation for 1D heat equation with nonlinear boundary conditions.
=#


L = 0.2; # Length of 1D rod

# Material
λ = 237;  # Thermal conductivity
ρ = 2700; # Density
c = 900;  # Specific heat capacity
α = λ / (ρ * c) # Diffusivity


using DelimitedFiles
path_time= string("results/data/","timegrid","_al",".txt")
path_ref = string("results/data/","reference_fbc","_al",".txt")

const tgrid = readdlm(path_time, '\t', Float64, '\n')[:,1]
const ref = readdlm(path_ref, '\t', Float64, '\n')[:,1]

# Parametrized input function
function input_signal_oc(time,p₁,p₂,p₃)
    return  exp(p₁- p₃^2 * (time - 1/p₂)^2)
end


# Diffusion: x-direction
function diffusion_x!(dx,x,Nx, Δx) # in-place
    
    for i in 2 : Nx-1
        dx[i] =  (x[i-1] - 2*x[i] + x[i+1])/Δx^2
    end
    i1 = 1      # West
    i2 = Nx     # East
    dx[i1] = (-2*x[i1] + 2*x[i1+1])/Δx^2 # Neumann BC West
    dx[i2] = (2*x[i2-1] - 2*x[i2])/Δx^2  # Neumann BC East
    
    nothing 
end


# Nonlinear boundary condition: Stefan-Boltzmann
function nonlinear_bc(θ)
    h = 20      # heat transfer coefficient
    ϵ = 0.5     # emissivity
    sb = 5.670374419 * 1e-8; # Stefan-Boltzmann constant
    k = ϵ*sb    # heat radiation coefficient
    θamb = 300  # ambient temperature
    return -h*(θ - θamb) - k*(θ^4 - θamb^4)
end


# 1D heat equation with optimal control
function heat_eq_oc!(dx,x,p,t)          
    diffusion_x!(dx,x,Nx,Δx)
    u_in = input_signal_oc(t/Tf, p[1],p[2],p[3])
    dx .= α * dx
    dx[1] = dx[1] + 2α/(λ * Δx) * (u_in + nonlinear_bc(x[1]))
    dx[end] = dx[end] + 2α/(λ * Δx) * nonlinear_bc(x[end])
end


# Simulation without optimization-based control signal
using OrdinaryDiffEq, ModelingToolkit

# Discretization  
const Nx = 101;     # Number of elements x-direction
const Δx = L/(Nx-1) # Spatial sampling

x0    =  300* ones(Nx) # Intial values
Tf    = tgrid[end];
tspan =  (0.0, Tf)   # Time span
p_orig= [13.0,2.2,200.0]

alg = KenCarp4()    # Numerical integrator
prob_orig = ODEProblem(heat_eq_oc!,x0,tspan,p_orig)
prob_mtk = modelingtoolkitize(prob_orig);
prob_he = ODEProblem(prob_mtk, [], tspan, jac = true);
# sol_orig = solve(prob_he, alg, saveat = tgrid)

ref_opt = ref # Reference
function loss_total(p,q,μ)
    c = p[2]*sqrt(p[1]-log(q))
    sol_temp = solve(prob_he, alg, p=[p[1],p[2],c], saveat = tgrid)
    
    if sol_temp.retcode != ReturnCode.Success
        return Inf
    end

    err_output = ref_opt-sol_temp[end,:]
    loss_output = sum(abs2, err_output)/length(err_output)
  
    θmean = sum(sol_temp, dims=1)[:]/Nx
    err_mean = ref_opt-θmean
    loss_mean = sum(abs2, err_mean)/length(err_mean)

    loss = (1-μ)*loss_output + μ*loss_mean

    return loss #, sol
end

callback = function (p, l)
    display(l)
    return false
end

#= Parameters from Approximation
p = [13.556764614099377
      2.262443438914027
     13.557062799838445]
=#

pinit = [13.6, 2.3] 
q = 1e-18

# loss_total(pinit,q,0.1) # Loss for μ=0.1 

using Optimization, SciMLSensitivity 
using ForwardDiff
using OptimizationOptimJL

adtype = Optimization.AutoForwardDiff();
optf = Optimization.OptimizationFunction((x, p) -> loss_total(x,p[1],p[2]), adtype);

μ_vec = collect(0:0.1:1);
p_store = zeros(2,length(μ_vec))

for (i, el) in enumerate(μ_vec)
    optprob = Optimization.OptimizationProblem(optf, pinit,[q,el], lb=[0.0, 1.0], ub=[Inf, Inf]);
    opt_pars = Optimization.solve(optprob, BFGS(), callback = callback, maxiters = 20); 
    p_store[:,i] = opt_pars
end

#=
p_store = 
[13.7964   13.7764   13.757   13.7383   13.7204   13.7035   13.6877   13.6729   13.659    13.646    13.6339
  2.28296   2.24899   2.2148   2.18095   2.14807   2.11659   2.08689   2.05954   2.03363   2.00996   1.98827]
=#