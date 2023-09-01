#=
  Simulation of 1D heat equation with optimization-based control.
  Create plots of input signals and output signals
=#

using DelimitedFiles
path_time= string("results/data/","timegrid","_al",".txt")
path_ref = string("results/data/","reference_fbc","_al",".txt")

path_u_fbc = string("results/data/","input_fbc","_al",".txt")
path_y_fbc = string("results/data/","output_fbc","_al",".txt")

const tgrid = readdlm(path_time, '\t', Float64, '\n')[:,1]
const Tf = tgrid[end]

const ref = readdlm(path_ref, '\t', Float64, '\n')[:,1]
const y_fbc = readdlm(path_y_fbc, '\t', Float64, '\n')[:,1]
const u_fbc = Float64.(readdlm(path_u_fbc, '\t', BigFloat, '\n')[:,1])


# Optimization-based control: parametrized input function
function input_signal_oc(time,p₁,p₂,p₃)
  return  exp(p₁- p₃^2 * (time - 1/p₂)^2)
end

# Continuous input signal for flatness-based control
function input_signal(t,dt)
  if t <= 0
      return u_fbc[1]
  elseif t >= Tf
      return u_fbc[end]
  end
  τ = t/dt + 1
  t0 = floor(Int, τ)
  t1 = t0 + 1;

  u0 = u_fbc[t0]
  u1 = u_fbc[t1]

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
  time = t/Tf;
  u = input_signal_oc(time, p...)
  
  diffusion_x!(dx,x,Nx,Δx)

  dx .= α * dx
  dx[1] = dx[1] + 2α/(λ * Δx) * (u + nonlinear_bc(x[1]))
  dx[end] = dx[end] + 2α/(λ * Δx) * nonlinear_bc(x[end])
end


# 1D heat equation with flatness-based control
function heat_eq_fbc!(dx,x,p,t)       
  # time = t/Tf;
  #u = input_signal(time, p)
  
  ts = 1
  u = max(input_signal(t,ts),0)
  
  diffusion_x!(dx,x,Nx,Δx)

  dx .= α * dx
  dx[1] = dx[1] + 2α/(λ * Δx) * (u + nonlinear_bc(x[1]))
  dx[end] = dx[end] + 2α/(λ * Δx) * nonlinear_bc(x[end])
end


L = 0.2; # Length of 1D rod

# Material
λ = 237;  # Thermal conductivity
ρ = 2700; # Density
c = 900;  # Specific heat capacity
α = λ / (ρ * c) # Diffusivity


# Discretization  
const Nx = 101;     # Number of elements x-direction
const Δx = L/(Nx-1) # Spatial sampling

# Simulation
using OrdinaryDiffEq

θ0 = 300
x0 = θ0 * ones(Nx) # Intial values
tspan =  (0.0, Tf)   # Time span

alg = KenCarp4()    # Numerical integrator

p_approx = [13.556764614099377, 2.262443438914027, 13.557062799838445]  # Parameters of approximation of FBC

p3_find(p1,p2,q) = p2*sqrt(p1-log(q))
p_opt_0 = [13.7964, 2.28296, p3_find(13.7964, 2.28296, 1e-18)]  # optimization-based control; output temperature tracking: μ=0 
p_opt_1 = [13.6339, 1.98827, p3_find(13.6339, 1.98827, 1e-18)]  # optimization-based control; mean temperature tracking: μ=1

prob_oc = ODEProblem(heat_eq_oc!,x0,tspan)
sol_approx = solve(prob_oc,alg, p=p_approx, saveat = tgrid) # Approximation
sol_opt_0 = solve(prob_oc,alg, p=p_opt_0, saveat = tgrid)   # Output temperature tracking
sol_opt_1 = solve(prob_oc,alg, p=p_opt_1, saveat = tgrid)   # Mean temperature tracking
θmean = sum(sol_opt_1, dims=1)[:]/Nx    # Mean temperature

prob_fbc = ODEProblem(heat_eq_fbc!,x0,tspan)
sol_fbc = solve(prob_fbc,alg, saveat = tgrid)   # Flatness-based control

u_approx = input_signal_oc.(tgrid/Tf,p_approx...) # Input signal: approximation
u_opt_0 = input_signal_oc.(tgrid/Tf,p_opt_0...)   # Input signal: output temperature tracking
u_opt_1 = input_signal_oc.(tgrid/Tf,p_opt_1...)   # Input signal: mean temperature tracking


using CairoMakie
t_ref = tgrid[1:50:end] # time grid for reference
begin
  fig1 = Figure(fontsize=20)
  ax1 = Axis(fig1[1, 1], xlabel =L"Time $t$ in $[s]$", ylabel = L"Temperature in $[K]$", ylabelsize = 24,
      xlabelsize = 24, xgridstyle = :dash, ygridstyle = :dash, 
      xtickalign = 1., xticksize = 10, 
      xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
      yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
      ytickalign = 1, yticksize = 10, xlabelpadding = 0)
  
  ax1.xticks = 0 : 100 : Tf;    
  #ax1.yticks = 0 : 0.2 : 1.0;
  lines!(tgrid, sol_fbc[end,:];   linestyle = :dot,     linewidth = 4, label = "Flatness-b. Control")
  lines!(tgrid, sol_approx[end,:];linestyle = :dashdot, linewidth = 4, label = "Approximation")
  lines!(tgrid, sol_opt_0[end,:];                       linewidth = 3, label = L"OC, $\mu=0$: $y=\theta(x=L)$")
  lines!(tgrid, θmean;            linestyle = :dash,    linewidth = 4, label = L"OC, $\mu=1$: $y=\overline{\vartheta}$")
  scatter!(t_ref, ref[1:50:end]; markersize = 15, marker = :diamond, color=:black, label = "Reference")
  axislegend(; position = :lt, bgcolor = (:grey90, 0.1));

  ax2 = Axis(fig1, bbox=BBox(618, 759, 369, 510), ylabelsize = 24)
  ax2.xticks = 800 : 100 : Tf;
  #ax2.yticks = -0.01 : 0.002 : 0.01;
  lines!(ax2, tgrid[800:end],sol_fbc[end,800:end];    linestyle = :dot,     linewidth = 5, color=Makie.wong_colors()[1])
  lines!(ax2, tgrid[800:end],sol_approx[end,800:end]; linestyle = :dashdot, linewidth = 5, color=Makie.wong_colors()[2])
  lines!(ax2, tgrid[800:end],sol_opt_0[end,800:end];                        linewidth = 4, color=Makie.wong_colors()[3])
  lines!(ax2, tgrid[800:end],θmean[800:end];          linestyle = :dash,    linewidth = 5, color=Makie.wong_colors()[4])
  translate!(ax2.scene, 0, 0, 10);

  fig1
  save("results/figures/"*"sim_output_signals.pdf", fig1, pt_per_unit = 1)    
end

begin
  fig1 = Figure(fontsize=20)
  ax1 = Axis(fig1[1, 1], xlabel = L"Time $t$ in $[s]$", ylabel = L"Input Signal $\times 10^{5}$", ylabelsize = 24,
      xlabelsize = 24, xgridstyle = :dash, ygridstyle = :dash, 
      xtickalign = 1., xticksize = 10, 
      xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
      yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
      ytickalign = 1, yticksize = 10, xlabelpadding = 0)
  
  ax1.xticks = 0 : 100 : Tf;    
  ax1.yticks = 0 : 2 : 10;
  lines!(tgrid, u_fbc*1e-5;   linestyle = :dot,     linewidth = 4, label = "Flatness-b. Control")
  lines!(tgrid, u_approx*1e-5;linestyle = :dashdot, linewidth = 4, label = "Approximation")
  lines!(tgrid, u_opt_0*1e-5;                       linewidth = 3, label = L"OC, $\mu=0$")
  lines!(tgrid, u_opt_1*1e-5; linestyle = :dash,    linewidth = 4, label = L"OC, $\mu=1$")
  axislegend(; position = :lt, bgcolor = (:grey90, 0.1));
  fig1
  save("results/figures/"*"sim_input_signals.pdf", fig1, pt_per_unit = 1)    
end



# Parameter fine-tuning
p1_arr = p_opt_0[1] * [0.995, 0.99, 0.95, 0.9]
p3_arr = [15.837158839196446, 14.781508946024806, 8.512374573838958, 4.2507541000686455]

sol_red_995 = solve(prob_oc,alg, p=[p1_arr[1],p_opt_0[2],p3_arr[1]], saveat = tgrid)
sol_red_990 = solve(prob_oc,alg, p=[p1_arr[2],p_opt_0[2],p3_arr[2]], saveat = tgrid)
sol_red_950 = solve(prob_oc,alg, p=[p1_arr[3],p_opt_0[2],p3_arr[3]], saveat = tgrid)
sol_red_900 = solve(prob_oc,alg, p=[p1_arr[4],p_opt_0[2],p3_arr[4]], saveat = tgrid)


u_995 = input_signal_oc.(tgrid/Tf,p1_arr[1],p_opt_0[2],p3_arr[1])
u_990 = input_signal_oc.(tgrid/Tf,p1_arr[2],p_opt_0[2],p3_arr[2])
u_950 = input_signal_oc.(tgrid/Tf,p1_arr[3],p_opt_0[2],p3_arr[3])
u_900 = input_signal_oc.(tgrid/Tf,p1_arr[4],p_opt_0[2],p3_arr[4])


begin
  fig1 = Figure(fontsize=20)
  ax1 = Axis(fig1[1, 1], xlabel =L"Time $t$ in $[s]$", ylabel = L"Temperature in $[K]$", ylabelsize = 24,
      xlabelsize = 24, xgridstyle = :dash, ygridstyle = :dash, 
      xtickalign = 1., xticksize = 10, 
      xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
      yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
      ytickalign = 1, yticksize = 10, xlabelpadding = 0)
  
  ax1.xticks = 0 : 100 : Tf;    
  #ax1.yticks = 0 : 0.2 : 1.0;
  lines!(tgrid, sol_opt_0[end,:];                           linewidth = 3, label = L"OC, $\mu=0$: $y=\theta(x=L)$")
  lines!(tgrid, sol_red_995[end,:]; linestyle = :dash,      linewidth = 4, label = L"99.5% of $p_{1}$")
  lines!(tgrid, sol_red_990[end,:]; linestyle = :dashdot,   linewidth = 4, label = L"99% of $p_{1}$")
  lines!(tgrid, sol_red_950[end,:]; linestyle = :dashdotdot,linewidth = 4, label = L"95% of $p_{1}$")
  lines!(tgrid, sol_red_900[end,:]; linestyle = :dot,       linewidth = 4, label = L"90% of $p_{1}$")
  axislegend(; position = :lt, bgcolor = (:grey90, 0.1));

  
  ax2 = Axis(fig1, bbox=BBox(618, 759, 314, 474), ylabelsize = 24)
  ax2.xticks = 900 : 50 : Tf;
  #ax2.yticks = -0.01 : 0.002 : 0.01;
  lines!(ax2, tgrid[900:end],sol_opt_0[end,900:end];                            linewidth = 4, color=Makie.wong_colors()[1])
  lines!(ax2, tgrid[900:end],sol_red_995[end,900:end]; linestyle = :dash,       linewidth = 5, color=Makie.wong_colors()[2])
  lines!(ax2, tgrid[900:end],sol_red_990[end,900:end]; linestyle = :dashdot,    linewidth = 5, color=Makie.wong_colors()[3])
  lines!(ax2, tgrid[900:end],sol_red_950[end,900:end]; linestyle = :dashdotdot, linewidth = 5, color=Makie.wong_colors()[4])
  lines!(ax2, tgrid[900:end],sol_red_900[end,900:end];  linestyle = :dot,       linewidth = 5, color=Makie.wong_colors()[5])
  translate!(ax2.scene, 0, 0, 10);
  
  fig1
  save("results/figures/"*"energy_output.pdf", fig1, pt_per_unit = 1)    
end

begin
  fig1 = Figure(fontsize=20)
  ax1 = Axis(fig1[1, 1], xlabel = L"Time $t$ in $[s]$", ylabel = L"Input Signal $\times 10^{5}$", ylabelsize = 24,
      xlabelsize = 24, xgridstyle = :dash, ygridstyle = :dash, 
      xtickalign = 1., xticksize = 10, 
      xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
      yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
      ytickalign = 1, yticksize = 10, xlabelpadding = 0)
  
  ax1.xticks = 0 : 100 : Tf;    
  ax1.yticks = 0 : 2 : 10;
  lines!(tgrid, u_opt_0*1e-5;                       linewidth = 3, label = L"OC, $\mu=0$")
  lines!(tgrid, u_995*1e-5; linestyle = :dash,      linewidth = 4, label = L"99.5% of $p_{1}$")
  lines!(tgrid, u_990*1e-5; linestyle = :dashdot,   linewidth = 4, label = L"99% of $p_{1}$")
  lines!(tgrid, u_950*1e-5; linestyle = :dashdotdot,linewidth = 4, label = L"95% of $p_{1}$")
  lines!(tgrid, u_900*1e-5; linestyle = :dot,       linewidth = 4, label = L"90% of $p_{1}$")
  axislegend(; position = :lt, bgcolor = (:grey90, 0.1));
  fig1
  save("results/figures/"*"energy_input.pdf", fig1, pt_per_unit = 1)    
end