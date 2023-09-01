#=
    Approximate optimal control signal from flatness-based control
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
path_y_fbc = string("results/data/","output_fbc","_al",".txt")
path_u_fbc = string("results/data/","input_fbc","_al",".txt")

const tgrid = readdlm(path_time, '\t', Float64, '\n')[:,1]
const Tf = tgrid[end]

const ref = readdlm(path_ref, '\t', Float64, '\n')[:,1]
const y_fbc = readdlm(path_y_fbc, '\t', Float64, '\n')[:,1]
const u_fbc = readdlm(path_u_fbc, '\t', BigFloat, '\n')[:,1]


# Parametrized input function
function input_signal_oc(time,p₁,p₂,p₃)
    return  exp(p₁- p₃^2 * (time - 1/p₂)^2)
end


# u - Optimization values
# p - addtional/known parameters
function loss_optim(u,p)
    input_oc(t) = input_signal_oc(t/Tf,p[1], p[2], u[1])
    input_error = u_fbc-input_oc.(tgrid)
    err = sum(abs2, input_error)/sum(abs2,input_fbc_data)
    return err
end

# Parameters p1, p2 can be found directly
input_fbc_data = convert.(Float64,u_fbc)
u_fbc_max = maximum(input_fbc_data)
t_max = tgrid[argmax(input_fbc_data)]

p1_fix = log(u_fbc_max)
p2_fix = Tf/t_max


# Find optimal p3
using Optimization, OptimizationOptimJL
opt_p = [p1_fix, p2_fix]
opt_u0 = [1.0]
loss_optim([140],[p1_fix, p2_fix])

optf = OptimizationFunction(loss_optim, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(optf, opt_u0, opt_p)
p3_opt = solve(opt_prob, BFGS())[1]

pars_all = [p1_fix, p2_fix, p3_opt]
#=
p = [13.556764614099377
      2.262443438914027
     13.557062799838445]
=#


loss_optim([p3_opt],[p1_fix, p2_fix])

