#=
    Parameter fine-tuning: choose new p₁ and search for new p₃; p₂ is fixed
=#


using SpecialFunctions

T=1000; # Final simulation time
u_int(p₂,p₃) = T*sqrt(pi) * (erf(p₃-p₃/p₂) - erf(-p₃/p₂)) / (2 * p₃)

# Loss function to search for p3
function loss_energy_p3(u,p)
    p_orig = [13.7964, 2.28296, 16.9682]    # Parameters from output tracking; μ=0
    E_orig = exp(p_orig[1]) * u_int(p_orig[2], p_orig[3])
    E_new = exp(p[1]) * u_int(p[2], u[1])
    return (E_orig-E_new)^2
end

using Optimization, OptimizationOptimJL
p0 = [13.7964, 2.28296, 16.9682]            # Parameters from output tracking; μ=0
p1_arr = p0[1] * [0.995, 0.99, 0.95, 0.9]
p3_arr = similar(p1_arr)

optf = OptimizationFunction(loss_energy_p3, Optimization.AutoForwardDiff())
opt_u0 = [round(p0[3], digits=2)]
for (i, p1) in enumerate(p1_arr)
    opt_prob = OptimizationProblem(optf, opt_u0, [p1,p0[2]], lb=[0.0], ub=[Inf])
    p3_arr[i] = solve(opt_prob, BFGS())[1]
end

#=
p3_arr = [15.837158839196446, 14.781508946024806, 8.512374573838958, 4.2507541000686455]
=#



