#=
Computation of derivatives Ω^{(n)}(t) dependend on final time T
=#

using BellBruno
N_der = 20;              # Number of derivatives

# Either create new Bell polynomials  ones
bp = bell_poly(N_der);   # Create bell polynomials

# Or load already existing Bell polynomial numbers
# bp_load = BellBruno.read_bell_poly()
# bp = bp_load[1:N_der]

# Compute Bell coefficients
bc = bell_coeff(bp);     # Compute bell coefficients

#=
    f(z) = -1 z^(-2)
    d^n/dz^n f(z) = -1 * z^(-2-n) π(-2-j) for j=0:1:n
=#
function outer_fun!(y, z, p) 
    c = -1;

    y[1] = simple_monomial(z, c, p)

    fi = firstindex(y)
    li = lastindex(y)

    for idx in fi:li-1
        y[idx+1] = simple_monomial_der(z, c, p, idx)
    end
end

#=
    g(t)  = t/T - (t/T)^2
    g'(t) = 1/T - 2*t/T^2
    g''(t) = -2/T^2
    g^(n)(t) = 0 for n>2
=#
function inner_fun!(x, t :: Float64; T = 1.0 :: Real)
    c₁ = 0;
    c₂ = 1/T;
    c₃ = -1/T^2;

    x[1] = c₁ + c₂*t + c₃*t^2;  # g(t)  = t/T - (t/T)^2
    x[2] = c₂ + 2*c₃*t;         # g'(t) = 1/T - (2/T^2)*t
    x[3] = 2*c₃;                # g''(t)= -2/T^2
    x[4:end] .= 0;              # g^(n)(t) = 0 for n>2
end



function build_derivative(n_der, bc, bp, data_inner, data_outer, tgrid)
    nt = size(data_inner)[1]
    res = zeros(nt)

    data_out_is_vector = false;

    if length(data_outer[1,:]) == 1
        data_out_is_vector = true
    end

    for k=1 : n_der
        fi = firstindex(bp[n_der+1][k][1,:])           
        li = lastindex(bp[n_der+1][k][1,:])
        sol_prod = zeros(BigFloat,nt)   # Solution of the product π
        for μ = fi : li
                
            sol_prod_temp = zeros(BigFloat,nt)
                
            a = bc[n_der+1][k][μ]   # Coefficients
                
            for (idx, _) in enumerate(tgrid)
                @views x = data_inner[idx,:]
                sol_prod_temp[idx] = a * mapreduce(^, *, x, bp[n_der+1][k][:,μ])
            end
            sol_prod = sol_prod + sol_prod_temp
        end

        if data_out_is_vector == true
            res = res + data_outer.*sol_prod
        else
            res = res + data_outer[:,k+1].*sol_prod
        end
    end

    return res
end

function compute_derivatives(bc, bp, T, dt; w=2, n_start=1, n_stop=length(bp)-1)
    tgrid = dt : dt : T-dt; # Time grid
    nt = length(tgrid)      # Number of time steps
    
    # Outer derivatives
    g̃ = zeros(nt, n_stop+1); # g̃_n(t) := d^n/dt^n g(t)
    f̃ = zeros(nt, n_stop+1); # f̃_n(t) := d^n/dy^n f(z)
    
    for (idx, elem) in enumerate(tgrid)
        @views x = g̃[idx,:]
        @views y = f̃[idx,:]
        inner_fun!(x, elem, T=T)
        outer_fun!(y, x[1], -1*w) 
    end

    num_der = n_stop-n_start+1
    
    q = zeros(nt, num_der);
    h = zeros(BigFloat,nt, num_der+1);
    h[:,1] = exp.(big.(f̃[:,1]))
    
    for n = n_start : n_stop
        i = n - n_start+1
        q[:,i] = build_derivative(n, bc, bp, g̃[:,2:end], f̃, tgrid)
        h[:,i+1] = build_derivative(n, bc, bp, q, h[:,1], tgrid)
        println("Iteration n= ", n)
    end

    return h
end

function compute_derivatives(n_der, bc, bp, T, dt; w=2)
    tgrid = dt : dt : T-dt; # Time grid
    nt = length(tgrid)      # Number of time steps
    
    # Outer derivatives
    g̃ = zeros(nt, n_der+1); # g̃_n(t) := d^n/dt^n g(t)
    f̃ = zeros(nt, n_der+1); # f̃_n(t) := d^n/dy^n f(z)
    
    for (idx, elem) in enumerate(tgrid)
        @views x = g̃[idx,:]
        @views y = f̃[idx,:]
        inner_fun!(x, elem, T=T)
        outer_fun!(y, x[1], -w) 
    end
    
    q = zeros(nt, n_der);
    h = zeros(BigFloat,nt, n_der+1);
    h[:,1] = exp.(big.(f̃[:,1]))
     
    for n = 1 : n_der
        q[:,n] = build_derivative(n, bc, bp, g̃[:,2:end], f̃, tgrid)
        h[:,n+1] = build_derivative(n, bc, bp, q, h[:,1], tgrid)
        println("Iteration n= ", n)
    end

    return h
end

w = 2.0
T = 1000.0;    # Final simulation time
N_dt = 1000;
dt = T/N_dt; # Sampling time
h = compute_derivatives(N_der, bc, bp, T, dt, w=w)


using DelimitedFiles
path_h = string("results/h_results/h_results_T_", round(Int,T), "_w_", round(Int,w*10), ".txt")

open(path_h, "w") do io
    writedlm(io, h)
end


