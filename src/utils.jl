# Shortname for functions used in cavity update
function gauss(x::Float64)
    return exp( - x^2 / 2) / sqrt( 2 * pi )
end

function mod_erf(x::Float64)
    return ( 1 + erf(x / sqrt(2)) ) / 2
end



"""
    get_index(pdv::PdfDegVec, k::Int)

Returns the index of a given degree in the degree distribution.

Arguments:
- `pdv::PdfDegVec`: Degree distribution structure.
- `k::Int`: Degree.

Returns:
- `idx`: Index of the degree.
"""
function get_index(pdv::PdfDegVec, k::Int)
    return get(pdv.index_dict, k, 0)  # Returns 0 if k is not found
end


function testvalues(sum_mu::Float64, sum_q::Float64, sum_chi::Float64, Epsilon::Float64, Delta::Float64)
    if sum_q < 0 || sum_chi == 1|| !isfinite(sum_mu) || !isfinite(sum_q) || !isfinite(sum_chi) || !isfinite(Epsilon) || !isfinite(Delta)
        println("sum_mu=$(sum_mu), sum_q=$(sum_q), sum_chi=$(sum_chi), Epsilon=$(Epsilon), Delta=$(Delta)")
        throw(ArgumentError("Invalid values"))
    end
end

function testvalues(mu::Float64, q::Float64, chi::Float64, sum_q::Float64, sum_chi::Float64, Delta::Float64)
    if mu < 0 || q < 0 || !isfinite(mu) || !isfinite(q) || !isfinite(chi)
        println("mu=$(mu), q=$(q), chi=$(chi), sum_q=$(sum_q), sum_chi=$(sum_chi), Delta=$(Delta)")
        throw(ArgumentError("Invalid values"))
    end
end


function error_func(check_vars::Dict{String, Float64}, mu_pop::Vector{Float64}, q_pop::Vector{Float64}, chi_pop::Vector{Float64}, tol::Dict{String, Float64})
    # Calculate new averages for convergence checking
    new_avg_mu = mean(mu_pop)

    # Calculate the maximum change in the updates for convergence checking
    max_diff_avg = abs(check_vars["avg_mu"] - new_avg_mu)

    check_vars["avg_mu"] = new_avg_mu

    return max_diff_avg, max_diff_avg < tol["avg"]
end


### Function to solve the fully-connected system (DMFT solution)
# Define the functions
w0_func(x) = mod_erf(x)
w1_func(x) = x * mod_erf(x) + gauss(x)
w2_func(x) = w0_func(x) + x * w1_func(x)


"""
    analytic_FC(m, gamma, npoints, Delta_min, Delta_max)

Compute the analytic solution for the fully-connected system using the Dynamic Mean-Field Theory (DMFT). It returns the values of the parameter `Delta`, and the fixed-point values of the parameters `mu`, `q`, `chi`, and `phi` as a function of `Delta`. It also returns the variance `sigma2`.

# Arguments:
- `m::Float64`: Mean value of the couplings.
- `gamma::Float64`: Correlation of the couplings.
- `npoints::Int`: Number of points to compute.
- `Delta_min::Float64`: Minimum value of the parameter `Delta`.
- `Delta_max::Float64`: Maximum value of the parameter `Delta`.

# Returns:
- `Deltas::Vector{Float64}`: Values of the parameter `Delta`.
- `mus::Vector{Float64}`: Fixed-point values of the mean abundances `mu`.
- `qs::Vector{Float64}`: Fixed-point values of the mean squared abundances `q`.
- `chis::Vector{Float64}`: Fixed-point values of the susceptibilities `chi`.
- `sigma2s::Vector{Float64}`: Variances `sigma2`.
- `phis::Vector{Float64}`: Fixed-point values of the survival probabilities `phi`.
""" 
function analytic_FC(m::Float64, gamma::Float64, npoints::Int, Delta_min::Float64, Delta_max::Float64)
    Deltas = range(Delta_min, Delta_max, length=npoints)
    mus = zeros(npoints)
    qs = zeros(npoints)
    chis = zeros(npoints)
    sigma2s = zeros(npoints)
    phis = zeros(npoints)
    
    for (iD,Delta) in enumerate(Deltas)
        w0 = w0_func(Delta)
        w1 = w1_func(Delta)
        w2 = w2_func(Delta)
    
        chis[iD] = w0 + gamma * w0^2 / w2
        sigma2s[iD] = w2 / (w2 + gamma * w0)^2
        mus[iD] = 1/(Delta / w1 * w2 / (w2 + gamma * w0) - m)
        qs[iD] = (w2 / (w2 + gamma * w0) * mus[iD] / (sqrt(sigma2s[iD]) * w1))^2
        phis[iD] = w0
    end

    return Deltas, mus, qs, chis, sigma2s, phis
end