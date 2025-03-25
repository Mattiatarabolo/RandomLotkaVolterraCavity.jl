

# Function to compute all the quantities obtain by the sum of neighbours' terms in the cavity update
function sumpop(mu_pop::Vector{Float64}, q_pop::Vector{Float64}, chi_pop::Vector{Float64}, J_pop::Vector{Float64}, Jp_pop::Vector{Float64}, neigh_idxs::Vector{Int}, k::Int)
    sum_mu = 0.0
    sum_q = 0.0
    sum_chi = 0.0
    @inbounds @simd for neigh_idx in 1:k
        j = neigh_idxs[neigh_idx]
        sum_mu += J_pop[j] * mu_pop[j]
        sum_q += J_pop[j]^2 * q_pop[j]
        sum_chi += J_pop[j] * Jp_pop[j] * chi_pop[j]
    end
    Eps = sqrt(sum_q)
    Delta = (1+sum_mu)/Eps

    testvalues(sum_mu, sum_q, sum_chi, Eps, Delta)
    return sum_q, sum_chi, Eps, Delta
end
   

# Placeholder update functions, where you can optimize the internal logic of f_mu, f_q, f_chi
function f_mu(sum_chi::Float64, Eps::Float64, Delta::Float64)
    if sum_chi < 1.0
        return Eps / (1 - sum_chi) * (Delta * mod_erf(Delta) + gauss(Delta))
    else
        return Eps / (1 - sum_chi) * (Delta * mod_erf(-Delta) - gauss(-Delta))
    end
end

function f_q(sum_q::Float64, sum_chi::Float64, Delta::Float64)
    if sum_chi < 1.0
        return sum_q / (1 - sum_chi)^2 * ((1+Delta^2) * mod_erf(Delta) + Delta * gauss(Delta))
    else
        return sum_q / (1 - sum_chi)^2 * ((1+Delta^2) * mod_erf(-Delta) - Delta * gauss(-Delta))
    end
end

function f_chi(sum_chi::Float64, Delta::Float64)
    if sum_chi < 1.0
        return 1 / (1 - sum_chi) * mod_erf(Delta)
    else
        return 1 / (1 - sum_chi) * mod_erf(-Delta)
    end
end


####################################################################################################################################################
########################################################## Sparse graph ###########################################################################
####################################################################################################################################################

function update_cav!(p_cav_k::PdfDegVec, P::Int, m::Float64, sigma2::Float64, gamma::Float64, K::Union{Int,Float64}, rng::AbstractRNG, mu_cav_pop::Vector{Float64}, q_cav_pop::Vector{Float64}, chi_cav_pop::Vector{Float64}, J_pop::Vector{Float64}, Jp_pop::Vector{Float64}, neigh_idxs::Vector{Int}, damp::Float64)
    # Update each site in the population
    i = rand(rng, 1:P)
    # Sample the degree k
    k_cav = sample_degree(rng, p_cav_k)

    if k_cav == 0
        mu_cav_pop[i] = 1.0
        q_cav_pop[i] = 1.0
        chi_cav_pop[i] = 0.0
        return
    end

    # Get k random neighbors
    sample_neighs!(rng, neigh_idxs, i, k_cav, P)
    
    # CAVITY UPDATE
    # Sample k_cav pairs of correlated J, J' values
        @inbounds @simd for j in neigh_idxs[1:end-1]
        J_pop[j], Jp_pop[j] = sample_couplings(rng, m, sigma2, gamma, K)
    end

    ## We could sample a bigger population of J at the beginning and sample from it ##

    # Compute the new values for mu, q, chi using the update functions
    sum_q_cav, sum_chi_cav, Eps, Delta = sumpop(mu_cav_pop, q_cav_pop, chi_cav_pop, J_pop, Jp_pop, neigh_idxs, k_cav)
    mu_cav_pop[i] = (1-damp)*f_mu(sum_chi_cav, Eps, Delta) + damp*mu_cav_pop[i]
    q_cav_pop[i] = (1-damp)*f_q(sum_q_cav, sum_chi_cav, Delta) + damp*q_cav_pop[i]
    chi_cav_pop[i] = (1-damp)*f_chi(sum_chi_cav, Delta) + damp*chi_cav_pop[i]
    testvalues(mu_cav_pop[i], q_cav_pop[i], chi_cav_pop[i], sum_q_cav, sum_chi_cav, Delta)
end

function update_full!(p_k::PdfDegVec, P::Int, m::Float64, sigma2::Float64, gamma::Float64, K::Union{Int,Float64}, rng::AbstractRNG, mu_cav_pop::Vector{Float64}, q_cav_pop::Vector{Float64}, chi_cav_pop::Vector{Float64}, J_pop::Vector{Float64}, Jp_pop::Vector{Float64}, mu_pop::Vector{Float64}, q_pop::Vector{Float64}, chi_pop::Vector{Float64}, neigh_idxs::Vector{Int})
    # Update each site in the population
    @inbounds for i in 1:P
        k_full = sample_degree(rng, p_k)

        if k_full == 0
            mu_pop[i] = 1.0
            q_pop[i] = 1.0
            chi_pop[i] = 0.0
            continue
        end

        sample_neighs!(rng, neigh_idxs, i, k_full, P)

        # FULL UPDATE
        # Sample k_cav pairs of correlated J, J' values
        @inbounds @simd for j in neigh_idxs
            J_pop[j], Jp_pop[j] = sample_couplings(rng, m, sigma2, gamma, K)
        end
        # Compute the new values for mu, q, chi using the update functions
        sum_q_full, sum_chi_full, Eps_full, Delta_full = sumpop(mu_cav_pop, q_cav_pop, chi_cav_pop, J_pop, Jp_pop, neigh_idxs, k_full)
        mu_pop[i] = f_mu(sum_chi_full, Eps_full, Delta_full)
        q_pop[i] = f_q(sum_q_full, sum_chi_full, Delta_full)
        chi_pop[i] = f_chi(sum_chi_full, Delta_full)
        testvalues(mu_pop[i], q_pop[i], chi_pop[i], sum_q_full, sum_chi_full, Delta_full) 
    end
end

# Function to run population dynamics with all the performance tips included
"""
    population_dynamics(
        m::Float64, 
        sigma2::Float64, 
        gamma::Float64, 
        p_k::PdfDegVec, 
        p_cav_k::PdfDegVec,
        P::Int,
        tol,
        max_iter::Int;
        check_vars::Dict{String, Float64} = Dict("avg_mu"=>0.0), 
        error_func::Function = error_func, 
        check_conv::Int = 30, 
        rng::AbstractRNG = Xoshiro(1234), 
        verbose::Bool = false,
        plothist::Bool = false,
        damp::Float64 = 0.0
    )

Run the population dynamics algorithm for the fixed point abundances of theRandom Lotka Volterra model on a sparse graph.

# Arguments
- `m::Float64`: mean of the Gaussian distribution of the couplings.
- `sigma2::Float64`: variance of the Gaussian distribution of the couplings.
- `gamma::Float64`: strength of the selection.
- `p_k::PdfDegVec`: degree distribution of the full graph.
- `p_cav_k::PdfDegVec`: degree distribution of the cavity graph.
- `P::Int`: number of sites in the population.
- `max_iter::Int`: maximum number of iterations.

# Optional arguments
- `tol::Dict{String, Float64}`: dictionary with the tolerance for convergence (default Dict("avg"=>1e-3)).
- `check_vars::Dict{String, Float64}`: dictionary with the variables to check for convergence (default Dict("avg_mu"=>0.0))
- `error_func::Function`: function to check for convergence. Must have arguments check_vars, mu_population, q_population, chi_population, tol (default error_func(check_vars, mu_population, q_population, chi_population, tol) which checks if the difference of the average mu between consecutive iterations is within the tolerance).
- `check_conv::Int`: number of iterations to check for convergence (default 30).
- `rng::AbstractRNG`: random number generator (default Xoshiro(1234)).
- `verbose::Bool`: print information about the convergence of the algorithm (default false).
- `plothist::Bool`: plot histograms of the populations at each check_conv iterations (default false).
- `damp::Float64`: damping factor for the update of the populations (default 0.0).

# Returns
- `mu_pop`: vector with the fixed point values of the variable mu.
- `q_pop`: vector with the fixed point values of the variable q.
- `chi_pop`: vector with the fixed point values of the variable chi.
- `mu_cav_pop`: vector with the fixed point values of the variable mu in the cavity.
- `q_cav_pop`: vector with the fixed point values of the variable q in the cavity.
- `chi_cav_pop`: vector with the fixed point values of the variable chi in the cavity.
- `converged`: boolean indicating if the algorithm converged.
"""
function population_dynamics(
    m::Float64, 
    sigma2::Float64, 
    gamma::Float64, 
    p_k::PdfDegVec, 
    p_cav_k::PdfDegVec,
    P::Int,
    max_iter::Int;
    tol::Dict{String, Float64} = Dict("avg"=>1e-3),
    check_vars::Dict{String, Float64} = Dict("avg_mu"=>0.0), 
    error_func::Function = error_func, 
    check_conv::Int = 30, 
    rng::AbstractRNG = Xoshiro(1234), 
    verbose::Bool = false,
    plothist::Bool = false,
    damp::Float64 = 0.0,
    init_pops::Dict{String, Float64} = Dict("mu"=>Inf, "q"=>Inf, "chi"=>Inf))

    K = p_k.K

    # Initialize populations
    if init_pops["mu"] == Inf
        mu_cav_pop = rand(rng, P)
    else
        mu_cav_pop = ones(P)*init_pops["mu"]
    end
    if init_pops["q"] == Inf
        q_cav_pop = rand(rng, P)
    else
        q_cav_pop = ones(P)*init_pops["q"]
    end
    if init_pops["chi"] == Inf
        chi_cav_pop = rand(rng, P) .- 0.3
    else
        chi_cav_pop = ones(P)*init_pops["chi"]
    end
    mu_pop = zeros(P)
    q_pop = zeros(P)
    chi_pop = zeros(P)
    J_pop = zeros(P)
    Jp_pop = zeros(P)
    neigh_idxs = zeros(Int, p_k.kmax)

    if plothist
        _, axs = plt.subplots(1,3,figsize=(10.5, 3))
    end

    muxlim = (0, 1)
    qxlim = (0, 1)
    chixlim = (0, 1)
    binedgesmu = 0:0.05:1
    binedgesq = 0:0.05:1
    binedgeschi = -0.3:0.05:1

    converged = false
    # Loop over iterations until convergence or max_iter is reached
    @inbounds @showprogress for t in 1:P*max_iter

        update_cav!(p_cav_k, P, m, sigma2, gamma, K, rng, mu_cav_pop, q_cav_pop, chi_cav_pop, J_pop, Jp_pop, neigh_idxs, damp)

        # Check for convergence
        if t % check_conv == 0  # Check every 10 iterations
            max_diff, converged = error_func(check_vars, mu_cav_pop, q_cav_pop, chi_cav_pop, tol)
            if verbose
                println("Iteration $t: max_diff = $max_diff")
            end
            if converged 
                if verbose
                    println("Converged after $t iterations.")
                end
                break
            end
            if plothist
                if t==check_conv
                    axs[1].cla()
                    binedgesmu = 0:0.05:maximum(mu_cav_pop)
                    fmu, _ = axs[1].hist(mu_cav_pop, bins=binedgesmu, alpha=0.5, density=true, color="C0")
                    muxlim = (0, maximum(mu_cav_pop))
                    axs[1].set_title("Histogram of μ values")
                    axs[1].set_xlabel("μ")
                    axs[1].set_ylabel("Frequency")
                    axs[1].set_xlim(muxlim)

                    axs[2].cla()
                    binedgesq = 0:0.05:maximum(q_cav_pop)
                    fq, _ = axs[2].hist(q_cav_pop, bins=binedgesq, alpha=0.5, density=true, color="C1")
                    qxlim = (0, maximum(q_cav_pop))
                    axs[2].set_title("Histogram of q values")
                    axs[2].set_xlabel("q")
                    axs[2].set_xlim(qxlim)

                    axs[3].cla()
                    binedgeschi = minimum(chi_cav_pop):0.05:maximum(chi_cav_pop)
                    fchi, _ = axs[3].hist(chi_cav_pop, bins=binedgeschi, alpha=0.5, density=true, color="C2")
                    chixlim = (minimum(chi_cav_pop), maximum(chi_cav_pop))
                    axs[3].set_title("Histogram of χ values")
                    axs[3].set_xlabel("χ")
                    axs[3].set_xlim(chixlim)
                else 
                    axs[1].cla()
                    axs[1].hist(mu_cav_pop, bins=binedgesmu, alpha=0.5, density=true, color="C0")
                    axs[1].set_title("Histogram of μ values")
                    axs[1].set_xlabel("μ")
                    axs[1].set_ylabel("Frequency")
                    axs[1].set_xlim(muxlim)

                    axs[2].cla()
                    axs[2].hist(q_cav_pop, bins=binedgesq, alpha=0.5, density=true, color="C1")
                    axs[2].set_title("Histogram of q values")
                    axs[2].set_xlabel("q")
                    axs[2].set_xlim(qxlim)

                    axs[3].cla()
                    axs[3].hist(chi_cav_pop, bins=binedgeschi, alpha=0.5, density=true, color="C2")
                    axs[3].set_title("Histogram of χ values")
                    axs[3].set_xlabel("χ")
                    axs[3].set_xlim(chixlim)
                end
            end
        end
    end

    update_full!(p_k, P, m, sigma2, gamma, K, rng, mu_cav_pop, q_cav_pop, chi_cav_pop, J_pop, Jp_pop, mu_pop, q_pop, chi_pop, neigh_idxs)

    if !converged && verbose
        println("Reached max iterations without convergence.")
    end

    return mu_pop, q_pop, chi_pop, mu_cav_pop, q_cav_pop, chi_cav_pop, converged
end

"""
    population_dynamics!(
        mu_cav_pop::Vector{Float64},
        q_cav_pop::Vector{Float64},
        chi_cav_pop::Vector{Float64},
        m::Float64, 
        sigma2::Float64, 
        gamma::Float64, 
        p_k::PdfDegVec, 
        p_cav_k::PdfDegVec,
        max_iter::Int;
        saveat::Union{Int, Vector{Int}} = 0,
        rng::AbstractRNG = Xoshiro(1234),
        damp::Float64 = 0.0
    )

Run the population dynamics algorithm for the fixed point abundances of the Random Lotka Volterra model on a sparse graph. It starts from already computed cavity populations, and iterate the population dynamics algorithm max_iter times saving both the cavity and full populations at the iterations specified by saveat.

# Arguments
- `mu_cav_pop::Vector{Float64}`: vector with the initial values of the variable mu in the cavity.
- `q_cav_pop::Vector{Float64}`: vector with the initial values of the variable q in the cavity.
- `chi_cav_pop::Vector{Float64}`: vector with the initial values of the variable chi in the cavity.
- `m::Float64`: mean of the Gaussian distribution of the couplings.
- `sigma2::Float64`: variance of the Gaussian distribution of the couplings.
- `gamma::Float64`: strength of the selection.
- `p_k::PdfDegVec`: degree distribution of the full graph.
- `p_cav_k::PdfDegVec`: degree distribution of the cavity graph.
- `max_iter::Int`: maximum number of iterations.

# Optional arguments
- `saveat::Union{Int, Vector{Int}}`: iteration numbers to save the populations (default 0). If it is an integer, it saves the populations every saveat iterations. If it is a vector, it saves the populations at the iterations specified by saveat.
- `rng::AbstractRNG`: random number generator (default Xoshiro(1234)).
- `damp::Float64`: damping factor for the update of the populations (default 0.0).

# Returns
- `mu_pop_vec`: matrix with the fixed point values of the variable mu at the iterations specified by saveat.
- `q_pop_vec`: matrix with the fixed point values of the variable q at the iterations specified by saveat.
- `chi_pop_vec`: matrix with the fixed point values of the variable chi at the iterations specified by saveat.
- `mu_cav_pop_vec`: matrix with the fixed point values of the variable mu in the cavity at the iterations specified by saveat.
- `q_cav_pop_vec`: matrix with the fixed point values of the variable q in the cavity at the iterations specified by saveat.
- `chi_cav_pop_vec`: matrix with the fixed point values of the variable chi in the cavity at the iterations specified by saveat.
"""
function population_dynamics!(
    mu_cav_pop::Vector{Float64},
    q_cav_pop::Vector{Float64},
    chi_cav_pop::Vector{Float64},
    m::Float64, 
    sigma2::Float64, 
    gamma::Float64, 
    p_k::PdfDegVec,
    p_cav_k::PdfDegVec,
    max_iter::Int;
    saveat::Union{Int, Vector{Int}} = 0,
    rng::AbstractRNG = Xoshiro(1234),
    damp::Float64 = 0.0)

    if saveat == 0
        saveat = max_iter
    end

    if typeof(saveat) == Int
        @assert saveat < max_iter "saveat must be less than max_iter"
        saveat = collect(1:saveat:max_iter)
    else
        @assert saveat[end] < max_iter "saveat values must be less than max_iter"
    end

    println("saveat = $saveat")
    P = length(mu_cav_pop)
    lensaveat = length(saveat)
    K = p_k.K

    # Initialize populations
    mu_pop = zeros(P)
    q_pop = zeros(P)
    chi_pop = zeros(P)
    mu_pop_vec = zeros(P, lensaveat)
    q_pop_vec = zeros(P, lensaveat)
    chi_pop_vec = zeros(P, lensaveat)
    mu_cav_pop_vec = zeros(P, lensaveat)
    q_cav_pop_vec = zeros(P, lensaveat)
    chi_cav_pop_vec = zeros(P, lensaveat)
    J_pop = zeros(P)
    Jp_pop = zeros(P)
    neigh_idxs = zeros(Int, p_k.kmax)

    # Loop over iterations until convergence or max_iter is reached
    idx = 1
    @inbounds @showprogress for t in 1:max_iter*P

        update_cav!(p_cav_k, P, m, sigma2, gamma, K, rng, mu_cav_pop, q_cav_pop, chi_cav_pop, J_pop, Jp_pop, neigh_idxs, damp)

        if t in saveat
            mu_cav_pop_vec[:, idx] .= mu_cav_pop
            q_cav_pop_vec[:, idx] .= q_cav_pop
            chi_cav_pop_vec[:, idx] .= chi_cav_pop
            update_full!(p_k, P, m, sigma2, gamma, K, rng, mu_cav_pop, q_cav_pop, chi_cav_pop, J_pop, Jp_pop, mu_pop, q_pop, chi_pop, neigh_idxs)
            mu_pop_vec[:, idx] .= mu_pop
            q_pop_vec[:, idx] .= q_pop
            chi_pop_vec[:, idx] .= chi_pop
            idx += 1
        end
    end

    return mu_pop_vec, q_pop_vec, chi_pop_vec, mu_cav_pop_vec, q_cav_pop_vec, chi_cav_pop_vec
end