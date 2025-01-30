# Shortname for functions used in cavity update
function gauss(x::Float64)
    return exp( - x^2 / 2) / sqrt( 2 * pi )
end

function mod_erf(x::Float64)
    return ( 1 + erf(x / sqrt(2)) ) / 2
end

# Function to compute all the quantities obtain by the sum of neighbours' terms in the cavity update
function sumpop(mu_pop::Vector{Float64}, q_pop::Vector{Float64}, chi_pop::Vector{Float64}, J_pop::Vector{Float64}, Jp_pop::Vector{Float64}, neigh_idxs::Vector{Int}, k::Int)
    sum_mu = 0.0
    sum_q = 0.0
    sum_chi = 0.0
    for neigh_idx in 1:k
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

function update_cav!(p_cav_k::PdfDegVec, P::Int, m::Float64, sigma2::Float64, gamma::Float64, K::Int, rng::AbstractRNG, mu_cav_pop::Vector{Float64}, q_cav_pop::Vector{Float64}, chi_cav_pop::Vector{Float64}, J_pop::Vector{Float64}, Jp_pop::Vector{Float64}, neigh_idxs::Vector{Int})
    # Update each site in the population
    for i in shuffle(rng, 1:P)
        # Sample the degree k
        k_cav = sample_degree(rng, p_cav_k)

        if k_cav == 0
            mu_cav_pop[i] = 1.0
            q_cav_pop[i] = 1.0
            chi_cav_pop[i] = 0.0
            continue
        end

        # Get k random neighbors
        sample_neighs!(rng, neigh_idxs, i, k_cav, P)

        println("neigh_idxs: ", neigh_idxs)
        
        # CAVITY UPDATE
        # Sample k_cav pairs of correlated J, J' values
        for j in neigh_idxs
            J_pop[j], Jp_pop[j] = sample_couplings(rng, m, sigma2, gamma, K)
        end

        ## We could sample a bigger population of J at the beginning and sample from it ##

        # Compute the new values for mu, q, chi using the update functions
        sum_q_cav, sum_chi_cav, Eps, Delta = sumpop(mu_cav_pop, q_cav_pop, chi_cav_pop, J_pop, Jp_pop, neigh_idxs, k_cav)
        mu_cav_pop[i] = f_mu(sum_chi_cav, Eps, Delta)
        q_cav_pop[i] = f_q(sum_q_cav, sum_chi_cav, Delta)
        chi_cav_pop[i] = f_chi(sum_chi_cav, Delta)
        testvalues(mu_cav_pop[i], q_cav_pop[i], chi_cav_pop[i], sum_q_cav, sum_chi_cav, Delta)
    end
end

function update_full!(p_k::PdfDegVec, P::Int, m::Float64, sigma2::Float64, gamma::Float64, K::Int, rng::AbstractRNG, mu_cav_pop::Vector{Float64}, q_cav_pop::Vector{Float64}, chi_cav_pop::Vector{Float64}, J_pop::Vector{Float64}, Jp_pop::Vector{Float64}, mu_pop::Vector{Float64}, q_pop::Vector{Float64}, chi_pop::Vector{Float64}, neigh_idxs::Vector{Int})
    # Update each site in the population
    for i in shuffle(rng, 1:P)
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
        for j in neigh_idxs
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
function population_dynamics(
    p_k::PdfDegVec, 
    p_cav_k::PdfDegVec, 
    P::Int, 
    tol, 
    max_iter::Int, 
    m::Float64, 
    sigma2::Float64, 
    gamma::Float64, 
    K::Int;
    check_vars::Dict{String, Float64} = Dict("avg_mu"=>0.0), 
    error_func::Function = error_func, 
    check_conv::Int = 30, 
    rng::AbstractRNG = Xoshiro(1234), 
    verbose::Bool = false,
    plothist::Bool = false)

    # Initialize populations
    mu_cav_pop = rand(rng, P)
    q_cav_pop = rand(rng, P)
    chi_cav_pop = rand(rng, P) .- 0.3
    mu_pop = zeros(P)
    q_pop = zeros(P)
    chi_pop = zeros(P)
    J_pop = zeros(P)
    Jp_pop = zeros(P)
    neigh_idxs = zeros(Int, p_k.kmax)

    if plothist
        _, axs = plt.subplots(1,3,figsize=(10, 4))
    end

    muxlim = (0, 1)
    qxlim = (0, 1)
    chixlim = (0, 1)
    muylim = (0, 1)
    qylim = (0, 1)
    chiylim = (0, 1)

    converged = false
    # Loop over iterations until convergence or max_iter is reached
    @showprogress for t in 1:max_iter

        update_cav!(p_cav_k, P, m, sigma2, gamma, K, rng, mu_cav_pop, q_cav_pop, chi_cav_pop, J_pop, Jp_pop, neigh_idxs)

        converged = error_func(check_vars, mu_cav_pop, q_cav_pop, chi_cav_pop, tol, t, check_conv)

        # Check for convergence
        if t % check_conv == 0  # Check every 10 iterations
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
                update_full!(p_k, P, m, sigma2, gamma, K, rng, mu_cav_pop, q_cav_pop, chi_cav_pop, J_pop, Jp_pop, mu_pop, q_pop, chi_pop, neigh_idxs)
                
                if t==check_conv
                    axs[1].cla()
                    fmu, _ = axs[1].hist(mu_pop, bins=30, alpha=0.5, density=true, color="C0")
                    muxlim = (0, maximum(mu_pop))
                    muylim = (0, maximum(fmu)*1.1)
                    axs[1].set_title("Histogram of μ values")
                    axs[1].set_xlabel("μ")
                    axs[1].set_ylabel("Frequency")
                    axs[1].set_xlim(muxlim)
                    axs[1].set_ylim(muylim)

                    axs[2].cla()
                    fq, _ = axs[2].hist(q_pop, bins=30, alpha=0.5, density=true, color="C1")
                    qxlim = (0, maximum(q_pop))
                    qylim = (0, maximum(fq)*1.1)
                    axs[2].set_title("Histogram of q values")
                    axs[2].set_xlabel("q")
                    axs[2].set_xlim(qxlim)
                    axs[2].set_ylim(qylim)

                    axs[3].cla()
                    fchi, _ = axs[3].hist(chi_pop, bins=30, alpha=0.5, density=true, color="C2")
                    chixlim = (minimum(chi_pop), maximum(chi_pop))
                    chiylim = (minimum(chi_pop), maximum(fchi)*1.1)
                    axs[3].set_title("Histogram of χ values")
                    axs[3].set_xlabel("χ")
                    axs[3].set_xlim(chixlim)
                    axs[3].set_ylim(chiylim)
                else 
                    axs[1].cla()
                    axs[1].hist(mu_pop, bins=30, alpha=0.5, density=true, color="C0")
                    axs[1].set_title("Histogram of μ values")
                    axs[1].set_xlabel("μ")
                    axs[1].set_ylabel("Frequency")
                    axs[1].set_xlim(muxlim)
                    axs[1].set_ylim(muylim)

                    axs[2].cla()
                    axs[2].hist(q_pop, bins=30, alpha=0.5, density=true, color="C1")
                    axs[2].set_title("Histogram of q values")
                    axs[2].set_xlabel("q")
                    axs[2].set_xlim(qxlim)
                    axs[2].set_ylim(qylim)

                    axs[3].cla()
                    axs[3].hist(chi_pop, bins=30, alpha=0.5, density=true, color="C2")
                    axs[3].set_title("Histogram of χ values")
                    axs[3].set_xlabel("χ")
                    axs[3].set_xlim(chixlim)
                    axs[3].set_ylim(chiylim)
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



#=
####################################################################################################################################################
########################################################## Fully connected graph ###################################################################
####################################################################################################################################################

# Function to compute all the quantities obtain by the sum of neighbours' terms in the cavity update
function sumpop_FC(
    mu_pop::Vector{Float64}, 
    q_pop::Vector{Float64}, 
    q_pop::Vector{Float64}, 
    J_pop::Vector{Float64}, 
    Jp_pop::Vector{Float64}, 
    i::Int, 
    P::Int)

    sum_mu = 0.0
    sum_q = 0.0
    sum_chi = 0.0
    for j in 1:P
        if j == i
            continue
        end
        sum_mu += J_pop[j] * mu_pop[j]
        sum_q += J_pop[j]^2 * q_pop[j]
        sum_chi += J_pop[j] * Jp_pop[j] * q_pop[j]
    end
    Eps = sqrt(sum_q)
    Delta = (1+sum_mu)/Eps

    testvalues(sum_mu, sum_q, sum_chi, Eps, Delta)
    return sum_q, sum_chi, Eps, Delta
end

# Function to run population dynamics with all the performance tips included
function population_dynamics_FC(
    P::Int, 
    tol, 
    max_iter::Int, 
    m::Float64, 
    sigma2::Float64, 
    gamma::Float64; 
    check_vars=Dict("avg_mu"=>0.0), 
    error_func=error_func, 
    check_conv=30, 
    rng=Xoshiro(1234), 
    verbose=false)

    # Initialize populations
    mu_pop = rand(rng, P)
    q_pop = rand(rng, P)
    q_pop = rand(rng, P)
    J_pop = zeros(P)
    Jp_pop = zeros(P)

    converged = false
    # Loop over iterations until convergence or max_iter is reached
    @showprogress for t in 1:max_iter

        # Update each site in the population
        for i in shuffle(rng, 1:P)
            for j in 1:P
                if j == i
                    continue
                end
                J_pop[j], Jp_pop[j] = sample_couplings(rng, m, sigma2, gamma, P)
            end
            # Compute the new values for mu, q, chi using the update functions
            sum_q, sum_chi, Eps, Delta = sumpop_FC(mu_pop, q_pop, q_pop, J_pop, Jp_pop, i, P)
            mu_pop[i] = f_mu(sum_chi, Eps, Delta)
            q_pop[i] = f_q(sum_q, sum_chi, Delta)
            q_pop[i] = f_chi(sum_chi, Delta)
            testvalues(mu_pop[i], q_pop[i], q_pop[i], sum_q, sum_chi, Delta)  
        end

        converged = error_func(check_vars, mu_pop, q_pop, q_pop, tol, t, check_conv)

        # Check for convergence
        if t % check_conv == 0  # Check every 10 iterations
            if verbose
                println("Iteration $t: max_diff = $max_diff")
            end
            if converged
                if verbose
                    println("Converged after $t iterations.")
                end
                break
            end
        end
    end

    if !converged && verbose
        println("Reached max iterations without convergence.")
    end

    return mu_pop, q_pop, q_pop, converged
end
=#