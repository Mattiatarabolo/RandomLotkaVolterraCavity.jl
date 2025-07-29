# Placeholder update functions, where you can optimize the internal logic of f_mu, f_q, f_chi
function f_mu(Eps::RT, Delta::RT, Gamma::RT, regularization::RT) where {RT<:Real}
    if Gamma > zero(RT)
        return max(regularization, Eps / Gamma * (Delta * mod_erf(Delta) + gauss(Delta)))
    else
        return max(regularization, Eps / Gamma * (Delta * mod_erf(-Delta) - gauss(-Delta)))
    end
end

function f_q_dc(sum_q::RT, Delta::RT, Gamma::RT, regularization::RT) where {RT<:Real}
    if Gamma > zero(RT)
        return max(regularization, sum_q / Gamma ^ 2 * ((1 + Delta ^ 2) * mod_erf(Delta) + Delta * gauss(Delta)))
    else
        return max(regularization, sum_q / Gamma ^ 2 * ((1 + Delta ^ 2) * mod_erf(-Delta) - Delta * gauss(-Delta)))
    end
end

function f_chi(Delta::RT, Gamma::RT) where {RT<:Real}
    if Gamma > zero(RT)
        return 1 / Gamma * mod_erf(Delta)
    else
        return 1 / Gamma * mod_erf(-Delta)
    end
end

###########################################################################################################
############################## Single instance of disordered model ########################################
###########################################################################################################
"""
    run_GECaM_FP(model::Model{Deterministic, I, RT, MT, D}, max_iter::I, conv_threshold::RT, damp::RT; init_type::Symbol=:random, mu0::RT=0.0, q0::RT=0.0, chi0::RT=0.0, rng::AbstractRNG=Xoshiro(1234), showprogress::Bool=false, regularization::RT=-Inf) where {I<:Integer, RT<:Real, MT<:AbstractMatrix{RT}, D<:Distribution}

Run the Gaussian Expansion Cavity Method (GECaM) fixed-point algorithm for deterministic dynamics. It computes the fixed-point values of the mean abundances, correlations, and susceptibilities for a given instance of the disordered model.

# Arguments
- `model::Model{Deterministic, I, RT, MT, D}`: The model to run the GECaM on.
- `max_iter::I`: Maximum number of iterations to run.
- `conv_threshold::RT`: Convergence threshold for the fixed-point iterations.
- `damp::RT`: Damping factor for the updates.

# Keyword Arguments
- `init_type::Symbol`: Initialization type for the nodes. Supported types are `:zero`, `:random`, and `:custom` (default is `:random`).
- `mu0::RT`: Initial value for the mean (used if `init_type` is `:custom`, default is `0.0`).
- `q0::RT`: Initial value for the correlation (used if `init_type` is `:custom`, default is `0.0`).
- `chi0::RT`: Initial value for the susceptibility (used if `init_type` is `:custom`, default is `0.0`).
- `rng::AbstractRNG`: Random number generator for random initialization (default is `Xoshiro(1234)`).
- `showprogress::Bool`: Whether to show progress during the iterations (default is `false`).
- `divergence_threshold::RT`: Threshold for detecting divergence in the algorithm (default is `1e6`).
- `regularization::RT`: Regularization parameter to avoid negative values (default is `-Inf`).

# Output
- `nodes::Vector{NodeFP{Deterministic, I, RT}}`: The nodes with their updated cavity and marginal values after running the GECaM fixed-point algorithm.
- `converged::Bool`: Whether the algorithm converged within the specified number of iterations.
- `diverged::Bool`: Whether the algorithm diverged during the iterations.
"""
function run_GECaM_FP(model::Model{Deterministic, I, RT, MT, D}, max_iter::I, conv_threshold::RT, damp::RT; init_type::Symbol=:random, mu0::RT=0.0, q0::RT=0.0, chi0::RT=0.0, rng::AbstractRNG=Xoshiro(1234), showprogress::Bool=false, divergence_threshold::RT=1e6, regularization::RT=-Inf) where {I<:Integer, RT<:Real, MT<:AbstractMatrix{RT}, D<:Distribution}
    @assert max_iter > 0 "Maximum number of iterations must be greater than zero."
    @assert conv_threshold > 0 "Convergence threshold must be greater than zero."
    @assert damp >= 0 && damp <= 1 "Damping factor must be in the range [0, 1]."
    @assert init_type in [:zero, :random, :custom] "Initialization type must be one of: :zero, :random, :custom."
    @assert divergence_threshold > 0 "Divergence threshold must be greater than zero."

    # Initialize the nodes
    nodes = init_nodes(model, init_type, mu0, q0, chi0, rng)
    
    # Initialize convergence check
    converged = false
    diverged = false
    norm = zero(RT) # Initialize the norm for convergence check

    # Cavity iterations
    if showprogress
        println("Starting cavity iterations...")
        start = now()
    end
    for iter in 1:max_iter
        # Reset the norm for this iteration
        norm = zero(RT)
        # Iterate over the nodes
        for inode in shuffle(rng, nodes)
            i = inode.i # Node index
            # Sum over neighbors' contributions
            sum_mu = zero(RT)
            sum_q = zero(RT)
            sum_chi = zero(RT)
            for (jidx, jicav) in enumerate(inode.cavs)
                j = inode.neighs[jidx]
                sum_mu += model.J[i, j] * jicav.mu # Sum of cavity means
                sum_q += model.J[i, j] ^ 2 * jicav.q # Sum of cavity correlations
                sum_chi += model.J[i, j] * model.J[j, i] * jicav.chi # Sum of cavity susceptibilities
            end
            # Iterate over the cavities
            for (jidx, jicav) in enumerate(inode.cavs)
                j = inode.neighs[jidx] # Neighbor index
                iidx = nodes[j].neighs_idx[i] # Get the index of node i in the neighbors of node j
                # Compute Eps, Delta, Gamma
                Eps = sqrt(sum_q - model.J[i, j] ^ 2 * jicav.q)
                Delta = (1 + sum_mu - model.J[i, j] * jicav.mu) / Eps
                Gamma = 1 - sum_chi - model.J[i, j] * model.J[j, i] * jicav.chi
                # Compute the new cavity values
                new_mu = f_mu(Eps, Delta, Gamma, regularization)
                new_q = max(regularization, f_q_dc(sum_q, Delta, Gamma, regularization) - new_mu ^ 2)
                new_chi = f_chi(Delta, Gamma)
                # Check for divergence
                if !isfinite(new_mu) || !isfinite(new_q) || !isfinite(new_chi) || new_mu < 0 || new_q < 0 || new_mu > divergence_threshold || new_q > divergence_threshold || new_chi > divergence_threshold
                    println("Divergence (or negative values) detected in cavity ($i, $j): mu=$(new_mu), q=$(new_q), chi=$(new_chi).")
                    diverged = true
                    break
                end
                # Compute the norm for convergence check
                old_mu, old_q, old_chi = nodes[j].cavs[iidx].mu, nodes[j].cavs[iidx].q, nodes[j].cavs[iidx].chi
                new_norm = damp * max(abs(new_mu - old_mu), abs(new_q - old_q), abs(new_chi - old_chi))
                norm = max(norm, new_norm) # Update the norm
                # Update the cavity values with damping
                nodes[j].cavs[iidx].mu = damp * new_mu + (1 - damp) * old_mu
                nodes[j].cavs[iidx].q = damp * new_q + (1 - damp) * old_q
                nodes[j].cavs[iidx].chi = damp * new_chi + (1 - damp) * old_chi
            end
            # Check for divergence 
            if diverged
                break
            end
        end
        # Print status
        if showprogress
            println("Iteration $iter: $norm (convergence threshold $conv_threshold)")
        end
        # Check for divergence or convergence
        if diverged
            println("Divergence detected in iteration $iter. Returning early.")
            break
        elseif norm < conv_threshold
            converged = true
            if showprogress
                println("Convergence achieved in iteration $iter with norm $norm (convergence threshold $conv_threshold).")
                println("Total time taken: $(now() - start).")
                break
            end
        end
    end

    # Checl for convergence
    if !converged && !diverged
        println("Maximum iterations reached without convergence. Final norm: $norm (convergence threshold $conv_threshold).")
    end

    # Marginal updates
    for inode in nodes
        i = inode.i # Node index
        # Sum over neighbors' contributions
        sum_mu = zero(RT)
        sum_q = zero(RT)
        sum_chi = zero(RT)
        for (jidx, jicav) in enumerate(inode.cavs)
            j = inode.neighs[jidx]
            sum_mu += model.J[i, j] * jicav.mu # Sum of cavity means
            sum_q += model.J[i, j] ^ 2 * jicav.q # Sum of cavity correlations
            sum_chi += model.J[i, j] * model.J[j, i] * jicav.chi # Sum of cavity susceptibilities
        end
        # Compute Eps, Delta, Gamma
        Eps = sqrt(sum_q)
        Delta = (1 + sum_mu) / Eps
        Gamma = 1 - sum_chi
        # Compute the new marginal values
        inode.marg.mu = f_mu(Eps, Delta, Gamma, regularization)
        inode.marg.q = max(regularization, f_q_dc(sum_q, Delta, Gamma, regularization) - inode.marg.mu ^ 2)
        inode.marg.chi = f_chi(Delta, Gamma)
        # Check for divergence
        if !isfinite(inode.marg.mu) || !isfinite(inode.marg.q) || !isfinite(inode.marg.chi) || inode.marg.mu < 0 || inode.marg.q < 0 || inode.marg.mu > divergence_threshold || inode.marg.q > divergence_threshold || inode.marg.chi > divergence_threshold
            println("Divergence (or negative values) detected in marginal node $(inode.i): mu=$(inode.marg.mu), q=$(inode.marg.q), chi=$(inode.marg.chi).")
            diverged = true
            break
        end
    end

    return nodes, converged, diverged
end


###########################################################################################################
################################# Average over disordered model ###########################################
###########################################################################################################
function sumpop(pop::PopFP{Deterministic, I, RT}, couplings_pop::PopJ{Deterministic, I, RT}, neigh_idxs::Vector{I}, k_cav::I) where {I<:Integer, RT<:Real}
    # Initialize sums
    sum_mu = zero(RT)
    sum_q = zero(RT)
    sum_chi = zero(RT)

    # Iterate over the neighbors' indices
    for j in neigh_idxs[1:k_cav]
        J = couplings_pop.J_pop[j]
        Jp = couplings_pop.Jp_pop[j]
        mu_j = pop.mu_pop[j]
        q_j = pop.q_pop[j]
        chi_j = pop.chi_pop[j]

        sum_mu += J * mu_j
        sum_q += J ^ 2 * q_j
        sum_chi += J * Jp * chi_j
    end

    # Compute Eps, Delta, Gamma
    Eps = sqrt(sum_q)
    Delta = (1 + sum_mu) / Eps
    Gamma = 1 - sum_chi

    return sum_q, Eps, Delta, Gamma
end

"""
    run_GECaM_FP(model::ModelDisordered{Deterministic, I, RT, D1, D2, D3, FT}, P::I, max_iter::I, conv_threshold::RT, damp::RT; init_type::Symbol=:random, mu0::RT=0.0, q0::RT=0.0, chi0::RT=0.0, rng::AbstractRNG=Xoshiro(1234), showprogress::Bool=false, divergence_threshold::RT=1e6, regularization::RT=-Inf) where {I<:Integer, RT<:Real, D1<:Distribution, D2<:Distribution, D3<:Distribution, FT<:Function}

Run the Gaussian Expansion Cavity Method (GECaM) fixed-point algorithm for a disordered model with deterministic dynamics. It computes through a population dynamics algorithm the fixed-point values of the mean abundances, correlations, and susceptibilities averaged over the disordered model.

# Arguments
- `model::ModelDisordered{Deterministic, I, RT, D1, D2, D3, FT}`: The disordered model to run the GECaM on.
- `P::I`: Number of elements of the populations.
- `max_iter::I`: Maximum number of iterations to run.
- `conv_threshold::RT`: Convergence threshold for the algorithm.
- `damp::RT`: Damping factor for the updates.

# Keyword Arguments
- `init_type::Symbol`: Initialization type for the nodes. Supported types are `:zero`, `:random`, and `:custom` (default is `:random`).
- `mu0::RT`: Initial value for the mean (used if `init_type` is `:custom`, default is `0.0`).
- `q0::RT`: Initial value for the correlation (used if `init_type` is `:custom`, default is `0.0`).
- `chi0::RT`: Initial value for the susceptibility (used if `init_type` is `:custom`, default is `0.0`).
- `rng::AbstractRNG`: Random number generator for random initialization (default is `Xoshiro(1234)`).
- `showprogress::Bool`: Whether to show progress during the iterations (default is `false`).
- `divergence_threshold::RT`: Threshold for detecting divergence in the algorithm (default  is `1e6`).
- `regularization::RT`: Regularization parameter to avoid negative values (default is `-Inf`).

# Output
- `cav_pop::PopFP{Deterministic, I, RT}`: The cavity population with updated values after running the GECaM fixed-point algorithm.
- `marg_pop::PopFP{Deterministic, I, RT}`: The marginal population with updated values after running the GECaM fixed-point algorithm.
- `converged::Bool`: Whether the algorithm converged within the specified number of iterations.
- `diverged::Bool`: Whether the algorithm diverged during the iterations.
"""
function run_GECaM_FP(model::ModelDisordered{Deterministic, I, RT, D1, D2, D3, FT}, P::I, max_iter::I, conv_threshold::RT, damp::RT; init_type::Symbol=:random, mu0::RT=0.0, q0::RT=0.0, chi0::RT=0.0, rng::AbstractRNG=Xoshiro(1234), showprogress::Bool=false, divergence_threshold::RT=1e6, regularization::RT=-Inf) where {I<:Integer, RT<:Real, D1<:Distribution, D2<:Distribution, D3<:Distribution, FT<:Function}
    @assert max_iter > 0 "Maximum number of iterations must be greater than zero."
    @assert conv_threshold > 0 "Convergence threshold must be greater than zero."
    @assert damp >= 0 && damp <= 1 "Damping factor must be in the range [0, 1]."
    @assert init_type in [:zero, :random, :custom] "Initialization type must be one of: :zero, :random, :custom."
    @assert divergence_threshold > 0 "Divergence threshold must be greater than zero."

    m, sigma2, corr, K = model.m, model.sigma2, model.corr, model.K # Default values from the model

    # Initialize the cavity populations
    cav_pop = PopFP(P, init_type, mu0, q0, chi0, rng)
    couplings_pop = PopJ(P, m, sigma2, corr, K, rng)

    # Initialize cavity neighbors indices
    k_cav_max = maximum(model.deg_cav_pdf) # Maximum number of neighbors in the cavity
    neigh_idxs = zeros(I, k_cav_max) # Neighbors indices

    # Initialize means of the populations for convergence check
    old_avg_mu = mean(cav_pop.mu_pop)
    old_avg_q = mean(cav_pop.q_pop)
    old_avg_chi = mean(cav_pop.chi_pop)
    
    # Initialize convergence check
    converged = false
    diverged = false
    norm = zero(RT) # Initialize the norm for convergence check

    # Cavity iterations
    if showprogress
        println("Starting cavity iterations...")
        start = now()
    end

    # Cavity iterations
    for iter in 1:max_iter
        # Reset the norm for this iteration
        norm = zero(RT)
        # Iterate over the population
        for ipop in shuffle(rng, 1:P)
            # Sample the degree of the cavity
            k_cav = rand(rng, model.deg_cav_pdf)
            # Check if degree is zero
            if k_cav == 0
                new_mu = one(RT)
                new_q = one(RT)
                new_chi = one(RT)
            else
                # Sample k_cav neighbors indices
                neigh_idxs[1:k_cav] .= sample(rng, 1:P, k_cav, replace=false)
                # Sample k_cav pairs of correlated couplings
                for j in neigh_idxs[1:k_cav]
                    couplings_pop.J_pop[j], couplings_pop.Jp_pop[j] = sample_couplings(m, sigma2, corr, K; rng=rng)
                end
                # Compute the sums over neighbors' contributions
                sum_q_cav, Eps_cav, Delta_cav, Gamma_cav = sumpop(cav_pop, couplings_pop, neigh_idxs, k_cav)
                # Compute the new cavity values
                new_mu = f_mu(Eps_cav, Delta_cav, Gamma_cav, regularization)
                new_q = max(regularization, f_q_dc(sum_q_cav, Delta_cav, Gamma_cav, regularization))# - new_mu ^ 2)
                new_chi = f_chi(Delta_cav, Gamma_cav)
            end
            # Check for divergence
            if (!isfinite(new_mu) || !isfinite(new_q) || !isfinite(new_chi) || new_mu < 0 || new_q < 0 || new_mu > divergence_threshold || new_q > divergence_threshold || new_chi > divergence_threshold) && showprogress
                println("Divergence (or negative values) detected in cavity $ipop: mu=$(new_mu), q=$(new_q), chi=$(new_chi).")
                diverged = true
                break
            end
            # Update the cavity values with damping
            old_mu, old_q, old_chi = cav_pop.mu_pop[ipop], cav_pop.q_pop[ipop], cav_pop.chi_pop[ipop]
            cav_pop.mu_pop[ipop] = damp * new_mu + (1 - damp) * old_mu
            cav_pop.q_pop[ipop] = damp * new_q + (1 - damp) * old_q
            cav_pop.chi_pop[ipop] = damp * new_chi + (1 - damp) * old_chi
        end
        # Check for divergence
        if diverged && showprogress
            println("Divergence detected in iteration $iter. Returning early.")
            break
        end
        # Compute the averages for convergence check
        new_avg_mu = mean(cav_pop.mu_pop)
        new_avg_q = mean(cav_pop.q_pop)
        new_avg_chi = mean(cav_pop.chi_pop)
        # Compute the norm for convergence check
        new_norm = max(abs(new_avg_mu - old_avg_mu), abs(new_avg_q - old_avg_q), abs(new_avg_chi - old_avg_chi))
        norm = max(norm, new_norm) # Update the norm
        # Update the averages for the next iteration
        old_avg_mu, old_avg_q, old_avg_chi = new_avg_mu, new_avg_q, new_avg_chi
        # Print status
        if showprogress
            println("Iteration $iter: $norm (convergence threshold $conv_threshold)")
        end
        # Check for convergence
        if norm < conv_threshold
            converged = true
            if showprogress
                println("Convergence achieved in iteration $iter with norm $norm (convergence threshold $conv_threshold).")
                println("Total time taken: $(now() - start).")
                break
            end
        end
    end

    # Check for convergence
    if !converged && !diverged && showprogress
        println("Maximum iterations reached without convergence. Final norm: $norm (convergence threshold $conv_threshold).")
    end

    # Initialize the marginal population
    marg_pop = PopFP(P, init_type, mu0, q0, chi0, rng)

    # Initialize marginal neighbors indices
    k_marg_max = maximum(model.deg_pdf) # Maximum number of neighbors in the cavity
    neigh_idxs = zeros(I, k_marg_max) # Neighbors indices

    # Marginal updates
    for ipop in 1:P
        # Sample the degree of the marginal
        k_marg = rand(rng, model.deg_pdf)
        # Check if degree is zero
        if k_marg == 0
            marg_pop.mu_pop[ipop] = one(RT)
            marg_pop.q_pop[ipop] = one(RT)
            marg_pop.chi_pop[ipop] = one(RT)
        else
            # Sample k_marg neighbors indices
            neigh_idxs[1:k_marg] .= sample(rng, 1:P, k_marg, replace=false)
            # Sample k_marg pairs of correlated couplings
            for j in neigh_idxs[1:k_marg]
                couplings_pop.J_pop[j], couplings_pop.Jp_pop[j] = sample_couplings(m, sigma2, corr, K; rng=rng)
            end
            # Compute the sums over neighbors' contributions
            sum_q_marg, Eps_marg, Delta_marg, Gamma_marg = sumpop(cav_pop, couplings_pop, neigh_idxs, k_marg)
            # Compute the new cavity values
            marg_pop.mu_pop[ipop] = f_mu(Eps_marg, Delta_marg, Gamma_marg, regularization)
            marg_pop.q_pop[ipop] = max(regularization, f_q_dc(sum_q_marg, Delta_marg, Gamma_marg, regularization))# - marg_pop.mu_pop[ipop] ^ 2)
            marg_pop.chi_pop[ipop] = f_chi(Delta_marg, Gamma_marg)
        end
        # Check for divergence
        if (!isfinite(marg_pop.mu_pop[ipop]) || !isfinite(marg_pop.q_pop[ipop]) || !isfinite(marg_pop.chi_pop[ipop]) || marg_pop.mu_pop[ipop] < 0 || marg_pop.q_pop[ipop] < 0 || marg_pop.mu_pop[ipop] > divergence_threshold || marg_pop.q_pop[ipop] > divergence_threshold || marg_pop.chi_pop[ipop] > divergence_threshold) && showprogress
            println("Divergence (or negative values) detected in marginal node $ipop: mu=$(marg_pop.mu_pop[ipop]), q=$(marg_pop.q_pop[ipop]), chi=$(marg_pop.chi_pop[ipop]).")
            diverged = true
            break
        end
    end

    return cav_pop, marg_pop, converged, diverged
end