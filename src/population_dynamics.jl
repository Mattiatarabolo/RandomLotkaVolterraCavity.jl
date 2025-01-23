# Shortname for functions used in cavity update
function φ_func(x::Float64)
    return exp(-x^2/2)/sqrt(2*pi)
end

function Φ_func(x::Float64)
    return (1+erf(x/sqrt(2)))/2
end

# Function to compute all the quantities obtain by the sum of neighbours' terms in the cavity update
function sumpop(
    mu_population::Vector{Float64}, 
    q_population::Vector{Float64}, 
    Chi_population::Vector{Float64}, 
    J_population::Vector{Float64}, 
    J_prime_population::Vector{Float64}, 
    neighbors_indices::Vector{Int})
    
    sum_mu = 0.0
    sum_q = 0.0
    sum_Chi = 0.0
    @inbounds @fastmath @simd for j in neighbors_indices
        sum_mu += J_population[j] * mu_population[j]
        sum_q += J_population[j]^2 * q_population[j]
        sum_Chi += J_population[j] * J_prime_population[j] * Chi_population[j]
    end
    Ε = sqrt(sum_q)
    Δ = (1+sum_mu)/Ε

    testvalues(sum_mu, sum_q, sum_Chi, Ε, Δ)
    return sum_q, sum_Chi, Ε, Δ
end

# Placeholder update functions, where you can optimize the internal logic of f_mu, f_q, f_chi
function f_mu(sum_Chi::Float64, Ε::Float64, Δ::Float64)
    if sum_Chi < 1.0
        return Ε / (1 - sum_Chi) * (Δ * Φ_func(Δ) + φ_func(Δ))
    else
        return Ε / (1 - sum_Chi) * (Δ * Φ_func(-Δ) - φ_func(-Δ))
    end
end

function f_q(sum_q::Float64, sum_Chi::Float64, Δ::Float64)
    if sum_Chi < 1.0
        return sum_q / (1 - sum_Chi)^2 * ((1+Δ^2) * Φ_func(Δ) + Δ * φ_func(Δ))
    else
        return sum_q / (1 - sum_Chi)^2 * ((1+Δ^2) * Φ_func(-Δ) - Δ * φ_func(-Δ))
    end
end

function f_Chi(sum_Chi::Float64, Δ::Float64)
    if sum_Chi < 1.0
        return 1 / (1 - sum_Chi) * Φ_func(Δ)
    else
        return 1 / (1 - sum_Chi) * Φ_func(-Δ)
    end
end


####################################################################################################################################################
########################################################## Sparse graph ###########################################################################
####################################################################################################################################################

function update_cav!(p_cav_k, P, m, σ², γ, K, rng, mu_cav_population, q_cav_population, Chi_cav_population, J_population, J_prime_population)
    # Update each site in the population
    @inbounds for i in shuffle(rng, 1:P)
        # Sample the degree k
        k_cav = sample_degree(rng, 0:length(p_cav_k)-1, p_cav_k)

        # Get k random neighbors
        neighbors_indices_cav = sample(rng, filter!(x->x≠i,collect(1:P)), k_cav; replace=false)
        
        # CAVITY UPDATE
        # Sample k_cav pairs of correlated J, J' values
        @inbounds for j in neighbors_indices_cav
            J_population[j], J_prime_population[j] = sample_couplings(rng, m, σ², γ, K)
        end

        # Compute the new values for mu, q, chi using the update functions
        sum_q_cav, sum_Chi_cav, Ε_cav, Δ_cav = sumpop(mu_cav_population, q_cav_population, Chi_cav_population, J_population, J_prime_population, neighbors_indices_cav)
        mu_cav_population[i] = f_mu(sum_Chi_cav, Ε_cav, Δ_cav)
        q_cav_population[i] = f_q(sum_q_cav, sum_Chi_cav, Δ_cav)
        Chi_cav_population[i] = f_Chi(sum_Chi_cav, Δ_cav)
        testvalues(mu_cav_population[i], q_cav_population[i], Chi_cav_population[i], sum_q_cav, sum_Chi_cav, Δ_cav)
    end
end

function update_full!(p_k, P, m, σ², γ, K, rng, mu_cav_population, q_cav_population, Chi_cav_population, J_population, J_prime_population, mu_full_population, q_full_population, Chi_full_population)
    # Update each site in the population
    @inbounds for i in shuffle(rng, 1:P)
        k_full = sample_degree(rng, 0:length(p_k)-1, p_k)
        neighbors_indices_full = sample(rng, filter!(x->x≠i,collect(1:P)), k_full; replace=false)

        # FULL UPDATE
        # Sample k_cav pairs of correlated J, J' values
        @inbounds for j in neighbors_indices_full
            J_population[j], J_prime_population[j] = sample_couplings(rng, m, σ², γ, K)
        end
        # Compute the new values for mu, q, chi using the update functions
        sum_q_full, sum_Chi_full, Ε_full, Δ_full = sumpop(mu_cav_population, q_cav_population, Chi_cav_population, J_population, J_prime_population, neighbors_indices_full)
        mu_full_population[i] = f_mu(sum_Chi_full, Ε_full, Δ_full)
        q_full_population[i] = f_q(sum_q_full, sum_Chi_full, Δ_full)
        Chi_full_population[i] = f_Chi(sum_Chi_full, Δ_full)
        testvalues(mu_full_population[i], q_full_population[i], Chi_full_population[i], sum_q_full, sum_Chi_full, Δ_full) 
    end
end

# Function to run population dynamics with all the performance tips included
function population_dynamics(
    p_k::Vector{Float64}, 
    p_cav_k::Vector{Float64}, 
    P::Int, 
    tol, 
    max_iter::Int, 
    m::Float64, 
    σ²::Float64, 
    γ::Float64, 
    K::Int;
    check_vars=Dict("avg_mu"=>0.0), 
    error_func=error_func, 
    check_conv=30, 
    rng=Xoshiro(1234), 
    verbose=false,
    plothist=false)

    # Initialize populations
    mu_cav_population = rand(rng, P)
    q_cav_population = rand(rng, P)
    Chi_cav_population = rand(rng, P)
    mu_full_population = zeros(P)
    q_full_population = zeros(P)
    Chi_full_population = zeros(P)
    J_population = zeros(P)
    J_prime_population = zeros(P)

    if plothist
        _, axs = plt.subplots(1,3,figsize=(10, 4))
    end

    muxlim = (0, 1)
    qxlim = (0, 1)
    Chixlim = (0, 1)
    muylim = (0, 1)
    qylim = (0, 1)
    Chiylim = (0, 1)

    converged = false
    # Loop over iterations until convergence or max_iter is reached
    @showprogress for t in 1:max_iter

        update_cav!(p_cav_k, P, m, σ², γ, K, rng, mu_cav_population, q_cav_population, Chi_cav_population, J_population, J_prime_population)

        converged = error_func(check_vars, mu_cav_population, q_cav_population, Chi_cav_population, tol, t, check_conv)

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
                update_full!(p_k, P, m, σ², γ, K, rng, mu_cav_population, q_cav_population, Chi_cav_population, J_population, J_prime_population, mu_full_population, q_full_population, Chi_full_population)
                
                if t==check_conv
                    axs[1].cla()
                    fmu, _ = axs[1].hist(mu_full_population, bins=20, alpha=0.5, density=true, color="C0")
                    muxlim = (0, maximum(mu_full_population))
                    muylim = (0, maximum(fmu)*1.1)
                    axs[1].set_title("Histogram of μ values")
                    axs[1].set_xlabel("μ")
                    axs[1].set_ylabel("Frequency")
                    axs[1].set_xlim(muxlim)
                    axs[1].set_ylim(muylim)

                    axs[2].cla()
                    fq, _ = axs[2].hist(q_full_population, bins=20, alpha=0.5, density=true, color="C1")
                    qxlim = (0, maximum(q_full_population))
                    qylim = (0, maximum(fq)*1.1)
                    axs[2].set_title("Histogram of q values")
                    axs[2].set_xlabel("q")
                    axs[2].set_xlim(qxlim)
                    axs[2].set_ylim(qylim)

                    axs[3].cla()
                    fChi, _ = axs[3].hist(Chi_full_population, bins=20, alpha=0.5, density=true, color="C2")
                    Chixlim = (0, maximum(Chi_full_population))
                    Chiylim = (0, maximum(fChi)*1.1)
                    axs[3].set_title("Histogram of Χ values")
                    axs[3].set_xlabel("Χ")
                    axs[3].set_xlim(Chixlim)
                    axs[3].set_ylim(Chiylim)
                else 
                    axs[1].cla()
                    axs[1].hist(mu_full_population, bins=20, alpha=0.5, density=true, color="C0")
                    axs[1].set_title("Histogram of μ values")
                    axs[1].set_xlabel("μ")
                    axs[1].set_ylabel("Frequency")
                    axs[1].set_xlim(muxlim)
                    axs[1].set_ylim(muylim)

                    axs[2].cla()
                    axs[2].hist(q_full_population, bins=20, alpha=0.5, density=true, color="C1")
                    axs[2].set_title("Histogram of q values")
                    axs[2].set_xlabel("q")
                    axs[2].set_xlim(qxlim)
                    axs[2].set_ylim(qylim)

                    axs[3].cla()
                    axs[3].hist(Chi_full_population, bins=20, alpha=0.5, density=true, color="C2")
                    axs[3].set_title("Histogram of Χ values")
                    axs[3].set_xlabel("Χ")
                    axs[3].set_xlim(Chixlim)
                    axs[3].set_ylim(Chiylim)
                end
            end
        end
    end

    update_full!(p_k, P, m, σ², γ, K, rng, mu_cav_population, q_cav_population, Chi_cav_population, J_population, J_prime_population, mu_full_population, q_full_population, Chi_full_population)

    if !converged && verbose
        println("Reached max iterations without convergence.")
    end

    return mu_full_population, q_full_population, Chi_full_population, mu_cav_population, q_cav_population, Chi_cav_population, converged
end




####################################################################################################################################################
########################################################## Fully connected graph ###################################################################
####################################################################################################################################################

# Function to compute all the quantities obtain by the sum of neighbours' terms in the cavity update
function sumpop_FC(
    mu_population::Vector{Float64}, 
    q_population::Vector{Float64}, 
    Chi_population::Vector{Float64}, 
    J_population::Vector{Float64}, 
    J_prime_population::Vector{Float64}, 
    i::Int, 
    P::Int)

    sum_mu = 0.0
    sum_q = 0.0
    sum_Chi = 0.0
    @inbounds @fastmath for j in 1:P
        if j == i
            continue
        end
        sum_mu += J_population[j] * mu_population[j]
        sum_q += J_population[j]^2 * q_population[j]
        sum_Chi += J_population[j] * J_prime_population[j] * Chi_population[j]
    end
    Ε = sqrt(sum_q)
    Δ = (1+sum_mu)/Ε

    testvalues(sum_mu, sum_q, sum_Chi, Ε, Δ)
    return sum_q, sum_Chi, Ε, Δ
end

# Function to run population dynamics with all the performance tips included
function population_dynamics_FC(
    P::Int, 
    tol, 
    max_iter::Int, 
    m::Float64, 
    σ²::Float64, 
    γ::Float64; 
    check_vars=Dict("avg_mu"=>0.0), 
    error_func=error_func, 
    check_conv=30, 
    rng=Xoshiro(1234), 
    verbose=false)

    # Initialize populations
    mu_population = rand(rng, P)
    q_population = rand(rng, P)
    Chi_population = rand(rng, P)
    J_population = zeros(P)
    J_prime_population = zeros(P)

    converged = false
    # Loop over iterations until convergence or max_iter is reached
    @showprogress for t in 1:max_iter

        # Update each site in the population
        @inbounds for i in shuffle(rng, 1:P)
            @inbounds for j in 1:P
                if j == i
                    continue
                end
                J_population[j], J_prime_population[j] = sample_couplings(rng, m, σ², γ, P)
            end
            # Compute the new values for mu, q, chi using the update functions
            sum_q, sum_Chi, Ε, Δ = sumpop_FC(mu_population, q_population, Chi_population, J_population, J_prime_population, i, P)
            mu_population[i] = f_mu(sum_Chi, Ε, Δ)
            q_population[i] = f_q(sum_q, sum_Chi, Δ)
            Chi_population[i] = f_Chi(sum_Chi, Δ)
            testvalues(mu_population[i], q_population[i], Chi_population[i], sum_q, sum_Chi, Δ)  
        end

        converged = error_func(check_vars, mu_population, q_population, Chi_population, tol, t, check_conv)

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

    return mu_population, q_population, Chi_population, converged
end