# Shortname for functions used in cavity update
function φ_func(x::Float64)
    return exp(-x^2/2)/sqrt(2*pi)
end

function Φ_func(x::Float64)
    return (1+erf(x/sqrt(2)))/2
end

# Function to compute all the quantities obtain by the sum of neighbours' terms in the cavity update
function sumpop(
    μ_population::Vector{Float64}, 
    q_population::Vector{Float64}, 
    χ_population::Vector{Float64}, 
    J_population::Vector{Float64}, 
    J_prime_population::Vector{Float64}, 
    neighbors_indices::Vector{Int})
    
    sum_μ = 0.0
    sum_q = 0.0
    sum_χ = 0.0
    @inbounds @fastmath @simd for j in neighbors_indices
        sum_μ += J_population[j] * μ_population[j]
        sum_q += J_population[j]^2 * q_population[j]
        sum_χ += J_population[j] * J_prime_population[j] * χ_population[j]
    end
    Ε = sqrt(sum_q)
    Δ = (1+sum_μ)/Ε

    testvalues(sum_μ, sum_q, sum_χ, Ε, Δ)
    return sum_q, sum_χ, Ε, Δ
end

# Placeholder update functions, where you can optimize the internal logic of f_mu, f_q, f_chi
function f_μ(sum_χ::Float64, Ε::Float64, Δ::Float64)
    if sum_χ < 1.0
        return Ε / (1 - sum_χ) * (Δ * Φ_func(Δ) + φ_func(Δ))
    else
        return Ε / (1 - sum_χ) * (Δ * Φ_func(-Δ) - φ_func(-Δ))
    end
end

function f_q(sum_q::Float64, sum_χ::Float64, Δ::Float64)
    if sum_χ < 1.0
        return sum_q / (1 - sum_χ)^2 * ((1+Δ^2) * Φ_func(Δ) + Δ * φ_func(Δ))
    else
        return sum_q / (1 - sum_χ)^2 * ((1+Δ^2) * Φ_func(-Δ) - Δ * φ_func(-Δ))
    end
end

function f_χ(sum_χ::Float64, Δ::Float64)
    if sum_χ < 1.0
        return 1 / (1 - sum_χ) * Φ_func(Δ)
    else
        return 1 / (1 - sum_χ) * Φ_func(-Δ)
    end
end

function update_cav!(p_cav_k, P, m, σ², γ, K, rng, μ_cav_population, q_cav_population, χ_cav_population, J_population, J_prime_population)
    # Update each site in the population
    @inbounds for i in shuffle(rng, 1:P)
        # Sample the degree k
        k_cav = sample_degree(rng, 1:length(p_cav_k), p_cav_k)

        # Get k random neighbors
        neighbors_indices_cav = sample(rng, filter!(x->x≠i,collect(1:P)), k_cav; replace=false)
        
        # CAVITY UPDATE
        # Sample k_cav pairs of correlated J, J' values
        @inbounds for j in neighbors_indices_cav
            J_population[j], J_prime_population[j] = sample_couplings(rng, m, σ², γ, K)
        end

        # Compute the new values for mu, q, chi using the update functions
        sum_q_cav, sum_χ_cav, Ε_cav, Δ_cav = sumpop(μ_cav_population, q_cav_population, χ_cav_population, J_population, J_prime_population, neighbors_indices_cav)
        μ_cav_population[i] = f_μ(sum_χ_cav, Ε_cav, Δ_cav)
        q_cav_population[i] = f_q(sum_q_cav, sum_χ_cav, Δ_cav)
        χ_cav_population[i] = f_χ(sum_χ_cav, Δ_cav)
        testvalues(μ_cav_population[i], q_cav_population[i], χ_cav_population[i], sum_q_cav, sum_χ_cav, Δ_cav)
    end
end

function update_full!(p_k, P, m, σ², γ, K, rng, μ_cav_population, q_cav_population, χ_cav_population, J_population, J_prime_population, μ_full_population, q_full_population, χ_full_population)
    # Update each site in the population
    @inbounds for i in shuffle(rng, 1:P)
        k_full = sample_degree(rng, 1:length(p_k), p_k)
        neighbors_indices_full = sample(rng, filter!(x->x≠i,collect(1:P)), k_full; replace=false)

        # FULL UPDATE
        # Sample k_cav pairs of correlated J, J' values
        @inbounds for j in neighbors_indices_full
            J_population[j], J_prime_population[j] = sample_couplings(rng, m, σ², γ, K)
        end
        # Compute the new values for mu, q, chi using the update functions
        sum_q_full, sum_χ_full, Ε_full, Δ_full = sumpop(μ_cav_population, q_cav_population, χ_cav_population, J_population, J_prime_population, neighbors_indices_full)
        μ_full_population[i] = f_μ(sum_χ_full, Ε_full, Δ_full)
        q_full_population[i] = f_q(sum_q_full, sum_χ_full, Δ_full)
        χ_full_population[i] = f_χ(sum_χ_full, Δ_full)
        testvalues(μ_full_population[i], q_full_population[i], χ_full_population[i], sum_q_full, sum_χ_full, Δ_full) 
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
    check_vars=Dict("avg_μ"=>0.0), 
    error_func=error_func, 
    check_conv=30, 
    rng=Xoshiro(1234), 
    verbose=false,
    plothist=false)

    # Initialize populations
    μ_cav_population = rand(rng, P)
    q_cav_population = rand(rng, P)
    χ_cav_population = rand(rng, P)
    μ_full_population = zeros(P)
    q_full_population = zeros(P)
    χ_full_population = zeros(P)
    J_population = zeros(P)
    J_prime_population = zeros(P)

    if plothist
        fig, axs = plt.subplots(1,3,figsize=(15, 8))
        axs[1].set_title("Histogram of μ values")
        axs[2].set_title("Histogram of q values")
        axs[3].set_title("Histogram of χ values")
        axs[1].set_xlabel("μ")
        axs[2].set_xlabel("q")
        axs[3].set_xlabel("χ")
        axs[1].set_ylabel("Frequency")
    end

    converged = false
    # Loop over iterations until convergence or max_iter is reached
    @showprogress for t in 1:max_iter

        update_cav!(p_cav_k, P, m, σ², γ, K, rng, μ_cav_population, q_cav_population, χ_cav_population, J_population, J_prime_population)

        converged = error_func(check_vars, μ_cav_population, q_cav_population, χ_cav_population, tol, t, check_conv)

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
                update_full!(p_k, P, m, σ², γ, K, rng, μ_cav_population, q_cav_population, χ_cav_population, J_population, J_prime_population, μ_full_population, q_full_population, χ_full_population)
                axs[1].hist(μ_full_population, bins=20, alpha=0.5)
                axs[2].hist(q_full_population, bins=20, alpha=0.5)
                axs[3].hist(χ_full_population, bins=20, alpha=0.5)
            end
        end
    end

    update_full!(p_k, P, m, σ², γ, K, rng, μ_cav_population, q_cav_population, χ_cav_population, J_population, J_prime_population, μ_full_population, q_full_population, χ_full_population)

    if !converged && verbose
        println("Reached max iterations without convergence.")
    end

    return μ_full_population, q_full_population, χ_full_population, μ_cav_population, q_cav_population, χ_cav_population, converged
end




#######################################################
################ Fully connected graph ################
#######################################################
# Function to compute all the quantities obtain by the sum of neighbours' terms in the cavity update
function sumpop_FC(
    μ_population::Vector{Float64}, 
    q_population::Vector{Float64}, 
    χ_population::Vector{Float64}, 
    J_population::Vector{Float64}, 
    J_prime_population::Vector{Float64}, 
    i::Int, 
    P::Int)

    sum_μ = 0.0
    sum_q = 0.0
    sum_χ = 0.0
    @inbounds @fastmath for j in 1:P
        if j == i
            continue
        end
        sum_μ += J_population[j] * μ_population[j]
        sum_q += J_population[j]^2 * q_population[j]
        sum_χ += J_population[j] * J_prime_population[j] * χ_population[j]
    end
    Ε = sqrt(sum_q)
    Δ = (1+sum_μ)/Ε

    testvalues(sum_μ, sum_q, sum_χ, Ε, Δ)
    return sum_q, sum_χ, Ε, Δ
end

# Function to run population dynamics with all the performance tips included
function population_dynamics_FC(
    P::Int, 
    tol, 
    max_iter::Int, 
    m::Float64, 
    σ²::Float64, 
    γ::Float64; 
    check_vars=Dict("avg_μ"=>0.0), 
    error_func=error_func, 
    check_conv=30, 
    rng=Xoshiro(1234), 
    verbose=false)

    # Initialize populations
    μ_population = rand(rng, P)
    q_population = rand(rng, P)
    χ_population = rand(rng, P)
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
            sum_q, sum_χ, Ε, Δ = sumpop_FC(μ_population, q_population, χ_population, J_population, J_prime_population, i, P)
            μ_population[i] = f_μ(sum_χ, Ε, Δ)
            q_population[i] = f_q(sum_q, sum_χ, Δ)
            χ_population[i] = f_χ(sum_χ, Δ)
            testvalues(μ_population[i], q_population[i], χ_population[i], sum_q, sum_χ, Δ)  
        end

        converged = error_func(check_vars, μ_population, q_population, χ_population, tol, t, check_conv)

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

    return μ_population, q_population, χ_population, converged
end