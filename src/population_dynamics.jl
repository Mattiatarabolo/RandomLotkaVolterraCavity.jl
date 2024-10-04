# Function to sample correlated Gaussian random variables J and J'
function sample_couplings(rng, m::Float64, σ::Float64, γ::Float64, N::Int)
    J = randn(rng) * σ / sqrt(N) + m / N
    J_prime = γ * J + sqrt(1 - γ^2) * randn(rng) * σ / sqrt(N) + m / N
    return J, J_prime
end

# Precompute the degree distribution cumulative sum for efficient sampling
function precompute_cdf(p_k::Vector{Float64})
    return cumsum(p_k)
end

function sample_degree(rng, cdf_p_k::Vector{Float64})
    u = rand(rng)
    return findfirst(x->x>=u, cdf_p_k)
end

# Shortname for functions used in cavity update
function φ_func(x::Float64)
    return exp(-x^2/2)/sqrt(2*pi)
end

function Φ_func(x::Float64)
    return (1+erf(x/sqrt(2)))/2
end

# Function to compute all the quantities obtain by the sum of neighbours' terms in the cavity update
function sumcav(μ_population::Vector{Float64}, q_population::Vector{Float64}, χ_population::Vector{Float64}, J_population::Vector{Float64}, J_prime_population::Vector{Float64}, neighbors_indices::Vector{Int})
    sum_μ = 0.0
    sum_q = 0.0
    sum_χ = 0.0
    @inbounds @simd for j in neighbors_indices
        sum_μ += J_population[j] * μ_population[j]
        sum_q += J_population[j]^2 * q_population[j]
        sum_χ += J_population[j] * J_prime_population[j] * χ_population[j]
    end
    Ε = sqrt(sum_q)
    Δ = (1+sum_μ)/Ε
    φ = φ_func(Δ)
    Φ = Φ_func(Δ)
    return sum_q, sum_χ, Ε, Δ, φ, Φ 
end

# Placeholder update functions, where you can optimize the internal logic of f_mu, f_q, f_chi
function f_μ(sum_χ::Float64, Ε::Float64, Δ::Float64, φ::Float64, Φ::Float64)
    return Ε / (1 - sum_χ) * (Δ * Φ + φ )
end

function f_q(sum_q::Float64, sum_χ::Float64, Δ::Float64, φ::Float64, Φ::Float64)
    return sum_q / (1 - sum_χ)^2 * ((1+Δ^2) * Φ + Δ * φ)
end

function f_χ(sum_χ::Float64, Φ::Float64)
    return 1 / (1 - sum_χ) * Φ
end

# Function to run population dynamics with all the performance tips included
function population_dynamics(p_k::Vector{Float64}, P::Int, tol::Float64, max_iter::Int, m::Float64, σ::Float64, γ::Float64, K::Int; rng=Xoshiro(1234))
    # Precompute degree distribution CDF
    cdf_p_k = precompute_cdf(p_k)

    # Initialize populations
    μ_population = rand(rng, P)
    q_population = rand(rng, P)
    χ_population = rand(rng, P)
    J_population = zeros(P)
    J_prime_population = zeros(P)

    # Initialize averages to check convergence
    avg_μ = 0.0
    avg_q = 0.0
    avg_χ = 0.0
    std_μ = 0.0
    std_q = 0.0
    std_χ = 0.0

    # Loop over iterations until convergence or max_iter is reached
    converged = false
    @showprogress for t in 1:max_iter

        max_diff = Inf

        # Update each site in the population
        @inbounds for i in shuffle(rng, 1:P)
            # Sample the degree k
            k = sample_degree(rng, cdf_p_k)

            # Get k random neighbors
            neighbors_indices = sample(rng, 1:P, k; replace=false)
            
            # Sample k pairs of correlated J, J' values
            @inbounds for j in neighbors_indices
                J_population[j], J_prime_population[j] = sample_couplings(rng, m, σ, γ, K)
            end

            # Compute the new values for mu, q, chi using the update functions
            sum_q, sum_χ, Ε, Δ, φ, Φ = sumcav(μ_population, q_population, χ_population, J_population, J_prime_population, neighbors_indices)
            μ_population[i] = f_μ(sum_χ, Ε, Δ, φ, Φ)
            q_population[i] = f_q(sum_q, sum_χ, φ, Δ, Φ)
            χ_population[i] = f_χ(sum_χ, Φ)
        end

        # Calculate new averages for convergence checking
        new_avg_μ = mean(μ_population)
        new_avg_q = mean(q_population)
        new_avg_χ = mean(χ_population)
        new_std_μ = std(μ_population; mean=new_avg_μ)
        new_std_q = std(q_population; mean=new_avg_q)
        new_std_χ = std(χ_population; mean=new_avg_χ)

        # Calculate the maximum change in the updates for convergence checking
        max_diff = max(max_diff, abs(avg_μ - new_avg_μ))
        max_diff = max(max_diff, abs(avg_q - new_avg_q))
        max_diff = max(max_diff, abs(avg_χ - new_avg_χ))
        max_diff = max(max_diff, abs(std_μ - new_std_μ))
        max_diff = max(max_diff, abs(std_q - new_std_q))
        max_diff = max(max_diff, abs(std_χ - new_std_χ))

        # Check for convergence
        if t % 10 == 0  # Check every 10 iterations
            if max_diff < tol
                converged = true
                println("Converged after $t iterations.")
                break
            end
        end
    end

    if !converged
        println("Reached max iterations without convergence.")
    end

    return μ_population, q_population, χ_population
end

#=
# Example usage
P = Int(1e4)  # Population size
K = 4  # Average degree of the graph
k_max = 19
p_k = [(k+1)*exp(-K)*K^k/factorial(k+1) for k in 1:k_max]  # This is the excess degree distribution, i.e. the cavity degree distribution of an Erdos-Renyi random graph
tol = 1e-5  # Convergence tolerance
max_iter = Int(1e6)  # Maximum number of iterations
m = 0.0  # Mean coupling
σ = 0.1  # Standard deviation of coupling
γ = -0.1  # Correlation coefficient between J and J'

μ_population, q_population, χ_population = population_dynamics(p_k, P, tol, max_iter, m, σ, γ, K)
=#