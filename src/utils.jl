# Function to sample correlated Gaussian random variables J and J'
function sample_couplings(rng, m::Float64, σ²::Float64, γ::Float64, K::Int)
    u, v = randn(rng, 2)
    J = m/K + sqrt(σ²/K)*u
    J_prime = m/K + sqrt(σ²/K)*(γ*u + sqrt(1-γ^2)*v)
    return J, J_prime
end

function sample_degree(rng::AbstractRNG, ks::AbstractRange, p_k::Vector{Float64})
    w = Weights(p_k)
    return sample(rng, ks, w)
end

function testvalues(sum_μ, sum_q, sum_χ, Ε, Δ)
    if sum_q <= 0 || sum_χ == 1|| !isfinite(sum_μ) || !isfinite(sum_q) || !isfinite(sum_χ) || !isfinite(Ε) || !isfinite(Δ)
        println("sum_μ=$(sum_μ), sum_q=$(sum_q), sum_χ=$(sum_χ), Ε=$(Ε), Δ=$(Δ)")
        throw(ArgumentError("Invalid values"))
    end
end

function testvalues(μ, q, χ, sum_q, sum_χ, Δ)
    if μ < 0 || q <= 0 || !isfinite(μ) || !isfinite(q) || !isfinite(χ)
        println("μ=$(μ), q=$(q), χ=$(χ), sum_q=$(sum_q), sum_χ=$(sum_χ), Δ=$(Δ)")
        throw(ArgumentError("Invalid values"))
    end
end


function error_func(check_vars, μ_population, q_population, χ_population)
    # Calculate new averages for convergence checking
    new_avg_μ = mean(μ_population)

    # Calculate the maximum change in the updates for convergence checking
    max_diff = abs(check_vars.avg_μ - new_avg_μ)
    check_vars.avg_μ = new_avg_μ

    return max_diff
end


struct CheckVars 
    avg_μ::Float64
end