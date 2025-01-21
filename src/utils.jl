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
    max_diff = abs(check_vars["avg_μ"] - new_avg_μ)
    check_vars["avg_μ"] = new_avg_μ

    return max_diff
end


########################## MCMC ###########################

# Define the Random-Lotka-Volterra system of equations
function glv!(du, u, p, t)  # p = (J, zero_threshold)
    J = p
    mul!(du, J, u)
    du .= u .* (1 .- u .+ du)
end

function jac_glv!(Jac, u, p, t)
    J = p
    # Fill the diagonal
    @inbounds @fastmath @simd for i in 1:length(u)
        Jac[i, i] = 1 - 2 * u[i]
    end
    # Fill the off-diagonal elements
    @inbounds @fastmath for j in 1:length(u)
        @inbounds @fastmath for k in nzrange(J, j)
            Jac[J.rowval[k], j] = J.nzval[k] * u[j]
        end
    end
end

"""
    sample_glv(
        J::SparseMatrixCSC{Float64, Int}, 
        x0::Vector{Float64}, 
        tmax::Float64, 
        tsave::Vector{Float64}, 
        zero_threshold::Float64)

Simulates the Generalized Lotka-Volterra system.

Arguments:
- J: Transposed sparse interaction matrix (NxN), where J[j, i] is the interaction strength from species i to species j.
- x0: Initial abundances (Vector of size N).
- tmax: End time for the simulation.
- tsave: Vector of times for saving trajectories.
- zero_threshold: Threshold below which abundances are set to zero.

Returns:
- t_vals: Time points where trajectories are saved.
- trajectories: Matrix of size (N x length(t_vals)) storing species abundances.
"""
function sample_glv(
            J::SparseMatrixCSC{Float64, Int}, 
            x0::Vector{Float64}, 
            tmax::Float64, 
            tsave::Vector{Float64})
            
    # Ensure the initial condition has the correct size
    @assert size(J, 1) == size(J, 2) "Interaction matrix J must be square."
    N = size(J, 1)
    @assert length(x0) == N "Initial condition x0 must have size N."

    # Problem setup
    tspan = (0.0, tmax)
    p = J
    Jac = deepcopy(J)
    @inbounds @fastmath @simd for i in 1:N
        Jac[i, i] = 1.0
    end
    f! = ODEFunction(glv!, jac=jac_glv!, jac_prototype=Jac)
    prob = ODEProblem(f!, x0, tspan, p)

    # Solver options
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=tsave)

    # Extract the time points and trajectories
    t_vals = sol.t
    trajectories = hcat(sol.u...) # Convert solution vectors to a matrix

    return t_vals, trajectories, sol
end

