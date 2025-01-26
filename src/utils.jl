# Function to sample correlated Gaussian random variables J and J'
function sample_couplings(rng, m::Float64, sigma2::Float64, gamma::Float64, K::Int)
    u, v = randn(rng, 2)
    J = m/K + sqrt(sigma2/K)*u
    J_prime = m/K + sqrt(sigma2/K)*(gamma*u + sqrt(1-gamma^2)*v)
    return J, J_prime
end

function sample_degree(rng::AbstractRNG, p_k::Vector{Float64})
    w = Weights(p_k)
    return sample(rng, w) - 1
end

function sample_neighs!(rng, neigh_idxs, i, k, P)
    for j in 1:k
        check = true
        while check
            neigh_idxs[i] = rand(rng, 1:P)
            check = (neigh_idxs[j]==i)
        end
    end
end

function testvalues(sum_mu, sum_q, sum_chi, Epsilon, Delta)
    if sum_q < 0 || sum_chi == 1|| !isfinite(sum_mu) || !isfinite(sum_q) || !isfinite(sum_chi) || !isfinite(Epsilon) || !isfinite(Delta)
        println("sum_mu=$(sum_mu), sum_q=$(sum_q), sum_chi=$(sum_chi), Epsilon=$(Epsilon), Delta=$(Delta)")
        throw(ArgumentError("Invalid values"))
    end
end

function testvalues(mu, q, chi, sum_q, sum_chi, Delta)
    if mu < 0 || q < 0 || !isfinite(mu) || !isfinite(q) || !isfinite(chi)
        println("mu=$(mu), q=$(q), chi=$(chi), sum_q=$(sum_q), sum_chi=$(sum_chi), Delta=$(Delta)")
        throw(ArgumentError("Invalid values"))
    end
end


function error_func(check_vars, mu_population, q_population, chi_population)
    # Calculate new averages for convergence checking
    new_avg_mu = mean(mu_population)

    # Calculate the maximum change in the updates for convergence checking
    max_diff = abs(check_vars["avg_mu"] - new_avg_mu)
    check_vars["avg_mu"] = new_avg_mu

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


function sample_x(m::Float64, sigma2::Float64, gamma::Float64, K::Int, p_k::Vector{Float64}, mu_pop::Vector{Float64}, q_pop::Vector{Float64}, chi_pop::Vector{Float64}, nsim::Int, P::Int, rng::AbstractRNG)
    xvec = zeros(nsim * P)
    neigh_idxs = zeros(Int, length(p_k))
    @inbounds @fastmath for i in 1:P
        @inbounds @fastmath for isim in 1:nsim
            k = sample(rng, 0:length(p_k)-1, Weights(p_k))
            if k == 0
                xvec[(i-1)*nsim+isim] = 1.0
                continue
            end
            sample_neighs!(rng, neigh_idxs, i, k, P)
            sum_mu = 0.0
            sum_q = 0.0
            sum_chi = 0.0
            @inbounds @fastmath for neigh_idx in 1:k
                j = neigh_idxs[neigh_idx]
                J, Jprime = sample_couplings(rng, m, sigma2, gamma, K)
                sum_mu += J * mu_pop[j]
                sum_q += J^2 * q_pop[j]
                sum_chi += J * Jprime * chi_pop[j]
            end
            xvec[(i-1)*nsim+isim] = max((1+sum_mu+sqrt(sum_q)*randn(rng))/(1-sum_chi), 0.0)
        end
    end
    return xvec
end


function sample_x(m::Float64, sigma2::Float64, gamma::Float64, K::Int, p_k::Vector{Float64}, mu_avg::Float64, q_avg::Float64, chi_avg::Float64, nsim::Int, P::Int, rng::AbstractRNG)
    xvec = zeros(nsim * P)
    @inbounds @fastmath for i in 1:P
        @inbounds @fastmath for isim in 1:nsim
            k = sample(rng, 0:length(p_k)-1, Weights(p_k))
            if k == 0
                xvec[(i-1)*nsim+isim] = 1.0
                continue
            end
            sum_mu = 0.0
            sum_q = 0.0
            sum_chi = 0.0
            @inbounds @fastmath for _ in 1:k
                J, Jprime = sample_couplings(rng, m, sigma2, gamma, K)
                sum_mu += J
                sum_q += J^2
                sum_chi += J * Jprime
            end
            sum_mu *= mu_avg
            sum_q *= q_avg
            sum_chi *= chi_avg
            xvec[(i-1)*nsim+isim] = max((1+sum_mu+sqrt(sum_q)*randn(rng))/(1-sum_chi), 0.0)
        end
    end
    return xvec
end
