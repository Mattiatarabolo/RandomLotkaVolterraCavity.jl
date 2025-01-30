struct PdfDegVec
    pdf::Vector{Float64}
    deg::Vector{Int}
    kmin::Int
    kmax::Int
    index_dict::Dict{Int, Int}  # Store indices instead of pdf values
    
    function PdfDegVec(pdf_deg::Function, deg::Vector{Int})
        pdf_vals = pdf_deg.(deg)
        index_map = Dict(deg .=> eachindex(deg))  # Map degree to index
        new(pdf_vals, deg, minimum(deg), maximum(deg), index_map)
    end
end

function get_index(pdv::PdfDegVec, k::Int)
    return get(pdv.index_dict, k, 0)  # Returns 0 if k is not found
end

# Function to sample correlated Gaussian random variables J and J'
function sample_couplings(rng, m::Float64, sigma2::Float64, gamma::Float64, K::Int)
    u, v = randn(rng, 2)
    J = m/K + sqrt(sigma2/K)*u
    J_prime = m/K + sqrt(sigma2/K)*(gamma*u + sqrt(1-gamma^2)*v)
    return J, J_prime
end

function sample_degree(rng::AbstractRNG, p_k::PdfDegVec)
    if length(p_k.pdf) == 1
        return p_k.deg[1]
    else
        w = Weights(p_k.pdf)
        return sample(rng, p_k.deg, w)
    end
end

function sample_neighs!(rng::AbstractRNG, neigh_idxs::Vector{Int}, i::Int, k::Int, P::Int)
    for j in 1:k
        check = true
        while check
            neigh_idxs[j] = rand(rng, 1:P)
            check = (neigh_idxs[j]==i)
        end
    end
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


function error_func(check_vars::Dict{String, Float64}, mu_population::Vector{Float64}, q_population::Vector{Float64}, chi_population::Vector{Float64})
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

"""
    sample_glv(J::SparseMatrixCSC{Float64, Int}, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64})

Simulates the Generalized Lotka-Volterra system on sparse networks.

Arguments:
- J: Sparse interaction matrix (NxN), where J[i,j] is the interaction strength from species j to species i.
- x0: Initial abundances (Vector of size N).
- tmax: End time for the simulation.
- tsave: Vector of times for saving trajectories.

Returns:
- t_vals: Time points where trajectories are saved.
- trajectories: Matrix of size (N x length(t_vals)) storing species abundances.
"""
function sample_glv(J::SparseMatrixCSC{Float64, Int}, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64})
            
    # Ensure the initial condition has the correct size
    @assert size(J, 1) == size(J, 2) "Interaction matrix J must be square."
    N = size(J, 1)
    @assert length(x0) == N "Initial condition x0 must have size N."

    # Problem setup
    tspan = (0.0, tmax)
    p = J
    prob = ODEProblem(glv!, x0, tspan, p)

    # Solver options
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=tsave)

    # Extract the time points and trajectories
    t_vals = sol.t
    trajectories = hcat(sol.u...) # Convert solution vectors to a matrix

    return t_vals, trajectories, sol
end

"""
    sample_glv(J::Matrix{Float64}, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64})

Simulates the Generalized Lotka-Volterra system on fully connected networks.

Arguments:
- J: Fully connected nteraction matrix (NxN), where J[i,j] is the interaction strength from species j to species i.
- x0: Initial abundances (Vector of size N).
- tmax: End time for the simulation.
- tsave: Vector of times for saving trajectories.

Returns:
- t_vals: Time points where trajectories are saved.
- trajectories: Matrix of size (N x length(t_vals)) storing species abundances.
"""
function sample_glv(J::Matrix{Float64}, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64})
            
    # Ensure the initial condition has the correct size
    @assert size(J, 1) == size(J, 2) "Interaction matrix J must be square."
    N = size(J, 1)
    @assert length(x0) == N "Initial condition x0 must have size N."

    # Problem setup
    tspan = (0.0, tmax)
    p = J
    prob = ODEProblem(glv!, x0, tspan, p)

    # Solver options
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=tsave)

    # Extract the time points and trajectories
    t_vals = sol.t
    trajectories = hcat(sol.u...) # Convert solution vectors to a matrix

    return t_vals, trajectories, sol
end


"""
    sample_glv(J::SparseMatrixCSC{Float64, Int}, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64}, zero_threshold::Float64)

Simulates the Generalized Lotka-Volterra system for sparse networks. It sets to zero the abundances that are below a certain threshold.

Arguments:
- J: Sparse interaction matrix (NxN), where J[i,j] is the interaction strength from species j to species i.
- x0: Initial abundances (Vector of size N).
- tmax: End time for the simulation.
- tsave: Vector of times for saving trajectories.
- zero_threshold: Threshold below which abundances are set to zero.

Returns:
- t_vals: Time points where trajectories are saved.
- trajectories: Matrix of size (N x length(t_vals)) storing species abundances.
"""
function sample_glv(J::SparseMatrixCSC{Float64, Int}, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64}, zero_threshold::Float64)
            
    # Ensure the initial condition has the correct size
    @assert size(J, 1) == size(J, 2) "Interaction matrix J must be square."
    N = size(J, 1)
    @assert length(x0) == N "Initial condition x0 must have size N."

    # Problem setup
    tspan = (0.0, tmax)
    p = J
    prob = ODEProblem(glv!, x0, tspan, p)

    # Solver options
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=tsave)

    # Extract the time points and trajectories
    t_vals = sol.t
    trajectories = hcat(sol.u...) # Convert solution vectors to a matrix
    trajectories[trajectories .< zero_threshold] .= 0.0

    return t_vals, trajectories, sol
end


"""
    sample_glv(J::Matrix{Float64}, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64}, zero_threshold::Float64)

Simulates the Generalized Lotka-Volterra system for fully connected networks. It sets to zero the abundances that are below a certain threshold.

Arguments:
- J: Fully connected interaction matrix (NxN), where J[i,j] is the interaction strength from species j to species i.
- x0: Initial abundances (Vector of size N).
- tmax: End time for the simulation.
- tsave: Vector of times for saving trajectories.
- zero_threshold: Threshold below which abundances are set to zero.

Returns:
- t_vals: Time points where trajectories are saved.
- trajectories: Matrix of size (N x length(t_vals)) storing species abundances.
"""
function sample_glv(J::Matrix{Float64}, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64}, zero_threshold::Float64)
            
    # Ensure the initial condition has the correct size
    @assert size(J, 1) == size(J, 2) "Interaction matrix J must be square."
    N = size(J, 1)
    @assert length(x0) == N "Initial condition x0 must have size N."

    # Problem setup
    tspan = (0.0, tmax)
    p = J
    prob = ODEProblem(glv!, x0, tspan, p)

    # Solver options
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=tsave)

    # Extract the time points and trajectories
    t_vals = sol.t
    trajectories = hcat(sol.u...) # Convert solution vectors to a matrix
    trajectories[trajectories .< zero_threshold] .= 0.0

    return t_vals, trajectories, sol
end


"""
    sample_x(m::Float64, sigma2::Float64, gamma::Float64, K::Int, p_k::PdfDegVec, mu_pop::Vector{Float64}, q_pop::Vector{Float64}, chi_pop::Vector{Float64}, nsim::Int, P::Int, rng::AbstractRNG,zero_threshold::Float64)

Samples nsim fixed-point abundances of a random Generalized Lotka-Volterra system with correlated gaussian random interactions and degree distribution p_k. It uses the populations of the mean abundances, mean squared abundances and susceptibilities obtained from the population dynamics algorithm.

Arguments:
- m: Mean of the Gaussian distribution of the interactions.
- sigma2: Variance of the Gaussian distribution of the interactions.
- gamma: Correlation between the two interaction parameters Jᵢⱼ and Jⱼᵢ.
- K: Number of neighbors.
- p_k: Degree distribution. Vector of type PdfDegVec the probability of having k neighbors (p_k.pdf) with k ∈ p_k.deg.
- mu_pop: Vector of size P with the population of the mean abundances. It is obtained from the population dynamics algorithm.
- q_pop: Vector of size P with the population of the mean squared abundances. It is obtained from the population dynamics algorithm.
- chi_pop: Vector of size P with the population of the susceptibilities. It is obtained from the population dynamics algorithm.
- nsim: Number of simulations per element of the population.
- P: Size of the population in the population dynamics algorithm.
- rng: Random number generator.
- zero_threshold: Threshold below which abundances are set to zero.

Returns:
- xvec: Vector of size nsim*P with the fixed-point abundances.
"""
function sample_x(m::Float64, sigma2::Float64, gamma::Float64, K::Int, p_k::PdfDegVec, mu_pop::Vector{Float64}, q_pop::Vector{Float64}, chi_pop::Vector{Float64}, nsim::Int, P::Int, rng::AbstractRNG, zero_threshold::Float64)
    xvec = zeros(nsim * P)
    neigh_idxs = zeros(Int, p_k.kmax)
    for i in 1:P
        for isim in 1:nsim
            k = sample_degree(rng, p_k)
            if k == 0
                xvec[(i-1)*nsim+isim] = 1.0
                continue
            end
            sample_neighs!(rng, neigh_idxs, i, k, P)
            sum_mu = 0.0
            sum_q = 0.0
            sum_chi = 0.0
            for neigh_idx in 1:k
                j = neigh_idxs[neigh_idx]
                J, Jprime = sample_couplings(rng, m, sigma2, gamma, K)
                sum_mu += J * mu_pop[j]
                sum_q += J^2 * q_pop[j]
                sum_chi += J * Jprime * chi_pop[j]
            end
            xvec[(i-1)*nsim+isim] = max((1+sum_mu+sqrt(sum_q)*randn(rng))/(1-sum_chi), zero_threshold)
        end
    end
    return xvec
end

"""
    sample_x(m::Float64, sigma2::Float64, gamma::Float64, K::Int, p_k::PdfDegVec, mu_avg::Float64, q_avg::Float64, chi_avg::Float64, nsim::Int, P::Int, rng::AbstractRNG, zero_threshold::Float64)

Samples nsim fixed-point abundances of a random Generalized Lotka-Volterra system with correlated gaussian random interactions and degree distribution p_k. It uses the averages of the populations of the mean abundances, mean squared abundances and susceptibilities obtained from the population dynamics algorithm.

Arguments:
- m: Mean of the Gaussian distribution of the interactions.
- sigma2: Variance of the Gaussian distribution of the interactions.
- gamma: Correlation between the two interaction parameters Jᵢⱼ and Jⱼᵢ.
- K: Number of neighbors.
- p_k: Degree distribution. Vector of type PdfDegVec the probability of having k neighbors (p_k.pdf) with k ∈ p_k.deg.
- mu_avg: Average of the population of the mean abundances. It is obtained from the population dynamics algorithm.
- q_pop: Average of the population of the mean squared abundances. It is obtained from the population dynamics algorithm.
- chi_pop: Average of the population of the susceptibilities. It is obtained from the population dynamics algorithm.
- nsim: Number of simulations per element of the population.
- P: Size of the population in the population dynamics algorithm.
- rng: Random number generator.
- zero_threshold: Threshold below which abundances are set to zero.

Returns:
- xvec: Vector of size nsim*P with the fixed-point abundances.
"""
function sample_x(m::Float64, sigma2::Float64, gamma::Float64, K::Int, p_k::PdfDegVec, mu_avg::Float64, q_avg::Float64, chi_avg::Float64, nsim::Int, P::Int, rng::AbstractRNG, zero_threshold::Float64)
    xvec = zeros(nsim * P)
    for i in 1:P
        for isim in 1:nsim
            k = sample_degree(rng, p_k)
            if k == 0
                xvec[(i-1)*nsim+isim] = 1.0
                continue
            end
            sum_mu = 0.0
            sum_q = 0.0
            sum_chi = 0.0
            for _ in 1:k
                J, Jprime = sample_couplings(rng, m, sigma2, gamma, K)
                sum_mu += J
                sum_q += J^2
                sum_chi += J * Jprime
            end
            sum_mu *= mu_avg
            sum_q *= q_avg
            sum_chi *= chi_avg
            xvec[(i-1)*nsim+isim] = max((1+sum_mu+sqrt(sum_q)*randn(rng))/(1-sum_chi), zero_threshold)
        end
    end
    return xvec
end
