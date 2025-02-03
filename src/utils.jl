"""
    PdfDegVec

Structure to store the degree distribution of a network.

# Fields

$(TYPEDFIELDS)

"""
struct PdfDegVec
    """Probability distribution of the degree."""
    pdf::Vector{Float64}
    """Vector of degrees."""
    deg::Vector{Int}
    """Minimum degree."""
    kmin::Int
    """Maximum degree."""
    kmax::Int
    """Average degree."""
    K::Union{Float64,Int64}
    """Dictionary to map degree to index."""
    index_dict::Dict{Int, Int}  # Store indices instead of pdf values
    @doc """
        PdfDegVec(pdf_deg::Function, deg::Vector{Int})

    Constructs a `PdfDegVec` structure from a degree distribution function.

    Arguments:
    - `pdf_deg::Function`: Function that returns the probability of a given degree.
    - `deg::Vector{Int}`: Vector of degrees.

    Returns:
    - `PdfDegVec`: Degree distribution structure.
    """
    function PdfDegVec(pdf_deg::Function, deg::Vector{Int})
        pdf_vals = pdf_deg.(deg)
        K = sum(pdf_vals .* deg)
        index_map = Dict(deg .=> eachindex(deg))  # Map degree to index
        new(pdf_vals, deg, minimum(deg), maximum(deg), K, index_map)
    end

    @doc """
        PdfDegVec(pdf_deg::Function, deg::Vector{Int}, K::Union{Float64,Int64})

    Constructs a `PdfDegVec` structure from a degree distribution function.

    Arguments:
    - `pdf_deg::Function`: Function that returns the probability of a given degree.
    - `deg::Vector{Int}`: Vector of degrees.
    - `K::Union{Float64,Int64}`: Average degree.

    Returns:
    - `PdfDegVec`: Degree distribution structure.
    """
    function PdfDegVec(pdf_deg::Function, deg::Vector{Int}, K::Union{Float64,Int64})
        pdf_vals = pdf_deg.(deg)
        index_map = Dict(deg .=> eachindex(deg))  # Map degree to index
        new(pdf_vals, deg, minimum(deg), maximum(deg), K, index_map)
    end
end

"""
    get_index(pdv::PdfDegVec, k::Int)

Returns the index of a given degree in the degree distribution.

Arguments:
- `pdv::PdfDegVec`: Degree distribution structure.
- `k::Int`: Degree.

Returns:
- `idx`: Index of the degree.
"""
function get_index(pdv::PdfDegVec, k::Int)
    return get(pdv.index_dict, k, 0)  # Returns 0 if k is not found
end

# Function to sample correlated Gaussian random variables J and J'
"""
    sample_couplings(rng::AbstractRNG, m::Float64, sigma2::Float64, gamma::Float64, K::Union{Int,Float64})

Samples correlated Gaussian random variables J and J'.

Arguments:
- `rng::AbstractRNG`: Random number generator.
- `m::Float64`: Mean of the Gaussian distribution.
- `sigma2::Float64`: Variance of the Gaussian distribution.
- `gamma::Float64`: Correlation coefficient between J and J'.
- `K::Union{Int,Float64}`: Average degree.

Returns:
- `J`: Sampled random variable J.
- `J_prime`: Sampled random variable J'.
"""
function sample_couplings(rng, m::Float64, sigma2::Float64, gamma::Float64, K::Union{Int,Float64})
    u, v = randn(rng, 2)
    J = m/K + sqrt(sigma2/K)*u
    J_prime = m/K + sqrt(sigma2/K)*(gamma*u + sqrt(1-gamma^2)*v)
    return J, J_prime
end

# Function to sample a degree from a degree distribution
"""
    sample_degree(rng::AbstractRNG, p_k::PdfDegVec)

Samples a degree from a degree distribution.

Arguments:
- `rng::AbstractRNG`: Random number generator.
- `p_k::PdfDegVec`: Degree distribution structure.

Returns:
- `k`: Sampled degree.
"""
function sample_degree(rng::AbstractRNG, p_k::PdfDegVec)
    if length(p_k.pdf) == 1
        return p_k.deg[1]
    else
        w = Weights(p_k.pdf)
        return sample(rng, p_k.deg, w)
    end
end

# Function to sample neighbors
"""
    sample_neighs!(rng::AbstractRNG, neigh_idxs::Vector{Int}, i::Int, k::Int, P::Int)

Samples k neighbors indices from a population of P nodes.

Arguments:
- `rng::AbstractRNG`: Random number generator.
- `neigh_idxs::Vector{Int}`: Vector to store the sampled neighbors.
- `i::Int`: Index of the node.
- `k::Int`: Number of neighbors to sample.
- `P::Int`: Number of nodes in the population.
"""
function sample_neighs!(rng::AbstractRNG, neigh_idxs::Vector{Int}, i::Int, k::Int, P::Int)
    @inbounds for j in 1:k
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


function error_func(check_vars::Dict{String, Float64}, mu_pop::Vector{Float64}, q_pop::Vector{Float64}, chi_pop::Vector{Float64}, tol::Dict{String, Float64})
    # Calculate new averages for convergence checking
    new_avg_mu = mean(mu_pop)

    # Calculate the maximum change in the updates for convergence checking
    max_diff_avg = abs(check_vars["avg_mu"] - new_avg_mu)

    check_vars["avg_mu"] = new_avg_mu

    return (max_diff_avg < tol["avg"])
end


########################## MCMC ###########################

# Define the Random-Lotka-Volterra system of equations
function glv!(du, u, p, t)  # p = (J, zero_threshold)
    mul!(du, p, u)
    du .= u .* (1 .- u .+ du)
end

function glv_jac!(Jac, u, p, t)
    N = length(u)
    @inbounds for col in 1:N   # Iterate over columns
        @inbounds for idx in nzrange(Jac, col)  # Get index range for this column
            row = Jac.rowval[idx]  # Get row index
            if row != col
                Jac[row,col] = p[row,col] * u[row]
            else
                summed = 0.0
                @inbounds for k in 1:N
                    summed += p[row,k] * u[k]
                end
                Jac[row,row] = 1 - 2 * u[row] + summed
            end
        end
    end
end


"""
    sample_glv(J::SparseMatrixCSC{Float64, Int}, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64})

Simulates the Generalized Lotka-Volterra system on sparse networks.

Arguments:
- `J::SparseMatrixCSC{Float64, Int}`: Sparse interaction matrix (NxN), where J[i,j] is the interaction strength from species j to species i.
- `x0::Vector{Float64}`: Initial abundances (Vector of size N).
- `tmax::Float64`: End time for the simulation.
- `tsave::Vector{Float64}`: Vector of times for saving trajectories.

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
    f! = ODEFunction(glv!, jac_prototype=deepcopy(J), jac=glv_jac!)
    prob = ODEProblem(f!, x0, tspan, p)

    # Solver options
    sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=tsave)

    # Extract the time points and trajectories
    t_vals = sol.t
    trajectories = hcat(sol.u...) # Convert solution vectors to a matrix

    return t_vals, trajectories, sol
end

"""
    sample_glv(J::Matrix{Float64}, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64})

Simulates the Generalized Lotka-Volterra system on fully connected networks.

Arguments:
- `J::Matrix{Float64}`: Fully connected interaction matrix (NxN), where J[i,j] is the interaction strength from species j to species i.
- `x0::Vector{Float64}`: Initial abundances (Vector of size N).
- `tmax::Float64`: End time for the simulation.
- `tsave::Vector{Float64}`: Vector of times for saving trajectories.

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
    f! = ODEFunction(glv!, jac_prototype=deepcopy(J), jac=glv_jac!)
    prob = ODEProblem(f!, x0, tspan, p)

    # Solver options
    sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=tsave)

    # Extract the time points and trajectories
    t_vals = sol.t
    trajectories = hcat(sol.u...) # Convert solution vectors to a matrix

    return t_vals, trajectories, sol
end


"""
    sample_glv(J::SparseMatrixCSC{Float64, Int}, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64}, zero_threshold::Float64)

Simulates the Generalized Lotka-Volterra system for sparse networks. It sets to zero the abundances that are below a certain threshold.

Arguments:
- `J::SparseMatrixCSC{Float64, Int}`: Sparse interaction matrix (NxN), where J[i,j] is the interaction strength from species j to species i.
- `x0::Vector{Float64}`: Initial abundances (Vector of size N).
- `tmax::Float64`: End time for the simulation.
- `tsave::Vector{Float64}`: Vector of times for saving trajectories.
- `zero_threshold::Float64`: Threshold below which abundances are set to zero.

Returns:
- t_vals: Time points where trajectories are saved.
- trajectories: Matrix of size (N x length(t_vals)) storing species abundances.x
"""
function sample_glv(J::SparseMatrixCSC{Float64, Int}, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64}, zero_threshold::Float64)
            
    # Ensure the initial condition has the correct size
    @assert size(J, 1) == size(J, 2) "Interaction matrix J must be square."
    N = size(J, 1)
    @assert length(x0) == N "Initial condition x0 must have size N."

    # Problem setup
    tspan = (0.0, tmax)
    p = J
    f! = ODEFunction(glv!, jac_prototype=deepcopy(J), jac=glv_jac!)
    prob = ODEProblem(f!, x0, tspan, p)

    # Solver options
    sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=tsave)

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
- `J::Matrix{Float64}`: Fully connected interaction matrix (NxN), where J[i,j] is the interaction strength from species j to species i.
- `x0::Vector{Float64}`: Initial abundances (Vector of size N).
- `tmax::Float64`: End time for the simulation.
- `tsave::Vector{Float64}`: Vector of times for saving trajectories.
- `zero_threshold::Float64`: Threshold below which abundances are set to zero.

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
    f! = ODEFunction(glv!, jac_prototype=deepcopy(J), jac=glv_jac!)
    prob = ODEProblem(f!, x0, tspan, p)

    # Solver options
    sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=tsave)

    # Extract the time points and trajectories
    t_vals = sol.t
    trajectories = hcat(sol.u...) # Convert solution vectors to a matrix
    trajectories[trajectories .< zero_threshold] .= 0.0

    return t_vals, trajectories, sol
end


"""
    sample_x(m::Float64, sigma2::Float64, gamma::Float64, p_k::PdfDegVec, mu_pop::Vector{Float64}, q_pop::Vector{Float64}, chi_pop::Vector{Float64}, nsim::Int; rng::AbstractRNG=Xoshiro(1234), zero_threshold::Float64=0.0)

Samples the abundances of a population of nodes in a network with correlated Gaussian couplings.

Arguments:
- `m::Float64`: Mean of the Gaussian distribution.
- `sigma2::Float64`: Variance of the Gaussian distribution.
- `gamma::Float64`: Correlation coefficient between J and J'.
- `p_k::PdfDegVec`: Degree distribution of the network.
- `mu_pop::Vector{Float64}`: Mean abundance for each node.
- `q_pop::Vector{Float64}`: Variance of abundances for each node.
- `chi_pop::Vector{Float64}`: Correlation of abundances for each node.
- `nsim::Int`: Number of simulations to run.

Keyword Arguments:
- `rng::AbstractRNG`: Random number generator (default Xoshiro(1234)).
- `zero_threshold::Float64`: Threshold below which abundances are set to zero (default 0.0).

Returns:
- `xvec`: Vector of sampled abundances.
"""
function sample_x(m::Float64, sigma2::Float64, gamma::Float64, p_k::PdfDegVec, mu_pop::Vector{Float64}, q_pop::Vector{Float64}, chi_pop::Vector{Float64}, nsim::Int; rng::AbstractRNG=Xoshiro(1234), zero_threshold::Float64=0.0)
    @assert length(mu_pop) == length(q_pop) == length(chi_pop) "Population vectors must have the same length." 
    K = p_k.K
    P = length(mu_pop)
    xvec = zeros(nsim * P)
    neigh_idxs = zeros(Int, p_k.kmax)
    @inbounds for i in 1:P
        @inbounds for isim in 1:nsim
            k = sample_degree(rng, p_k)
            if k == 0
                xvec[(i-1)*nsim+isim] = 1.0
                continue
            end
            sample_neighs!(rng, neigh_idxs, i, k, P)
            sum_mu = 0.0
            sum_q = 0.0
            sum_chi = 0.0
            @inbounds @simd for neigh_idx in 1:k
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
    sample_x(m::Float64, sigma2::Float64, gamma::Float64, p_k::PdfDegVec, mu_avg::Float64, q_avg::Float64, chi_avg::Float64, nsim::Int; rng::AbstractRNG=Xoshiro(1234), zero_threshold::Float64=0.0)

Samples the abundances of a population of nodes in a network with correlated Gaussian couplings.

Arguments:
- `m::Float64`: Mean of the Gaussian distribution.
- `sigma2::Float64`: Variance of the Gaussian distribution.
- `gamma::Float64`: Correlation coefficient between J and J'.
- `p_k::PdfDegVec`: Degree distribution of the network.
- `mu_avg::Float64`: Mean abundance for each node.
- `q_avg::Float64`: Variance of abundances for each node.
- `chi_avg::Float64`: Correlation of abundances for each node.
- `nsim::Int`: Number of simulations to run.

Keyword Arguments:
- `rng::AbstractRNG`: Random number generator (default Xoshiro(1234)).
- `zero_threshold::Float64`: Threshold below which abundances are set to zero (default 0.0).

Returns:
- `xvec`: Vector of sampled abundances.
"""
function sample_x(m::Float64, sigma2::Float64, gamma::Float64, p_k::PdfDegVec, mu_avg::Float64, q_avg::Float64, chi_avg::Float64, nsim::Int; rng::AbstractRNG=Xoshiro(1234), zero_threshold::Float64=0.0)
    K = p_k.K
    xvec = zeros(nsim)
    @inbounds for i in 1:P
        @inbounds for isim in 1:nsim
            k = sample_degree(rng, p_k)
            if k == 0
                xvec[(i-1)*nsim+isim] = 1.0
                continue
            end
            sum_mu = 0.0
            sum_q = 0.0
            sum_chi = 0.0
            @inbounds @simd for _ in 1:k
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
