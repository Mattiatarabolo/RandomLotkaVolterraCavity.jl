"""
    sample_couplings(m::RT, sigma2::RT, corr::RT, K::RT; rng::AbstractRNG=Xoshiro(1234)) where {RT<:Real}

Sample couplings J and Jp from a bivariate Gaussian distribution with means `m/K`, variances `sigma2/K`, and correlation `corr`.

# Arguments
- `m::RT`: Mean value of the couplings.
- `sigma2::RT`: Variance of the couplings.
- `corr::RT`: Correlation between the couplings.
- `K::RT`: Average degree of the network (used for proper scaling of the couplings).

# Returns
- `J::RT`: Sampled coupling J.
- `Jp::RT`: Sampled coupling J'.

# Example
```julia
m = 1.0
sigma2 = 0.1
corr = 0.5

J, J_prime = sample_couplings(m, sigma2, corr, 10)
``` 
"""
function sample_couplings(m::RT, sigma2::RT, corr::RT, K::RT; rng::AbstractRNG=Xoshiro(1234)) where {RT<:Real}
    u, v = randn(rng, 2)
    J = m / K + sqrt(sigma2 / K) * u
    J_prime = m / K + sqrt(sigma2 / K) * (corr * u + sqrt(1 - corr ^ 2) * v)
    return J, J_prime
end

#################################################################################################################################
#################################################################################################################################

"""
    run_MC(model::Model{NK, I, RT, MT, D}, dt::RT; rng=Xoshiro(1234), showprogress=false, idxs_tsave=nothing, divergence_threshold=1e6) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, MT<:AbstractMatrix{RT}, D<:Distribution}

Run a Monte Carlo simulation of the model for a given time step `dt`. The function samples trajectories of the model and saves them at specified time indices.

# Arguments
- `model::Model{NK, I, RT, MT, D}`: The model to simulate, where `NK` is the noise kind, `I` is the integer type for indices, `RT` is the real type, `MT` is the matrix type, and `D` is the distribution type.
- `dt::RT`: Time step for the simulation.

# Optional arguments
- `rng::AbstractRNG`: Random number generator (default Xoshiro(1234)).
- `showprogress::Bool`: Whether to show a progress bar (default false). 
- `idxs_tsave::Union{Nothing, Vector{Int}}`: Indices at which to save the trajectory (default nothing, which saves at every time step).
- `divergence_threshold::RT`: Threshold for divergence detection (default 1e6).

# Output
- `traj::Matrix{RT}`: Matrix of sampled trajectories, where each column corresponds to a time point.
- `tsave::Vector{RT}`: Vector of time points at which the trajectories are saved.
- `convergence::Bool`: Boolean indicating whether the integration was successful (true) or diverged/failed (false).
"""
function run_MC(model::Model{NK, I, RT, MT, D}, dt::RT; rng=Xoshiro(1234), showprogress=false, idxs_tsave=nothing, divergence_threshold=1e6) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, MT<:AbstractMatrix{RT}, D<:Distribution}
    # Get the model parameters
    N, M, lam = model.N, model.M, model.lam

    # Initialize vector of indices to save values at (if not provided, save at every time step)
    if idxs_tsave === nothing
        idxs_tsave = 0:M # Save at every time step
    else
        filter!(n -> 0 <= n <= M, idxs_tsave) # Ensure indices are within bounds
    end
    tsave = idxs_tsave .* dt # Time points to save
    Msave = length(tsave) # Number of time points to save
    
    # Initialize storage for trajectories
    traj = fill(NaN, N, Msave)

    # Sample initial condition
    x0 = max.(rand(rng, model.p0, N), lam)
    
    # Initialize internal variables
    x = copy(x0) # Current state
    h = zeros(RT, N) # Vector to store J * x
    save_idx = 1 # Index for saving trajectories

    # Save initial condition if the time index is in idxs_tsave
    if 0 in idxs_tsave
        traj[:, save_idx] .= x
        save_idx += 1
    end

    # Initialize the progress bar
    p = Progress((N ^ 2 + 1) * M; enabled=showprogress, desc="Sampling trajectory")

    # Initialize convergence flag
    convergence = true
    
    # Loop over all time steps
    for n in 1:M
        converged = _sample!(x, h, model, dt, divergence_threshold, rng, p) # Sample the next state using the model's sampling function
        # Save the current state if the time index is in idxs_tsave
        if n in idxs_tsave
            traj[:, save_idx] .= x
            save_idx += 1
        end
        # Stop the iterations if the time index exceeds the maximum saved time
        if save_idx > idxs_tsave[end] + 1
            break
        end
        # Check for convergence
        if !converged
            convergence = false
            @warn("Simulation stopped due to divergence or failure to converge.")
            break
        end
    end
    return traj, tsave, convergence
end

"""
    run_MC(model_dis::ModelDisordered{NK, I, RT, D1, D2, D3, FT}, dt::RT; rng=Xoshiro(1234), showprogress=false, idxs_tsave=nothing, divergence_threshold=1e6) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D1<:Distribution, D2<:Distribution, D3<:Distribution, FT<:Function}

Run a Monte Carlo simulation of the disordered model for a given time step `dt` by sampling an instance of disorder. The function samples trajectories of the model and saves them at specified time indices.

# Arguments
- `model_dis::ModelDisordered{NK, I, RT, D1, D2, D3, FT}`: The disordered model to simulate, where `NK` is the noise kind.
- `dt::RT`: Time step for the simulation.

# Optional arguments
- `rng::AbstractRNG`: Random number generator (default Xoshiro(1234)).
- `showprogress::Bool`: Whether to show a progress bar (default false).
- `idxs_tsave::Union{Nothing, Vector{Int}}`: Indices at which to save the trajectory (default nothing, which saves at every time step).
- `divergence_threshold::RT`: Threshold for divergence detection (default 1e6).

# Output
- `traj::Matrix{RT}`: Matrix of sampled trajectories, where each column corresponds to a time point.
- `tsave::Vector{RT}`: Vector of time points at which the trajectories are saved.
- `convergence::Bool`: Boolean indicating whether the integration was successful (true) or diverged/failed (false).
"""
function run_MC(model_dis::ModelDisordered{NK, I, RT, D1, D2, D3, FT}, dt::RT; rng=Xoshiro(1234), showprogress=false, idxs_tsave=nothing, divergence_threshold=1e6) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D1<:Distribution, D2<:Distribution, D3<:Distribution, FT<:Function}
    # Get the model parameters
    N, K, M, lam, p0, noise = model_dis.N, model_dis.K, model_dis.M, model_dis.lam, model_dis.p0, model_dis.noise
    m, sigma2, corr = model_dis.m, model_dis.sigma2, model_dis.corr

    # Generate the local adjacency matrix instance
    Jmat = model_dis.generate_adj(N, K; rng=rng)
    @inbounds @fastmath for i in 1:N
        @inbounds @fastmath @simd for j in 1:i-1
            if Jmat[i, j] != 0.0
                Jmat[i, j], Jmat[j, i] = sample_couplings(m, sigma2, corr, K; rng=rng)
            end
        end
    end

    # Generate the local model instance
    model = Model(N, M, Jmat, lam, p0, noise)

    run_MC(model, dt; rng=rng, showprogress=showprogress, idxs_tsave=idxs_tsave, divergence_threshold=divergence_threshold)
end


"""
    run_MC(model_dis::ModelDisorderedFC{NK, I, RT, D}, dt::RT; rng=Xoshiro(1234), showprogress=false, idxs_tsave=nothing, divergence_threshold=1e6) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D<:Distribution}

Run a Monte Carlo simulation of the fully-connected disordered model for a given time step `dt` by sampling an instance of disorder. The function samples trajectories of the model and saves them at specified time indices.

# Arguments
- `model_dis::ModelDisorderedFC{NK, I, RT, D}`: The disordered model to simulate, where `NK` is the noise kind.
- `dt::RT`: Time step for the simulation.

# Optional arguments
- `rng::AbstractRNG`: Random number generator (default Xoshiro(1234)).
- `showprogress::Bool`: Whether to show a progress bar (default false).
- `idxs_tsave::Union{Nothing, Vector{Int}}`: Indices at which to save the trajectory (default nothing, which saves at every time step).
- `divergence_threshold::RT`: Threshold for divergence detection (default 1e6).

# Output
- `traj::Matrix{RT}`: Matrix of sampled trajectories, where each column corresponds to a time point.
- `tsave::Vector{RT}`: Vector of time points at which the trajectories are saved.
- `convergence::Bool`: Boolean indicating whether the integration was successful (true) or diverged/failed (false).
"""
function run_MC(model_dis::ModelDisorderedFC{NK, I, RT, D}, dt::RT; rng=Xoshiro(1234), showprogress=false, idxs_tsave=nothing, divergence_threshold=1e6) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D<:Distribution}
    # Get the model parameters
    N, M, lam, p0, noise = model_dis.N, model_dis.M, model_dis.lam, model_dis.p0, model_dis.noise
    m, sigma2, corr = model_dis.m, model_dis.sigma2, model_dis.corr

    # Generate the local adjacency matrix instance
    Jmat = zeros(RT, N, N)
    @inbounds @fastmath for i in 1:N
        @inbounds @fastmath @simd for j in 1:i-1
            Jmat[i, j], Jmat[j, i] = sample_couplings(m, sigma2, corr, N; rng=rng)
        end
    end
    
    # Generate the local model instance
    model = Model(N, M, Jmat, lam, p0, noise)

    run_MC(model, dt; rng=rng, showprogress=showprogress, idxs_tsave=idxs_tsave, divergence_threshold=divergence_threshold)
end