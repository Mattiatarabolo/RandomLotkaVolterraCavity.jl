#################################################################################################################################
#################################################################################################################################

"""
    run_MC_ODE(model::Model{NK, I, RT, MT, D}, dt::RT; rng=Xoshiro(1234), idxs_tsave=nothing, divergence_threshold=1e6) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, MT<:AbstractMatrix{RT}, D<:Distribution}

Run a Monte Carlo simulation of the model for a given time step `dt`. It uses the `OrdinaryDiffEq.jl` package to integrate the ODEs. The function samples trajectories of the model and saves them at specified time indices.

# Arguments
- `model::Model{NK, I, RT, MT, D}`: The model to simulate, where `NK` is the noise kind, `I` is the integer type for indices, `RT` is the real type, `MT` is the matrix type, and `D` is the distribution type.
- `dt::RT`: Time step for the simulation.

# Optional arguments
- `rng::AbstractRNG`: Random number generator (default Xoshiro(1234)).
- `idxs_tsave::Union{Nothing, Vector{Int}}`: Indices at which to save the trajectory (default nothing, which saves at every time step).
- `divergence_threshold::RT`: Threshold for divergence detection (default 1e6).

# Output
- `traj::Matrix{RT}`: Matrix of sampled trajectories, where each column corresponds to a time point.
- `tsave::Vector{RT}`: Vector of time points at which the trajectories are saved.
- `convergence::Bool`: Boolean indicating whether the integration was successful (true) or diverged/failed (false).
"""
function run_MC_ODE(model::Model{NK, I, RT, MT, D}, dt::RT; rng=Xoshiro(1234), idxs_tsave=nothing, divergence_threshold=1e6) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, MT<:AbstractMatrix{RT}, D<:Distribution}
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
    traj = zeros(RT, N, Msave)

    # Sample initial condition
    x0 = max.(rand(rng, model.p0, N), lam)
    
    # Define the ODE problem
    pars = (model.J, lam)
    Jac = sparse(model.J .+ Diagonal(ones(RT, N))) # Jacobian prototype
    f(du, u, p, t) = ODEFunction(_sample_ODE!(du, u, p, t), jac = _jac_ODE, jac_prototype = Jac)
    prob = ODEProblem(_sample_ODE!, x0, (0.0, tsave[end]), pars, )
    
    # Solve the ODE problem using a stiff solver if necessary
    sol = solve(prob, AutoTsit5(Rosenbrock23()), dt=dt, saveat=tsave, unstable_check=(dt, u, p, t) -> any(!isfinite, u) || any(u .> divergence_threshold))
    convergence = !(sol.retcode == ReturnCode.Unstable || sol.retcode == ReturnCode.Failure)

    # Extract the trajectory into the storage array
    traj .= sol[:, :]
    
    return traj, tsave, convergence
end

"""
    run_MC_ODE(model_dis::ModelDisordered{NK, I, RT, D1, D2, D3, FT}, dt::RT; rng=Xoshiro(1234), idxs_tsave=nothing, divergence_threshold=1e6) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D1<:Distribution, D2<:Distribution, D3<:Distribution, FT<:Function}

Run a Monte Carlo simulation of the disordered model for a given time step `dt` by sampling an instance of disorder. It uses the `OrdinaryDiffEq.jl` package to integrate the ODEs. The function samples trajectories of the model and saves them at specified time indices.

# Arguments
- `model_dis::ModelDisordered{NK, I, RT, D1, D2, D3, FT}`: The disordered model to simulate, where `NK` is the noise kind.
- `dt::RT`: Time step for the simulation.

# Optional arguments
- `rng::AbstractRNG`: Random number generator (default Xoshiro(1234)).
- `idxs_tsave::Union{Nothing, Vector{Int}}`: Indices at which to save the trajectory (default nothing, which saves at every time step).
- `divergence_threshold::RT`: Threshold for divergence detection (default 1e6).

# Output
- `traj::Matrix{RT}`: Matrix of sampled trajectories, where each column corresponds to a time point.
- `tsave::Vector{RT}`: Vector of time points at which the trajectories are saved.
- `convergence::Bool`: Boolean indicating whether the integration was successful (true) or diverged/failed (false).
"""
function run_MC_ODE(model_dis::ModelDisordered{NK, I, RT, D1, D2, D3, FT}, dt::RT; rng=Xoshiro(1234), idxs_tsave=nothing, divergence_threshold=1e6) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D1<:Distribution, D2<:Distribution, D3<:Distribution, FT<:Function}
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

    run_MC_ODE(model, dt; rng=rng, idxs_tsave=idxs_tsave, divergence_threshold=divergence_threshold)
end


"""
    run_MC_ODE(model_dis::ModelDisorderedFC{NK, I, RT, D}, dt::RT; rng=Xoshiro(1234), idxs_tsave=nothing, divergence_threshold=1e6) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D<:Distribution}

Run a Monte Carlo simulation of the  fully-connected disordered model for a given time step `dt`, by sampling an instance of disorder. It uses the `OrdinaryDiffEq.jl` package to integrate the ODEs. The function samples trajectories of the model and saves them at specified time indices.

# Arguments
- `model_dis::ModelDisorderedFC{NK, I, RT, D}`: The disordered model to simulate, where `NK` is the noise kind.
- `dt::RT`: Time step for the simulation.

# Optional arguments
- `rng::AbstractRNG`: Random number generator (default Xoshiro(1234)).
- `idxs_tsave::Union{Nothing, Vector{Int}}`: Indices at which to save the trajectory (default nothing, which saves at every time step).
- `divergence_threshold::RT`: Threshold for divergence detection (default 1e6).

# Output
- `traj::Matrix{RT}`: Matrix of sampled trajectories, where each column corresponds to a time point.
- `tsave::Vector{RT}`: Vector of time points at which the trajectories are saved.
- `convergence::Bool`: Boolean indicating whether the integration was successful (true) or diverged/failed (false).
"""
function run_MC_ODE(model_dis::ModelDisorderedFC{NK, I, RT, D}, dt::RT; rng=Xoshiro(1234), idxs_tsave=nothing, divergence_threshold=1e6) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D<:Distribution}
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

    run_MC_ODE(model, dt; rng=rng, idxs_tsave=idxs_tsave, divergence_threshold=divergence_threshold)
end