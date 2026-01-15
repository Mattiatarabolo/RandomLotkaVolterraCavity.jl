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

J, J_prime = sample_couplings(m, sigma2, corr, 5.5)
``` 
"""
function sample_couplings(m::RT, sigma2::RT, corr::RT, K::RT; rng::AbstractRNG=Xoshiro(1234)) where {RT<:Real}
    u = randn(rng, RT)
    v = randn(rng, RT)
    scale = sqrt(sigma2 / K)
    mean = m / K
    J = mean + scale * u
    J_prime = mean + scale * (corr * u + sqrt(one(RT) - corr ^ 2) * v)
    return J, J_prime
end

"""
    sample_couplings(m::RT, sigma2::RT, corr::RT, K::IT; rng::AbstractRNG=Xoshiro(1234)) where {RT<:Real, IT<:Integer}

Sample couplings J and Jp from a bivariate Gaussian distribution with means `m/K`, variances `sigma2/K`, and correlation `corr`.

# Arguments
- `m::RT`: Mean value of the couplings.
- `sigma2::RT`: Variance of the couplings.
- `corr::RT`: Correlation between the couplings.
- `K::IT`: Average degree of the network (used for proper scaling of the couplings).

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
function sample_couplings(m::RT, sigma2::RT, corr::RT, K::IT; rng::AbstractRNG=Xoshiro(1234)) where {RT<:Real, IT<:Integer}
    u = randn(rng, RT)
    v = randn(rng, RT)
    scale = sqrt(sigma2 / K)
    mean = m / K
    J = mean + scale * u
    J_prime = mean + scale * (corr * u + sqrt(one(RT) - corr ^ 2) * v)
    return J, J_prime
end


#################################################################################################################################
#################################################################################################################################

"""
    run_MC(model::Model{NK, I, RT, MT, D}, dt::RT; rng=Xoshiro(1234), tsave=nothing, divergence_threshold=1e6, stopateq=false, eq_threshold=1e-10, min_t_eq=nothing, verbose=false, integrator="adaptive", x0=nothing, tinit=0.0) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, MT<:AbstractMatrix{RT}, D<:Distribution}

Run a Monte Carlo simulation of the model for a given time step `dt`. It uses the `OrdinaryDiffEq.jl` package to integrate the ODEs. The function samples trajectories of the model and saves them at specified time indices.

# Arguments
- `model::Model{NK, I, RT, MT, D}`: The model to simulate, where `NK` is the noise kind, `I` is the integer type for indices, `RT` is the real type, `MT` is the matrix type, and `D` is the distribution type.
- `dt::RT`: Time step for the simulation.

# Optional arguments
- `rng::AbstractRNG`: Random number generator (default Xoshiro(1234)).
- `tsave::Union{Nothing, Vector{RT}}`: Vector of time points at which the trajectories are saved.
- `divergence_threshold::RT`: Threshold for divergence detection (default 1e6).
- `stopateq::Bool`: Whether to stop the simulation upon reaching a steady state (default false).
- `eq_threshold::RT`: Threshold for steady state detection (default 1e-10).
- `min_t_eq::Union{Nothing, RT}`: Minimum time before checking for steady state (default nothing, which uses the default in `TerminateSteadyState`).
- `verbose::Bool`: Whether to print information about the simulation (default false).
- `integrator::String`: Choice of integrator, either "adaptive" (automatically switching integrator with adaptive time-step) or "fixed" (Euler integrator with fixed time-step) (default "adaptive").
- `x0::Union{Nothing, Vector{RT}}`: Initial condition for the ODE (default nothing, which samples a random initial condition).
- `tinit::RT`: Initial time for the ODE integration (default 0.0).

# Output
- `traj::Matrix{RT}`: Matrix of sampled trajectories, where each column corresponds to a time point.
- `tsave::Vector{RT}`: Vector of time points at which the trajectories are saved.
- `convergence::Bool`: Boolean indicating whether the integration was successful (true) or diverged/failed (false).
- `t_eq::RT`: Equilibriation time. If stopateq is false, returns the final time of integration.
"""
function run_MC(model::Model{NK, I, RT, MT, D}, dt::RT; rng=Xoshiro(1234), tsave=nothing, divergence_threshold=1e6, stopateq=false, eq_threshold=1e-10, min_t_eq=nothing, verbose=false, integrator="adaptive", x0=nothing, tinit=0.0) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, MT<:AbstractMatrix{RT}, D<:Distribution}
    # Get the model parameters
    N, _, lam = model.N, model.M, model.lam

    # Initialize the array to store the trajectory
    traj = fill(NaN, N, length(tsave))

    # Sample initial condition if not provided
    if x0 === nothing
        x0 = max.(rand(rng, model.p0, N), lam)
    elseif length(x0) != N
        throw(ArgumentError("Initial condition x0 must have length N = $N"))
    end

    # Define the ODE problem
    pars = (model.J, lam) # Parameters for the ODE function
    Jac = sparse(model.J .+ Diagonal(ones(RT, N))) # Jacobian prototype
    fun = ODEFunction(_sample!; jac_prototype=Jac) # ODE function with Jacobian
    prob = ODEProblem(fun, x0, (tinit, tsave[end]), pars) # Define the ODE problem

    # Define the callback
    if stopateq
        term_ss = TerminateSteadyState(eq_threshold, eq_threshold; min_t=min_t_eq)
        cb = CallbackSet(hard_wall, term_ss)
    else
        cb = hard_wall
    end
    
    # Solve the ODE problem using a stiff solver if necessary
    if integrator == "adaptive"
        sol = solve(prob, AutoTsit5(Rosenbrock23()), dt=dt, saveat=tsave, unstable_check=(dt, u, p, t) -> any(!isfinite, u) || any(u .> divergence_threshold), callback=cb, reltol=1e-10, abstol=1e-10)
    else
        sol = solve(prob, Euler(), dt=dt, saveat=tsave, unstable_check=(dt, u, p, t) -> any(!isfinite, u) || any(u .> divergence_threshold), callback=cb, reltol=1e-10, abstol=1e-10)
    end
    convergence = !(sol.retcode == ReturnCode.Unstable || sol.retcode == ReturnCode.Failure)

    # Complete the output if stopped at steady state
    if sol.retcode == ReturnCode.Terminated && stopateq
        if verbose
            @info "Stopped at steady state at t = $(sol.t[end])"
        end
        t_conv = sol.t[end]
        t_conv_idx = findfirst(x-> x >= t_conv, tsave)
        traj[:, 1:t_conv_idx] .= sol[:, 1:t_conv_idx]
        traj[:, t_conv_idx+1:end] .= sol[:, end] # Fill the rest with the last value
    else
        traj[:, :] .= sol[:, :]
    end

    return traj, tsave, convergence, sol.t[end]
end

"""
    run_MC(model_dis::ModelDisordered{NK, I, RT, D1, D2, D3, FT}, dt::RT; rng=Xoshiro(1234), tsave=nothing, divergence_threshold=1e6, stopateq=false, eq_threshold=1e-10, min_t_eq=1e2, verbose=false, integrator="adaptive", x0=nothing, tinit=0.0) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D1<:Distribution, D2<:Distribution, D3<:Distribution, FT<:Function}

Run a Monte Carlo simulation of the disordered model for a given time step `dt` by sampling an instance of disorder. It uses the `OrdinaryDiffEq.jl` package to integrate the ODEs. The function samples trajectories of the model and saves them at specified time indices.

# Arguments
- `model_dis::ModelDisordered{NK, I, RT, D1, D2, D3, FT}`: The disordered model to simulate, where `NK` is the noise kind.
- `dt::RT`: Time step for the simulation.

# Optional arguments
- `rng::AbstractRNG`: Random number generator (default Xoshiro(1234)).
- `tsave::Union{Nothing, Vector{Int}}`: Vector of time points at which the trajectories are saved.
- `divergence_threshold::RT`: Threshold for divergence detection (default 1e6).
- `stopateq::Bool`: Whether to stop the simulation upon reaching a steady state (default false).
- `eq_threshold::RT`: Threshold for steady state detection (default 1e-10).
- `min_t_eq::Union{Nothing, RT}`: Minimum time before checking for steady state (default nothing, which uses the default in `TerminateSteadyState`).
- `verbose::Bool`: Whether to print information about the simulation (default false).
- `integrator::String`: Choice of integrator, either "adaptive" (automatically switching integrator with adaptive time-step) or "fixed" (Euler integrator with fixed time-step) (default "adaptive").
- `x0::Union{Nothing, Vector{RT}}`: Initial condition for the ODE (default nothing, which samples a random initial condition).
- `tinit::RT`: Initial time for the ODE integration (default 0.0).

# Output
- `traj::Matrix{RT}`: Matrix of sampled trajectories, where each column corresponds to a time point.
- `tsave::Vector{RT}`: Vector of time points at which the trajectories are saved.
- `convergence::Bool`: Boolean indicating whether the integration was successful (true) or diverged/failed (false).
- `t_eq::RT`: Equilibriation time. If stopateq is false, returns the final time of integration.
"""
function run_MC(model_dis::ModelDisordered{NK, I, RT, D1, D2, D3, FT}, dt::RT; rng=Xoshiro(1234), tsave=nothing, divergence_threshold=1e6, stopateq=false, eq_threshold=1e-10, min_t_eq=1e2, verbose=false, integrator="adaptive", x0=nothing, tinit=0.0) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D1<:Distribution, D2<:Distribution, D3<:Distribution, FT<:Function}
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

    run_MC(model, dt; rng=rng, tsave=tsave, divergence_threshold=divergence_threshold, stopateq=stopateq, eq_threshold=eq_threshold, min_t_eq=min_t_eq, verbose=verbose, integrator=integrator, x0=x0, tinit=tinit)
end


"""
    run_MC(model_dis::ModelDisorderedFC{NK, I, RT, D}, dt::RT; rng=Xoshiro(1234), tsave=nothing, divergence_threshold=1e6, stopateq=false, eq_threshold=1e-10, min_t_eq=1e2, verbose=false, integrator="adaptive", x0=nothing, tinit=0.0) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D<:Distribution}

Run a Monte Carlo simulation of the  fully-connected disordered model for a given time step `dt`, by sampling an instance of disorder. It uses the `OrdinaryDiffEq.jl` package to integrate the ODEs. The function samples trajectories of the model and saves them at specified time indices.

# Arguments
- `model_dis::ModelDisorderedFC{NK, I, RT, D}`: The disordered model to simulate, where `NK` is the noise kind.
- `dt::RT`: Time step for the simulation.

# Optional arguments
- `rng::AbstractRNG`: Random number generator (default Xoshiro(1234)).
- `tsave::Union{Nothing, Vector{Int}}`: Vector of time points at which the trajectories are saved.
- `divergence_threshold::RT`: Threshold for divergence detection (default 1e6).
- `stopateq::Bool`: Whether to stop the simulation upon reaching a steady state (default false).
- `eq_threshold::RT`: Threshold for steady state detection (default 1e-10).
- `min_t_eq::Union{Nothing, RT}`: Minimum time before checking for steady state (default nothing, which uses the default in `TerminateSteadyState`).
- `verbose::Bool`: Whether to print information about the simulation (default false).
- `integrator::String`: Choice of integrator, either "adaptive" (automatically switching integrator with adaptive time-step) or "fixed" (Euler integrator with fixed time-step) (default "adaptive").
- `x0::Union{Nothing, Vector{RT}}`: Initial condition for the ODE (default nothing, which samples a random initial condition).
- `tinit::RT`: Initial time for the ODE integration (default 0.0).

# Output
- `traj::Matrix{RT}`: Matrix of sampled trajectories, where each column corresponds to a time point.
- `tsave::Vector{RT}`: Vector of time points at which the trajectories are saved.
- `convergence::Bool`: Boolean indicating whether the integration was successful (true) or diverged/failed (false).
- `t_eq::RT`: Equilibriation time. If stopateq is false, returns the final time of integration.
"""
function run_MC(model_dis::ModelDisorderedFC{NK, I, RT, D}, dt::RT; rng=Xoshiro(1234), tsave=nothing, divergence_threshold=1e6, stopateq=false, eq_threshold=1e-10, min_t_eq=1e2, verbose=false, integrator="adaptive", x0=nothing, tinit=0.0) where {NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D<:Distribution}
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

    run_MC(model, dt; rng=rng, tsave=tsave, divergence_threshold=divergence_threshold, stopateq=stopateq, eq_threshold=eq_threshold, min_t_eq=min_t_eq, verbose=verbose, integrator=integrator, x0=x0, tinit=tinit)
end