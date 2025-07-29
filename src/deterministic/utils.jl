######################################################################################################
########################################## Monte Carlo ###############################################
######################################################################################################

function _sample!(x::Vector{RT}, h::Vector{RT}, model::Model{Deterministic, I, RT, MT, D}, dt::RT, divergence_threshold::RT, rng::AbstractRNG, p::Progress) where {I<:Integer, RT<:Real, MT<:AbstractMatrix{RT}, D<:Distribution}
    # Get the model parameters
    N, lam = model.N, model.lam # number of nodes in the graph and the coupling coefficient

    # Compute the fields h
    mul!(h, model.J, x)
    next!(p; step = N^2)

    # Update the process path
    @inbounds for i in 1:N
        # Compute the term into square brackets
        F = 1 - x[i] + h[i]
        # Compute the new process path
        x[i] *= exp(dt * F)
        x[i] = lam + abs(x[i] - lam)   # reflective boundary condition at lam
        # Check for divergence
        if x[i] > divergence_threshold
            @warn("Divergence detected in the trajectory.")
            x[i] = divergence_threshold # Set to threshold to avoid divergence
        end
        next!(p)
    end
end

#######################################################################################################
############################################# FP GECaM ################################################
#######################################################################################################

function init_nodes(model::Model{Deterministic, I, RT, MT, D}, init_type::Symbol, mu0::RT, q0::RT, chi0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real, MT<:AbstractMatrix{RT}, D<:Distribution}
    nodes = Vector{NodeFP{Deterministic, I, RT}}(undef, model.N)
    for i in 1:model.N
        neighs = findall(model.J[i, :] .!= 0) # Find neighbors of node i
        nodes[i] = NodeFP(i, neighs, init_type, mu0, q0, chi0, rng) # Initialize the node with its neighbors
    end
    return nodes
end

# Shortname for functions used in cavity update
function gauss(x::RT) where {RT<:Real}
    return exp( - x^2 / 2) / sqrt( 2 * pi )
end

function mod_erf(x::RT) where {RT<:Real}
    return ( 1 + erf(x / sqrt(2)) ) / 2
end


### Function to solve the fully-connected system (DMFT solution)
# Define the functions
w0_func(x::RT) where {RT<:Real} = mod_erf(x)
w1_func(x::RT) where {RT<:Real} = x * mod_erf(x) + gauss(x)
w2_func(x::RT) where {RT<:Real} = w0_func(x) + x * w1_func(x)

#############################################################################################
########################################## FP DMFT ##########################################
#############################################################################################

"""
    analytic_FC(m::RT, gamma::RT, npoints::I, Delta_min::RT, Delta_max::RT) where {RT<:Real, I<:Integer}

Returns the analytic solution of the fully-connected system for given parameters.

# Arguments
- `m::RT`: Average coupling strength.
- `gamma::RT`: Coupling strength.
- `npoints::I`: Number of points in the range.
- `Delta_min::RT`: Minimum value of the parameter Delta.
- `Delta_max::RT`: Maximum value of the parameter Delta.

# Returns
- `Deltas::Vector{RT}`: Values of the parameter Delta.
- `mus::Vector{RT}`: Fixed-point values of the mean abundances mu.
- `qs::Vector{RT}`: Fixed-point values of the mean squared abundances q.
- `chis::Vector{RT}`: Fixed-point values of the susceptibilities chi.
- `sigma2s::Vector{RT}`: Variances sigma2.
- `phis::Vector{RT}`: Fixed-point values of the survival probabilities phi.
""" 
function analytic_FC(m::RT, gamma::RT, npoints::I, Delta_min::RT, Delta_max::RT) where {RT<:Real, I<:Integer}
    Deltas = range(Delta_min, Delta_max, length=npoints)
    mus = zeros(RT, npoints)
    qs = zeros(RT, npoints)
    chis = zeros(RT, npoints)
    sigma2s = zeros(RT, npoints)
    phis = zeros(RT, npoints)
    
    for (iD,Delta) in enumerate(Deltas)
        w0 = w0_func(Delta)
        w1 = w1_func(Delta)
        w2 = w2_func(Delta)
    
        chis[iD] = w0 + gamma * w0^2 / w2
        sigma2s[iD] = w2 / (w2 + gamma * w0)^2
        mus[iD] = 1/(Delta / w1 * w2 / (w2 + gamma * w0) - m)
        qs[iD] = (w2 / (w2 + gamma * w0) * mus[iD] / (sqrt(sigma2s[iD]) * w1))^2
        phis[iD] = w0
    end

    return Deltas, mus, qs, chis, sigma2s, phis
end

