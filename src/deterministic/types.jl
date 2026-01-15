"""
    Deterministic <: AbstractNoiseKind

A type that represents deterministic dynamics, i.e., no noise.
"""
struct Deterministic <: AbstractNoiseKind end

nnoises(Deterministic::Type{Deterministic}) = 1

##############################################################################################
################################         FIXED POINT          ################################
##############################################################################################

############################## Single instance of disordered model ########################################
"""
    CavityFP{Deterministic, I, RT}

A type that represents a cavity in the fixed-point algorithm for deterministic dynamics.

# Fields
$(TYPEDFIELDS)
"""
mutable struct CavityFP{Deterministic, I<:Integer, RT<:Real}
    """ Node index """
    i::I
    """ Neighbor index """
    j::I
    """ Cavity mean """
    mu::RT
    """ Cavity correlation """
    q::RT
    """ Cavity susceptibility """
    chi::RT
    """ Cavity survival probability """
    psi::RT
    """
        CavityFP{Deterministic, I, RT}(i::I, j::I)

    Create a new dixed-point cavity with the specified node index and neighbor index.

    # Arguments
    - `i::I`: Node index.
    - `j::I`: Neighbor index.

    # Output
    - `CavityFP`: Cavity with the specified node and neighbor indices.
    """
    function CavityFP(i::I, j::I, init_type::Symbol, mu0::RT, q0::RT, chi0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}
        if init_type == :zero
            return new{Deterministic, I, RT}(i, j, zero(RT), zero(RT), zero(RT), zero(RT))
        elseif init_type == :random
            return new{Deterministic, I, RT}(i, j, rand(rng), rand(rng), zero(RT), zero(RT))
        elseif init_type == :custom
            return new{Deterministic, I, RT}(i, j, mu0, q0, chi0, zero(RT))
        end
    end
end

"""
    CavityFP_q0{Deterministic, I, RT}

A type that represents a cavity in the fixed-point algorithm for deterministic dynamics, using the simplified version where q = χ = 0, i.e. a first order approximation of the cavity messages.

# Fields
$(TYPEDFIELDS)
"""
mutable struct CavityFP_q0{Deterministic, I<:Integer, RT<:Real}
    """ Node index """
    i::I
    """ Neighbor index """
    j::I
    """ Cavity abundance """
    x::RT
    """
        CavityFP_q0{Deterministic, I, RT}(i::I, j::I, init_type::Symbol, x0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}

    Create a new dixed-point cavity with the specified node index and neighbor index.

    # Arguments
    - `i::I`: Node index.
    - `j::I`: Neighbor index.
    - `init_type::Symbol`: Initialization type, can be `:zero`, `:random`, or `:custom`.
    - `x0::RT`: Initial abundance value (used if `init_type` is `:custom`).
    - `rng::AbstractRNG`: Random number generator.

    # Output
    - `CavityFP_q0`: Cavity with the specified node and neighbor indices.
    """
    function CavityFP_q0(i::I, j::I, init_type::Symbol, x0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}
        if init_type == :zero
            return new{Deterministic, I, RT}(i, j, zero(RT))
        elseif init_type == :random
            return new{Deterministic, I, RT}(i, j, rand(rng))
        elseif init_type == :custom
            return new{Deterministic, I, RT}(i, j, x0)
        end
    end
end

"""
    MarginalFP{Deterministic, I, RT}

A type that represents a marginal in the fixed-point algorithm for deterministic dynamics.

# Fields
$(TYPEDFIELDS)
"""
mutable struct MarginalFP{Deterministic, I<:Integer, RT<:Real}
    """ Node index """
    i::I
    """ Marginal mean """
    mu::RT
    """ Marginal correlation """
    q::RT
    """ Marginal susceptibility """
    chi::RT
    """ Marginal survival probability """
    psi::RT
    """
        MarginalFP{Deterministic, I, RT}(i::I)

    Create a new fixed-point marginal with the specified node index.

    # Arguments
    - `i::I`: Node index.

    # Output
    - `MarginalFP`: Marginal with the specified node index.
    """
    function MarginalFP(i::I, init_type::Symbol, mu0::RT, q0::RT, chi0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}
        if init_type == :zero
            return new{Deterministic, I, RT}(i, zero(RT), zero(RT), zero(RT), zero(RT))
        elseif init_type == :random
            return new{Deterministic, I, RT}(i, rand(rng), rand(rng), zero(RT), zero(RT))
        elseif init_type == :custom
            return new{Deterministic, I, RT}(i, mu0, q0, chi0, zero(RT))
        end
    end
end

"""
    MarginalFP_q0{Deterministic, I, RT}

A type that represents a marginal in the fixed-point algorithm for deterministic dynamics, using the simplified version where q = χ = 0, i.e. a first order approximation of the cavity messages.

# Fields
$(TYPEDFIELDS)
"""
mutable struct MarginalFP_q0{Deterministic, I<:Integer, RT<:Real}
    """ Node index """
    i::I
    """ Marginal abundance """
    x::RT
    """
        MarginalFP_q0{Deterministic, I, RT}(i::I, init_type::Symbol, x0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}

    Create a new fixed-point marginal with the specified node index.

    # Arguments
    - `i::I`: Node index.
    - `init_type::Symbol`: Initialization type, can be `:zero`, `:random`, or `:custom`.
    - `x0::RT`: Initial abundance value (used if `init_type` is `:custom`).
    - `rng::AbstractRNG`: Random number generator.

    # Output
    - `MarginalFP_q0`: Marginal with the specified node index.
    """
    function MarginalFP_q0(i::I, init_type::Symbol, x0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}
        if init_type == :zero
            return new{Deterministic, I, RT}(i, zero(RT))
        elseif init_type == :random
            return new{Deterministic, I, RT}(i, rand(rng))
        elseif init_type == :custom
            return new{Deterministic, I, RT}(i, x0)
        end
    end
end


"""
    NodeFP{Deterministic, I, RT}

A type that represents a node in the fixed-point algorithm for deterministic dynamics.

# Fields
$(TYPEDFIELDS)
"""
struct NodeFP{Deterministic, I<:Integer, RT<:Real}
    """ Node index """
    i::I
    """ Neighbors' indices """
    neighs::Vector{I}
    """ Dictionary to map neighbors' indices to their positions in the `neighs` vector """
    neighs_idx::Dict{I, I} # Value is the 1-based index in the neighs vector
    """ Cavity messages """
    cavs::Vector{CavityFP{Deterministic, I, RT}}
    """ Marginal """
    marg::MarginalFP{Deterministic, I, RT}
    """
        NodeFP{Deterministic, I, RT}(i::I, neighs::AbstractVector{I}) where {I<:Integer, RT<:Real}

    Create a new fixed-point node with the specified index and neighbors.

    # Arguments
    - `i::I`: Node index.
    - `neighs::Vector{I}`: Neighbors' indices.

    # Output
    - `NodeFP`: Node with the specified index and neighbors.
    """
    function NodeFP(i::I, neighs::Vector{I}, init_type::Symbol, mu0::RT, q0::RT, chi0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}
        neighs_idx = Dict(neighs .=> eachindex(neighs))  # Map neighbor indices to their positions in the `neighs` vector
        cavs = [CavityFP(i, j, init_type, mu0, q0, chi0, rng) for j in neighs] # Initialize cavities
        marg = MarginalFP(i, init_type, mu0, q0, chi0, rng) # Initialize marginal
        new{Deterministic, I, RT}(i, neighs, neighs_idx, cavs, marg)
    end
end

"""
    CavityFP_q0{Deterministic, I, RT}

A type that represents a cavity in the fixed-point algorithm for deterministic dynamics, using the simplified version where q = χ = 0, i.e. a first order approximation of the cavity messages.

# Fields
$(TYPEDFIELDS)
"""
struct NodeFP_q0{Deterministic, I<:Integer, RT<:Real}
    """ Node index """
    i::I
    """ Neighbors' indices """
    neighs::Vector{I}
    """ Dictionary to map neighbors' indices to their positions in the `neighs` vector """
    neighs_idx::Dict{I, I} # Value is the 1-based index in the neighs vector
    """ Cavity messages """
    cavs::Vector{CavityFP_q0{Deterministic, I, RT}}
    """ Marginal """
    marg::MarginalFP_q0{Deterministic, I, RT}
    """
        NodeFP_q0{Deterministic, I, RT}(i::I, neighs::AbstractVector{I}, init_type::Symbol, x0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}

    Create a new fixed-point node with the specified index and neighbors.

    # Arguments
    - `i::I`: Node index.
    - `neighs::Vector{I}`: Neighbors' indices.
    - `init_type::Symbol`: Initialization type, can be `:zero`, `:random`, or `:custom`.
    - `x0::RT`: Initial abundance value (used if `init_type` is `:custom`).
    - `rng::AbstractRNG`: Random number generator.

    # Output
    - `NodeFP_q0`: Node with the specified index and neighbors.
    """
    function NodeFP_q0(i::I, neighs::Vector{I}, init_type::Symbol, x0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}
        neighs_idx = Dict(neighs .=> eachindex(neighs))  # Map neighbor indices to their positions in the `neighs` vector
        cavs = [CavityFP_q0(i, j, init_type, x0, rng) for j in neighs] # Initialize cavities
        marg = MarginalFP_q0(i, init_type, x0, rng) # Initialize marginal
        new{Deterministic, I, RT}(i, neighs, neighs_idx, cavs, marg)
    end
end

################################# Average over disordered model ###########################################

"""
    PopFP{Deterministic, I, RT}

A type that represents a population of messages for the fixed-point algorithm in deterministic dynamics.

# Fields
$(TYPEDFIELDS)
"""
struct PopFP{Deterministic, I<:Integer, RT<:Real}
    """Population of averages"""
    mu_pop::Vector{RT}
    """Population of cavity correlations"""
    q_pop::Vector{RT}
    """Population of cavity susceptibilities"""
    chi_pop::Vector{RT}
    """Population of cavity survival probabilities"""
    psi_pop::Vector{RT}
    """
        PopFP(P::I, init_type::Symbol, mu0::RT, q0::RT, chi0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}

    Create a population of messages for the fixed-point algorithm in deterministic dynamics.

    # Arguments
    - `P::I`: Number of population elements.
    - `init_type::Symbol`: Initialization type, can be `:zero`, `:random`, or `:custom`.
    - `mu0::RT`: Initial mean value (used if `init_type` is `:custom`).
    - `q0::RT`: Initial correlation value (used if `init_type` is `:custom`).
    - `chi0::RT`: Initial susceptibility value (used if `init_type` is `:custom`).

    # Keyword Arguments
    - `rng::AbstractRNG`: Random number generator (default is `Xoshiro(1234)`). 

    # Returns
    - `PopFP{Deterministic, I, RT}`: Population of fixed-point cavity averages.
    """
    function PopFP(P::I, init_type::Symbol, mu0::RT, q0::RT, chi0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}
        if init_type == :zero
            return new{Deterministic, I, RT}(zeros(RT, P), zeros(RT, P), zeros(RT, P), zeros(RT, P))
        elseif init_type == :random
            return new{Deterministic, I, RT}(rand(rng, RT, P), rand(rng, RT, P), zeros(RT, P), zeros(RT, P))
        elseif init_type == :custom
            return new{Deterministic, I, RT}(fill(mu0, P), fill(q0, P), fill(chi0, P), zeros(RT, P))
        end
    end
end

"""
    PopFP_q0{Deterministic, I, RT}

A type that represents a population of messages for the fixed-point algorithm in deterministic dynamics. It uses the simplified version where q = χ = 0, i.e. a first order approximation of the cavity messages.

# Fields
$(TYPEDFIELDS)
"""
struct PopFP_q0{Deterministic, I<:Integer, RT<:Real}
    """Population of abundances"""
    x_pop::Vector{RT}
    """
        PopFP_q0(P::I, init_type::Symbol, x0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}

    Create a population of messages for the fixed-point algorithm in deterministic dynamics.

    # Arguments
    - `P::I`: Number of population elements.
    - `init_type::Symbol`: Initialization type, can be `:zero`, `:random`, or `:custom`.
    - `x0::RT`: Initial abundance value (used if `init_type` is `:custom`).

    # Keyword Arguments
    - `rng::AbstractRNG`: Random number generator (default is `Xoshiro(1234)`). 

    # Returns
    - `PopFP_q0{Deterministic, I, RT}`: Population of fixed-point cavity averages.
    """
    function PopFP_q0(P::I, init_type::Symbol, x0::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}
        if init_type == :zero
            return new{Deterministic, I, RT}(zeros(RT, P))
        elseif init_type == :random
            return new{Deterministic, I, RT}(rand(rng, RT, P))
        elseif init_type == :custom
            return new{Deterministic, I, RT}(fill(x0, P))
        end
    end
end

"""
    PopJ{Deterministic, I, RT}

A type that represents a population of interaction strengths J and J' for a given number of nodes.

# Fields
$(TYPEDFIELDS)
"""
struct PopJ{Deterministic, I<:Integer, RT<:Real}
    """Population of interaction strengths J"""
    J_pop::Vector{RT}
    """Population of interaction strengths J'"""
    Jp_pop::Vector{RT}
    """
        PopJ{Deterministic, I, RT}(P::I, m::RT, sigma2::RT, gamma::RT, K::RT; rng::AbstractRNG=Xoshiro(1234)) where {I<:Integer, RT<:Real}

    Create a population of interaction strengths J and J' for a given number of nodes.

    # Arguments
    - `P::I`: Number of nodes in the population.
    - `m::RT`: Mean value of the couplings.
    - `sigma2::RT`: Variance of the couplings.
    - `gamma::RT`: Correlation between the couplings.
    - `K::RT`: Average degree of the network (used for proper scaling of the couplings).

    # Keyword Arguments
    - `rng::AbstractRNG`: Random number generator (default Xoshiro(1234)).
    
    # Returns
    - `PopJ{Deterministic, I, RT}`: Population of interaction strengths J and J'.
    """
    function PopJ(P::I, m::RT, sigma2::RT, gamma::RT, K::RT, rng::AbstractRNG) where {I<:Integer, RT<:Real}
        J_pop = zeros(RT, P)
        Jp_pop = zeros(RT, P)
        for i in 1:P
            J_pop[i], Jp_pop[i] = sample_couplings(m, sigma2, gamma, K; rng=rng)
        end
        return new{Deterministic, I, RT}(J_pop, Jp_pop)
    end
end