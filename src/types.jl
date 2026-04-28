##############################################################################################
###################################         MODELS         ###################################
##############################################################################################

"""
    AbstractNoiseKind

An abstract type that represents the kind of noise in the model. It can be one of the following:
- `Deterministic`: Represents deterministic dynamics, i.e., no noise.
- `Demographic`: Represents demographic noise, i.e., emerging from birth and death processes.
- `Environmental`: Represents environmental noise, i.e., arising from external factors affecting the population dynamics.
"""
abstract type AbstractNoiseKind end



"""
    ModelDisordered{NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D1<:Distribution, D2<:Distribution, D3<:Distribution, FT<:Function}

A type to store the main parameters for simulating a generalized Lotka-Volterra system with disordered couplings on a random graph.

# Fields
$(TYPEDFIELDS)
"""
struct ModelDisordered{NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D1<:Distribution, D2<:Distribution, D3<:Distribution, FT<:Function}
    """Number of nodes in the system."""
    N::I
    """Average degree of the interaction graph."""
    K::RT
    """Number of discretized time steps for the simulation trajectories."""
    M::I
    """Function to generate the adjacency matrix"""
    generate_adj::FT
    """Distribution or sampler for the degree of the nodes. This should be a distribution object from the package Distributions.jl."""
    deg_pdf::D1
    """Distribution or sampler for the cavity degree of the nodes. This should be a distribution object from the package Distributions.jl."""
    deg_cav_pdf::D2
    """Average coupling strength"""
    m::RT
    """Variance of the coupling strength"""
    sigma2::RT
    """Correlation of the couplings strength"""
    corr::RT
    """Immigration rate (lambda)."""
    lam::RT
    """Distribution or sampler for initial conditions (p0) of trajectories. This should be a distribution object from the package Distributions.jl."""
    p0::D3
    """Noise of the system."""
    noise::NK
end



"""
    ModelDisorderedFC{NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D<:Distribution}

A type to store the main parameters for simulating a generalized Lotka-Volterra system with disordered couplings on a fully-connected graph.

# Fields
$(TYPEDFIELDS)
"""
struct ModelDisorderedFC{NK<:AbstractNoiseKind, I<:Integer, RT<:Real, D<:Distribution}
    """Number of nodes in the system."""
    N::I
    """Number of discretized time steps for the simulation trajectories."""
    M::I
    """Average coupling strength"""
    m::RT
    """Variance of the coupling strength"""
    sigma2::RT
    """Correlation of the couplings strength"""
    corr::RT
    """Immigration rate (lambda)."""
    lam::RT
    """Distribution or sampler for initial conditions (p0) of trajectories. This should be a distribution object from the package Distributions.jl."""
    p0::D
    """Noise of the system."""
    noise::NK
end



"""
    Model{NK<:AbstractNoiseKind, I<:Integer, RT<:Real, MT<:AbstractMatrix{RT}}<

A type to store the main parameters for simulating a generalized Lotka-Volterra on a graph.

# Fields
$(TYPEDFIELDS)
"""
struct Model{NK<:AbstractNoiseKind, I<:Integer, RT<:Real, MT<:AbstractMatrix{RT}, D<:Distribution}
    """Number of nodes in the system."""
    N::I
    """Number of discretized time steps for the simulation trajectories."""
    M::I
    """Adjacency matrix representing the interactions between nodes.
    Can be a dense or sparse matrix with Real-valued elements."""
    J::MT
    """Immigration rate (lambda)."""
    lam::RT
    """Distribution or sampler for initial conditions (p0) of trajectories.
    This could be a function (e.g., `rng -> initial_state`), a distribution
    object from a package like Distributions.jl, or a custom sampler type.
    """
    p0::D
    """Noise of the system."""
    noise::NK
end
