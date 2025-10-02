"""
Module for "RandomLotkaVolterraCavity.jl" -- A Julia package for the study of the fixed-point structure of the Lotka-Volterra model on random graphs with cavity method.

# Exports

$(EXPORTS)

"""
module RandomLotkaVolterraCavity
    using DocStringExtensions, Dates, ProgressMeter
    using Random, SparseArrays, LinearAlgebra, StatsBase, Distributions, Graphs, SpecialFunctions
    using OrdinaryDiffEq, DiffEqCallbacks

    export AbstractNoiseKind, ModelDisordered, ModelDisorderedFC, Model, sample_couplings, run_MC, run_MC_ODE
    export Deterministic, CavityFP, MarginalFP, NodeFP, PopFP, PopJ, run_GECaM_FP, analytic_FC
    
    include("types.jl")
    include("sample.jl")
    include("sample_ODE.jl")
    include("deterministic/types.jl")
    include("deterministic/utils.jl")
    include("deterministic/GECaM_FP.jl")
end