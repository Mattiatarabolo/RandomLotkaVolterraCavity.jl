"""
Module for "RandomLotkaVolterraCavity.jl" -- A Julia package for the study of the fixed-point structure of the Lotka-Volterra model on random graphs with cavity method.

# Exports

$(EXPORTS)

"""
module RandomLotkaVolterraCavity
    using DocStringExtensions, Dates, ProgressMeter
    using Random, SparseArrays, LinearAlgebra, StatsBase, Distributions, Graphs, SpecialFunctions
    using OrdinaryDiffEq, DiffEqCallbacks

    export AbstractNoiseKind, ModelDisordered, ModelDisorderedFC, Model, sample_couplings, run_MC
    export Deterministic, CavityFP, MarginalFP, NodeFP, PopFP, PopJ, run_GECaM_FP, analytic_FC, CavityFP_q0, MarginalFP_q0, NodeFP_q0, PopFP_q0, run_GECaM_FP_q0
    
    include("types.jl")
    include("sample.jl")
    include("deterministic/types.jl")
    include("deterministic/utils.jl")
    include("deterministic/GECaM_FP.jl")
end