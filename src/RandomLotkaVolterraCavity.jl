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
    export Deterministic, CavityFP, MarginalFP, NodeFP, PopFP, PopJ, run_GECaM_FP, analytic_FC, NodeFP_IBMF, run_IBMF_FP, CavityFP_BP, MarginalFP_BP, NodeFP_BP, PopFP_BP, run_BP_FP
    
    include("types.jl")
    include("sample.jl")
    include("deterministic/types.jl")
    include("deterministic/utils.jl")
    include("deterministic/GECaM_FP.jl")
    include("deterministic/FokkerPlanck.jl")
end