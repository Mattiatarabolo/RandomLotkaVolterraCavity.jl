"""
Module for "RandomLotkaVolterraCavity.jl" -- A Julia package for the study of the fixed-point structure of the Lotka-Volterra model on random graphs with cavity method.

# Exports

$(EXPORTS)

"""
module RandomLotkaVolterraCavity
    using DocStringExtensions, Dates, ProgressMeter
    using Random, SparseArrays, LinearAlgebra, StatsBase, Distributions, Graphs, SpecialFunctions
    # using Random, StatsBase, SpecialFunctions, ProgressMeter, SparseArrays, Graphs, OrdinaryDiffEq, LinearAlgebra, PyPlot, PyCall

    export AbstractNoiseKind, ModelDisordered, ModelDisorderedFC, Model, sample_couplings, run_MC
    export Deterministic, CavityFP, MarginalFP, NodeFP, PopFP, PopJ, run_GECaM_FP, analytic_FC
    #export population_dynamics, population_dynamics!, population_dynamics_t, population_dynamics_t!, sample_glv, sample_couplings, sample_x, PdfDegVec, sample_degree, get_index, analytic_FC

    include("types.jl")
    include("sample.jl")
    include("deterministic/types.jl")
    include("deterministic/utils.jl")
    include("deterministic/GECaM_FP.jl")
end