"""
Module for "RandomLotkaVolterraCavity.jl" -- A Julia package for the study of the fixed-point structure of the Lotka-Volterra model on random graphs with cavity method.

# Exports

$(EXPORTS)

"""
module RandomLotkaVolterraCavity
    using DocStringExtensions, Random, StatsBase, SpecialFunctions, ProgressMeter, SparseArrays, Graphs, OrdinaryDiffEq, LinearAlgebra, PyPlot, PyCall

    export  population_dynamics, population_dynamics!, population_dynamics_t, population_dynamics_t!, sample_glv, sample_couplings, sample_x, PdfDegVec, sample_degree, get_index

    include("utils.jl")
    include("population_dynamics.jl")
end