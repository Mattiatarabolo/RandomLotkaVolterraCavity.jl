"""
    PdfDegVec

Structure to store the degree distribution of a network.

# Fields

$(TYPEDFIELDS)

"""
struct PdfDegVec
    """Probability distribution of the degree."""
    pdf::Vector{Float64}
    """Vector of degrees."""
    deg::Vector{Int}
    """Minimum degree."""
    kmin::Int
    """Maximum degree."""
    kmax::Int
    """Average degree."""
    K::Union{Float64,Int64}
    """Dictionary to map degree to index."""
    index_dict::Dict{Int, Int}  # Store indices instead of pdf values
    @doc """
        PdfDegVec(pdf_deg::Function, deg::Vector{Int})

    Constructs a `PdfDegVec` structure from a degree distribution function.

    Arguments:
    - `pdf_deg::Function`: Function that returns the probability of a given degree.
    - `deg::Vector{Int}`: Vector of degrees.

    Returns:
    - `PdfDegVec`: Degree distribution structure.
    """
    function PdfDegVec(pdf_deg::Function, deg::Vector{Int})
        pdf_vals = pdf_deg.(deg)
        K = sum(pdf_vals .* deg)
        index_map = Dict(deg .=> eachindex(deg))  # Map degree to index
        new(pdf_vals, deg, minimum(deg), maximum(deg), K, index_map)
    end

    @doc """
        PdfDegVec(pdf_deg::Function, deg::Vector{Int}, K::Union{Float64,Int64})

    Constructs a `PdfDegVec` structure from a degree distribution function.

    Arguments:
    - `pdf_deg::Function`: Function that returns the probability of a given degree.
    - `deg::Vector{Int}`: Vector of degrees.
    - `K::Union{Float64,Int64}`: Average degree.

    Returns:
    - `PdfDegVec`: Degree distribution structure.
    """
    function PdfDegVec(pdf_deg::Function, deg::Vector{Int}, K::Union{Float64,Int64})
        pdf_vals = pdf_deg.(deg)
        index_map = Dict(deg .=> eachindex(deg))  # Map degree to index
        new(pdf_vals, deg, minimum(deg), maximum(deg), K, index_map)
    end
end