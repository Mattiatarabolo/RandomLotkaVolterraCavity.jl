using Documenter, DocStringExtensions
using RandomLotkaVolterraCavity

push!(LOAD_PATH,"../src/")
makedocs(
    modules=[RandomLotkaVolterraCavity],
    authors="Mattia Tarabolo <mattia.tarabolo@gmail.com> and contributors",
    sitename="RandomLotkaVolterraCavity.jl Documentation",
    format=Documenter.HTML(prettyurls = false),
    pages=[
        "Home" => "index.md",
        "Guide" => "guide.md",
        "Functions" => "functions.md"
    ],
)

deploydocs(;
    repo="github.com/Mattiatarabolo/RandomLotkaVolterraCavity.jl.git",
    devbranch="main",
)
