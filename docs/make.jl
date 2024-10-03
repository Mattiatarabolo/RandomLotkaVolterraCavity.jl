using RandomLotkaVolterraCavity
using Documenter

DocMeta.setdocmeta!(RandomLotkaVolterraCavity, :DocTestSetup, :(using RandomLotkaVolterraCavity); recursive=true)

makedocs(;
    modules=[RandomLotkaVolterraCavity],
    authors="Mattia Tarabolo <mattia.tarabolo@gmail.com> and contributors",
    sitename="RandomLotkaVolterraCavity.jl",
    format=Documenter.HTML(;
        canonical="https://Mattia Tarabolo.github.io/RandomLotkaVolterraCavity.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Mattia Tarabolo/RandomLotkaVolterraCavity.jl",
    devbranch="main",
)
