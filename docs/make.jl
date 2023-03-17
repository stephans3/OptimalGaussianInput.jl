using OptimalGaussianInput
using Documenter

DocMeta.setdocmeta!(OptimalGaussianInput, :DocTestSetup, :(using OptimalGaussianInput); recursive=true)

makedocs(;
    modules=[OptimalGaussianInput],
    authors="Stephan Scholz",
    repo="https://github.com/stephans3/OptimalGaussianInput.jl/blob/{commit}{path}#{line}",
    sitename="OptimalGaussianInput.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://stephans3.github.io/OptimalGaussianInput.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/stephans3/OptimalGaussianInput.jl",
    devbranch="main",
)
