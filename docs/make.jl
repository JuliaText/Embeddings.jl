using Documenter, PretrainedEmbeddings

makedocs(;
    modules=[PretrainedEmbeddings],
    format=:html,
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/JuliaText/PretrainedEmbeddings.jl/blob/{commit}{path}#L{line}",
    sitename="PretrainedEmbeddings.jl",
    authors="Lyndon White (aka oxinabox)",
    assets=[],
)

deploydocs(;
    repo="github.com/JuliaText/PretrainedEmbeddings.jl",
    target="build",
    julia="0.6",
    deps=nothing,
    make=nothing,
)
