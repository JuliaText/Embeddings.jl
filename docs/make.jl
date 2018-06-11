using Documenter, Embeddings

makedocs(;
    modules=[Embeddings],
    format=:html,
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/JuliaText/Embeddings.jl/blob/{commit}{path}#L{line}",
    sitename="Embeddings.jl",
    authors="Lyndon White (aka oxinabox)",
    assets=[],
)

deploydocs(;
    repo="github.com/JuliaText/Embeddings.jl",
    target="build",
    julia="0.6",
    deps=nothing,
    make=nothing,
)
