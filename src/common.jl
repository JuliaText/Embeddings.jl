
function _load_embeddings_csv( embedding_file, max_vocab_size, keep_words, Delim::AbstractChar=' ')
    local LL, indexed_words, index
    if length(keep_words) > 0
        max_vocab_size = length(keep_words)
    end
    indexed_words = Vector{String}()
    LL = Vector{Vector{Float32}}()
    open(embedding_file) do f
        index = 1
        for line in eachline(f)
            xs = split(line, Delim)
            word = xs[1]
            if length(keep_words) == 0 || (word in keep_words)
                index > max_vocab_size && break
                push!(indexed_words, word)
                try
                    push!(LL, parse.(Float32, @view(xs[2:end])))
                catch err
                    err isa ArgumentError || rethrow()
                    @warn "Could not parse word vector" index word exception=err
                end
                index += 1
            end
        end
    end
    return reduce(hcat, LL), indexed_words
end
