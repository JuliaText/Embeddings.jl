abstract type Word2Vec{LANG} <: EmbeddingSystem{LANG} end


function init(::Type{Word2Vec})
    register(DataDep("word2vec 300d",
    """
    Pretrained Word2Vec Word emeddings
    Website: https://code.google.com/archive/p/word2vec/
    Author: Mikolov et al.
    Year: 2013

    Pre-trained vectors trained on part of Google News dataset (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases.

    Paper:
        Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.
    """,
    "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz",
    "21c05ae916a67a4da59b1d006903355cced7de7da1e42bff9f0504198c748da8";
    post_fetch_method=DataDeps.unpack
    )) 
    
    push!(language_files(Word2Vec{:en}), "word2vec 300d/GoogleNews-vectors-negative300.bin")
end


function _load_embeddings(::Type{<:Word2Vec}, embedding_file, max_vocab_size, keep_words)
    local LL, indexed_words, index
    open(embedding_file,"r") do fh
        vocab_size, vector_size = parse.(Int64, split(readline(fh)))
        max_stored_vocab_size = min(max_vocab_size, vocab_size)

        indexed_words = Vector{String}(undef, max_stored_vocab_size)
        LL = Array{Float32}(undef, vector_size, max_stored_vocab_size)

        index = 1
        @inbounds for _ in 1:vocab_size
            word = readuntil(fh, ' '; keep=false)
            vector = Vector{Float32}(undef, vector_size)
            @inbounds for i = 1:vector_size
                vector[i] = read(fh, Float32)
            end

            if isempty(keep_words) || word ∈ keep_words
                LL[:, index] = vector ./ norm(vector)
                indexed_words[index] = word

                index += 1
                if index > max_stored_vocab_size
                    break
                end
            end

        end
    end

    LL = LL[:,1:index-1] #throw away unused columns
    indexed_words = indexed_words[1:index-1] #throw away unused columns
    LL, indexed_words
end

