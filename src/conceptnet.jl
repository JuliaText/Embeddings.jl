abstract type ConceptNet{LANG} <: EmbeddingSystem{LANG} end


function init(::Type{ConceptNet})
    for (source, link, file, hashstring, post_fetch_method, lang) in [
            ("Multilingual",
             "https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-17.06.txt.gz",
             "numberbatch-17.06.txt",
             "375ac41e8f9caab172fc169e54e8658be23ece6cd92188bf700a99330cd1a81b",
             DataDeps.unpack,
             :multi),
            ("English",
             "https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-17.06.txt.gz",
             "numberbatch-en-17.06.txt",
             "72faf0a487c61b9a6a8c9ff0a1440d2f4936bb19102bddf27a833c2567620f2d",
             DataDeps.unpack,
             :en),
            ("Compressed",
             "http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/17.06/mini.h5",
             "mini.h5",
             "87f8b48fd01088b72013dfbcc3160f0d0e16942797959e0824b9b69f31a7b222",
             identity,
             :compressed)
        ]
        register(DataDep("ConceptNet_$source",
        """
        Pretrained ConceptNetNumberbatch Word emeddings
        Website: https://github.com/commonsense/conceptnet-numberbatch
        License: CC-SA 4.0
        Author: Luminoso Technologies Inc.
        Year: 2017

        ConceptNet Numberbatch consists of state-of-the-art semantic vectors (also known as word embeddings) that can be used directly as a representation of word meanings or as a starting point for further machine learning.

        Paper:
            Robert Speer, Joshua Chin, and Catherine Havasi (2017). "ConceptNet 5.5: An Open Multilingual Graph of General Knowledge." In proceedings of AAAI 2017.
        """,
        link,
        hashstring;
        post_fetch_method=post_fetch_method
        ))

        push!(language_files(ConceptNet{lang}), "ConceptNet_$(source)/$(file)")
    end
end


function _load_embeddings(::Type{<:ConceptNet}, embedding_file, max_vocab_size, keep_words)
    local LL, indexed_words, index
    if any(endswith.(embedding_file, [".h5", ".hdf5"]))
        LL, indexed_words = _load_hdf5_embeddings(embedding_file,
                                                  max_vocab_size=max_vocab_size,
                                                  keep_words=keep_words)
    else
        LL, indexed_words = _load_txt_embeddings(embedding_file,
                                                 max_vocab_size=max_vocab_size,
                                                 keep_words=keep_words)
    end
    return LL, indexed_words
end


# Function that calculates how many embeddings to retreive
function _get_vocab_size(real_vocab_size, max_vocab_size)
    real_vocab_size = max(0, real_vocab_size)
    max_vocab_size = min(real_vocab_size, max_vocab_size)
    return max_vocab_size
end


function _load_txt_embeddings(file::AbstractString, max_vocab_size, keep_words)
    open(file, "r") do fid
        vocab_size, vector_size = map(x->parse(Int,x), split(readline(fid)))
        max_stored_vocab_size = _get_vocab_size(vocab_size, max_vocab_size)
        data = readlines(fid)

        indexed_words = Vector{String}(undef, max_stored_vocab_size)
        LL = Array{Float32}(undef, vector_size, max_stored_vocab_size)
        _parseline = (buf)-> begin
            bufvec = split(buf, " ")
            word = string(popfirst!(bufvec))
            embedding = parse.(Float64, bufvec)
            #embedding = map(x->parse(Float64,x), bufvec)
            return word, embedding
        end

        cnt = 0
        for (index, row) in enumerate(data)
            word, embedding = _parseline(row)
            if length(keep_words)==0 || word in keep_words
                LL[:, index] = embedding
                idexed_words[index] = word
                cnt+=1
                if cnt > max_stored_vocab_size
                    break
                end
            end
        end
    end
    return LL, indexed_words
end


# Loads the ConceptNetNumberbatch from a HDF5 file
function _load_hdf5_embeddings(file::AbstractString, max_vocab_size, keep_words)
    payload = h5open(read, file)["mat"]
    words = payload["axis1"]
    vectors = payload["block0_values"]
    max_vocab_size = _get_vocab_size(length(words), max_vocab_size)

    indices = Int[]
    for (index, word) in enumerate(words)
        if length(keep_words)==0 || word in keep_words
            push!(indices, index)
            cnt+=1
            if cnt > max_stored_vocab_size
                break
            end
        end
    end
    return vectors[:, indices], words[indices]
end
