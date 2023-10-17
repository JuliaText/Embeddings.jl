abstract type GloVe{LANG} <: EmbeddingSystem{LANG} end

const glove_max_size = 4_000_000

function init(::Type{GloVe})
    vectors = [("glove.6B",
                "Trained on 6B tokens (Wikipedia 2014, Gigaword 5), 400K vocab, uncased. Includes 50d, 100d, 200d, & 300d vectors. 822 MB download.",
                "617afb2fe6cbd085c235baf7a465b96f4112bd7f7ccb2b2cbd649fed9cbcf2fb",
                ["50d", "100d", "200d", "300d"]),
               ("glove.42B.300d",
                "Trained on 42B tokens (Common Crawl), 1.9M vocab, uncased, Includes 300d vectors. 1.75 GB download.",
                "03d5d7fa28e58762ace4b85fb71fe86a345ef0b5ff39f5390c14869da0fc1970",
               []),
               ("glove.840B.300d",
                "Trained on 840B tokens (Common Crawl), 2.2M vocab, cased. Includes 300d vectors. 2.03 GB download.",
                "c06db255e65095393609f19a4cfca20bf3a71e20cc53e892aafa490347e3849f",
                []),
               ("glove.twitter.27B",
                "Trained on 2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download.",
                "792af52f795d1a32c9842a3240f5f3fe5e941a8ff6df5eb0f9d668092ebc019c",
                ["25d", "50d", "100d", "200d"])]
    for (depname, description, sha, dims) in vectors
        register(DataDep(depname,
                         """
                         Pretrained GloVe word embeddings.
                         Website: https://nlp.stanford.edu/projects/glove/
                         Author: Jeffrey Pennington, Richard Socher, Christopher D. Manning 
                         Year: 2014
                         Licence: Open Data Commons Public Domain Dedication and License (PDDL)

                         Paper:
                             Jeffrey Pennington, Richard Socher, Christopher D. Manning. GloVe: Global Vectors for Word Representation. In Proceedings of EMNLP, 2014.

                         $description
                         """,
                         "https://nlp.stanford.edu/data/$(depname).zip",
                         sha,
                         post_fetch_method = unpack))
        if length(dims) >= 1
            append!(language_files(GloVe{:en}), ["$(depname)/$(depname).$(dim).txt" for dim in dims])
        else
            push!(language_files(GloVe{:en}), "$depname/$depname.txt")
        end
    end
end

_load_embeddings(::Type{<:GloVe}, embedding_file::IO, max_vocab_size, keep_words) = _load_embeddings_csv(embedding_file, max_vocab_size, keep_words, ' ')

