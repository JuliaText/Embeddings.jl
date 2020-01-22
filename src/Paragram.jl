abstract type Paragram{LANG} <: EmbeddingSystem{LANG} end
using GoogleDrive
using Random: randstring
using HTTP 

function init(::Type{Paragram})
    vectors = [("paragram_300_ws353",
                " Paragram-WS353 300 dimensional Paragram embeddings tuned on WordSim353 dataset. 1.7 GB download.",
                "8ed9a19f8bc400cdbca5dae7f024c0310a3e5711a46ba48036a7542614440721",
              "https://drive.google.com/uc?id=0B9w48e1rj-MOLVdZRzFfTlNsem8&export=download"),
                # "https://drive.google.com/file/d/0B9w48e1rj-MOLVdZRzFfTlNsem8/view"),
               ("paragram_300_sl999",
                "Paragram-SL999 300 dimensional Paragram embeddings tuned on SimLex999 dataset. 1.7 GB download.",
                "9a16adc7d620642f863278451db4c03a2646016440ccea7e30a37ba17868781d",
               "https://drive.google.com/uc?id=0B9w48e1rj-MOck1fRGxaZW1LU2M&export=download"),
]
#https://drive.google.com/file/d/0B9w48e1rj-MOLVdZRzFfTlNsem8/view?usp=sharing
    for (depname, description, sha, link) in vectors
        register(DataDep(depname,
                         """
                         Pretrained Paragram word embeddings.
                         Website: https://www.cs.cmu.edu/~jwieting/
                         Author: John Wieting
                         Year: 2015
                         Licence: Open Data Commons Public Domain Dedication and License (PDDL)

                         Paper:
                             John Wieting, Mohit Bansal, Kevin Gimpel, Karen Livescu, Dan Roth. From Paraphrase Database to Compositional Paraphrase Model and Back.

                         $description
                         """,
                         link,
                         sha,
                         fetch_method = google_download,
			 post_fetch_method = unpack))
       
            append!(language_files(Paragram{:en}), ["$(depname)/$(depname).txt"])
       
            
      
    end
end

_load_embeddings(::Type{<:Paragram}, embedding_file, max_vocab_size, keep_words) = _load_embeddings_csv( embedding_file, max_vocab_size, keep_words)


