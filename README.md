# Embeddings


[![Build Status](https://travis-ci.org/JuliaText/Embeddings.jl.svg?branch=master)](https://travis-ci.org/JuliaText/Embeddings.jl)
[![CodeCov](https://codecov.io/gh/JuliaText/Embeddings.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaText/Embeddings.jl)



## Introduction

Word Embeddings present words as high-dimensional vectors, where every dimension corresponds to some latent feature [1]. This makes it possible to utilize different mathematical operations between words. With these we can discover semantic relationships between words. E.g. when using [Word2Vec](https://code.google.com/archive/p/word2vec/) embeddings and utilizing cosine similarity between vectors, the calculation `vector(“Madrid”) - vector(“Spain”) + vector(“France”)` gives as an answer the vector for word “Paris” [2]. 
Pretraining Word Embeddings are commonly uses to initialize the bottom layer of a more advanced NLP method, such as a LSTM [3].
Simply summing the embeddings in a sentence or phrase can in and of itself be a surprisingly powerful way to represent the sentence/phrase, and can be used as a input to simple ML models like SVM 4]. 

This package gives access to pretrained embeddings. At its current state it includes following word embeddings: [Word2Vec](https://code.google.com/archive/p/word2vec/) (English), [GloVe](https://nlp.stanford.edu/projects/glove/) (English), and [FastText](https://fasttext.cc/) (hundreds of languages). 

### Installation
The package can be installed using the [julia package manager in the normal way.](https://julialang.github.io/Pkg.jl/v1/managing-packages/#Adding-packages-1).

Open the REPL, press <kbd>]</kbd> to enter package mode, and then:

```julia
pkg> add Embeddings
```
There are no further steps.
Pretrained embeddings will be downloaded the first time you use them.


## Details


### `load_embeddings`

    load_embeddings(EmbeddingSystem, [embedding_file|default_file_number])
    load_embeddings(EmbeddingSystem{:lang}, [embedding_file|default_file_number])

Loaded the embeddings from a embedding file.
The embeddings should be of the type given by the Embedding system.

If the `embedding file` is not provided, a default embedding file will be used.
(It will be automatically installed if required).
EmbeddingSystems have a language type parameter.
For example `FastText_Text{:fr}` or `Word2Vec{:en}`, if that language parameter is not given it defaults to English.
(I am sorry for the poor state of the NLP field that many embedding formats are only available pretrained in English.)
Using this the correct  default embedding file will be installed for that language.
For some languages and embedding systems there are multiple possible files.
You can check the list of them using for example `language_files(FastText_Text{:de})`.
The first is nominally the most popular, but if you want to default to another you can do so by setting the `default_file_num`.

### This returns an `EmbeddingTable` object.
This has 2 fields.

 - `embeddings` is a matrix, each column is the embedding for a word.
 - `vocab` is a vector of strings, ordered as per the columns of `embeddings`, such that the first string in vocab is the first column of `embeddings` etc

We do not include a method for getting the index of a column from a word.
This is trivial to define in code (`vocab2ind(vocab)=Dict(word=>ii for (ii,word) in enumerate(vocab))`),
and you might like to be doing this in a more consistant way, e.g using [MLLabelUtils.jl](https://github.com/JuliaML/MLLabelUtils.jl),
or you might like to build a much faster Dict solution on top of [InternedStrings.jl](https://github.com/JuliaString/InternedStrings.jl)


## Configuration
This package is build on top of [DataDeps.jl](https://github.com/oxinabox/DataDeps.jl).
To configure, e.g., where downloaded files save to, and read from (and to understand how that works),
see the DataDeps.jl readme.


## Examples

Load the package with

```
julia> using Embeddings
```
### Basic example
The Following script loads up the embeddings,
and defines a `Dict` to map from vocabulary word to index, in the embedding matrix,
and a function that used it to get an embedding vector.
This is a basic way to access the embedding for a word.

```
using Embeddings
const embtable = load_embeddings(Word2Vec) # or load_embeddings(FastText_Text) or ...

const get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

function get_embedding(word)
    ind = get_word_index[word]
    emb = embtable.embeddings[:,ind]
    return emb
end
```

This can be used like so:
```
julia> get_embedding("blue")
300-element Array{Float32,1}:
  0.01540828
  0.03409082
  0.0882124
  0.04680265
 -0.03409082
...
```

### Loading different Embeddings

load up the default word2vec embeddings:
```
julia> load_embeddings(Word2Vec) 
Embeddings.EmbeddingTable{Array{Float32,2},Array{String,1}}(Float32[0.0673199 0.0529562 … -0.21143 0.0136373; -0.0534466 0.0654598 … -0.0087888 -0.0742876; … ; -0.00733469 0.0108946 … -0.00405157 0.0156112; -0.00514565 -0.0470722 … -0.0341579 0.0396559], String["</s>", "in", "for", "that", "is", "on", "##", "The", "with", "said"  …  "#-###-PA-PARKS", "Lackmeyer", "PERVEZ", "KUNDI", "Budhadeb", "Nautsch", "Antuane", "tricorne", "VISIONPAD", "RAFFAELE"])
```


Load up the first 100 embeddings from the default French FastText embeddings:
```
julia> load_embeddings(FastText_Text{:fr}; max_vocab_size=100) 
Embeddings.EmbeddingTable{Array{Float32,2},Array{String,1}}(Float32[0.0058 -0.0842 … -0.062 -0.0687; 0.0478 -0.0388 … 0.0064 -0.339; … ; 0.023 -0.0106 … -0.022 -0.1581; 0.0378 0.0579 … 0.0417 0.0714], String[",", "de", ".", "</s>", "la", "et", ":", "à", "le", "\""  …  "faire", "c'", "aussi", ">", "leur", "%", "si", "entre", "qu", "€"])
```


List all the default files for FastText in English:
```
julia> language_files(FastText_Text{:en}) # List all the possible default files for FastText in English
3-element Array{String,1}:
 "FastText Common Crawl/crawl-300d-2M.vec"
 "FastText Wiki News/wiki-news-300d-1M.vec"
 "FastText en Wiki Text/wiki.en.vec"
```

From the second of those default files, load the embeddings just for "red", "green", and "blue": 
```
julia> load_embeddings(FastText_Text{:en}, 2; keep_words=Set(["red", "green", "blue"]))
Embeddings.EmbeddingTable{Array{Float32,2},Array{String,1}}(Float32[-0.0054 0.0404 -0.0293; 0.0406 0.0621 0.0224; … ; 0.218 0.1542 0.2256; 0.1315 0.1528 0.1051], String["red", "green", "blue"])
```

List all the default files for GloVe in English:
```
julia> language_files(GloVe{:en})
10-element Array{String,1}:
 "glove.6B/glove.6B.50d.txt"
 "glove.6B/glove.6B.100d.txt"
 "glove.6B/glove.6B.200d.txt"
 "glove.6B/glove.6B.300d.txt"
 "glove.42B.300d/glove.42B.300d.txt"
 "glove.840B.300d/glove.840B.300d.txt"
 "glove.twitter.27B/glove.twitter.27B.25d.txt"
 "glove.twitter.27B/glove.twitter.27B.50d.txt"
 "glove.twitter.27B/glove.twitter.27B.100d.txt"
 "glove.twitter.27B/glove.twitter.27B.200d.txt"
```

Load the 200d GloVe embedding matrix for the top 10000 words trained on 6B words:
```
julia> glove = load_embeddings(GloVe{:en}, 3, max_vocab_size=10000)
Embeddings.EmbeddingTable{Array{Float32,2},Array{String,1}}(Float32[-0.071549 0.17651 … 0.19765 -0.22419; 0.093459 0.29208 … -0.31357 0.039311; … ; 0.030591 -0.23189 … -0.72917 0.49645; 0.25577 -0.10814 … 0.07403 0.41581], ["the", ",", ".", "of", "to", "and", "in", "a", "\"", "'s"  …  "slashed", "23-year", "communique", "hawk", "necessity", "petty", "stretching", "taxpayer", "resistant", "quinn"])

julia> size(glove)
(200, 10000)
```

## Contributing
Contributions, in the form of bug-reports, pull requests, additional documentation are encouraged.
They can be made to the Github repository.

**All contributions and communications should abide by the [Julia Community Standards](https://julialang.org/community/standards/).**

The following software contributions would particularly be appreciated:

 - Provide Hashstrings: I have only filled in the checksums for the FastText Embeddings that I have downloaded, which is only a small fraction. If you using embeddings files for a language that doesn't have its hashstring set, then DataDeps.jl will tell you the hashstring that need to be added to the file. It is a quick and easy PR.
 - Provide Implementations of other loaders: If you have implementations of code to load another format (e.g. Binary FastText) it would be great if you could contribute them. I know I have a few others kicking around somewhere.

Software contributions should follow the prevailing style within the code-base.
If your pull request (or issues) are not getting responses within a few days do not hesitate to "bump" them,
by posting a comment such as "Any update on the status of this?".
Sometimes Github notifications get lost.
 
## Support

Feel free to ask for help on the [Julia Discourse forum](https://discourse.julialang.org/),
or in the `#natural-language` channel on julia-slack. (Which you can [join here](https://slackinvite.julialang.org/)).
You can also raise issues in this repository to request improvements to the documentation.

## Sources
[1]: [Turian, Joseph, Lev Ratinov, and Yoshua Bengio. "Word representations: a simple and general method for semi-supervised learning." Proceedings of the 48th annual meeting of the association for computational linguistics. Association for Computational Linguistics, 2010.](https://www.aclweb.org/anthology/P10-1040/)

[2]: [Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

[3]: [White, Lyndon et al. Neural Representations of Natural Language. Springer: Studies in Computational Intelligence. 2018.](https://www.springer.com/us/book/9789811300615)

[4]: [White, Lyndon. On the surprising capacity of linear combinations of embeddings for natural language processing.
Doctorial Thesis, The University of Western Australia. 2019](https://research-repository.uwa.edu.au/en/publications/on-the-surprising-capacity-of-linear-combinations-of-embeddings-f)
