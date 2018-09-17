using Embeddings
using Test

using DataDeps

"""
    tempdatadeps(fun)

Run the function and delete all created datadeps afterwards
"""
function tempdatadeps(fun)
    tempdir = mktempdir()
    try
        @info "sending all datadeps to $tempdir"
        withenv("DATADEPS_LOAD_PATH"=>tempdir) do
            fun()
        end
    finally
        try
            @info "removing $tempdir"
            rm(tempdir, recursive=true, force=true)
        catch err
            @warn "Something went wrong with removing tempdir" tempdir exception=err
        end
    end   
end


# uncomment the below to not use tempdatadeps (i.e. if not debugging)
# tempdatadeps(fun) = fun()



"""
@testset_nokeep_data

Use just like @testset,
but know that it deletes any downloaded data dependencies when it is done.
"""
macro testset_nokeep_data(name, expr)
    quote
        tempdatadeps() do
            @testset $name $expr
        end
    end |> esc
end


@testset_nokeep_data "Word2Vec" begin
    embs_full = load_embeddings(Word2Vec)

    @test size(embs_full.embeddings) == (300, length(embs_full.vocab))

    embs_mini = load_embeddings(Word2Vec; max_vocab_size=100)
    @test length(embs_mini.vocab)==100

    @test embs_mini.embeddings == embs_full.embeddings[:, 1:100]
    @test embs_mini.vocab == embs_full.vocab[1:100]

    @test "for" ∈ embs_mini.vocab

    embs_specific =  load_embeddings(Word2Vec; keep_words=Set(["red", "green", "blue"]))

    @test size(embs_specific.embeddings) == (300, 3)
    @test Set(embs_specific.vocab) == Set(["red", "green", "blue"])
end

@testset "GloVe" begin
    # just test one file from each of provided sets
    tests = ["glove.6B/glove.6B.50d.txt",
             #"glove.42B.300d/glove.42B.300d.txt",     # These files are too slow to download
             #"glove.840B.300d/glove.840B.300d.txt",   # They are not that big bt are on a slow server
             "glove.twitter.27B/glove.twitter.27B.25d.txt"]

    # read dimensionality from name (e.g. glove.6B.300d.txt -> 300)
    dim(x) = parse(Int, match(r"\.([0-9]+)d\.", x).captures[1])

    for file in tests
        filename = split(file, Base.Filesystem.path_separator)[end]

        @testset_nokeep_data "$filename" begin
            @testset "Basic" begin
                glove = load_embeddings(GloVe{:en}, @datadep_str(file), max_vocab_size=1000)
                @test length(glove.vocab) == 1000
                @test size(glove.embeddings) == (dim(file), 1000)
                @test "for" ∈ glove.vocab
            end

            @testset "Specific" begin
                colors = ["red", "green", "blue"]
                glove_colors = load_embeddings(GloVe, @datadep_str(file), keep_words=colors)
                @test length(glove_colors.vocab) == 3
                @test size(glove_colors.embeddings) == (dim(file), 3)
                @test Set(glove_colors.vocab) == Set(colors)
            end
        end
    end

    @testset "Custom" begin
        # first 100 lines of official glove.6B.50d.txt
        custom_glove_file = joinpath(@__DIR__, "data", "custom.glove.txt")
        @testset "Basic" begin
            glove = load_embeddings(GloVe, custom_glove_file)
            @test length(glove.vocab) == 100
            @test size(glove.embeddings) == (50, 100)
            @test "the" ∈ glove.vocab
        end
        @testset "Specific" begin
            punct = [".", ","]
            glove_punct = load_embeddings(GloVe, custom_glove_file, keep_words=punct)
            @test length(glove_punct.vocab) == 2
            @test size(glove_punct.embeddings) == (50, 2)
            @test Set(glove_punct.vocab) == Set(punct)
        end
    end

end

@testset "FastText" begin
    @testset_nokeep_data "English 1" begin
        @testset "Basic" begin
            embs1 = load_embeddings(FastText_Text; max_vocab_size=100)
            @test length(embs1.vocab)==100
            @test size(embs1.embeddings) == (300, 100)
        end
            
        @testset "Specific" begin           
            embs_specific =  load_embeddings(FastText_Text; keep_words=Set(["red", "green", "blue"]))
            @test size(embs_specific.embeddings) == (300, 3)
            @test Set(embs_specific.vocab) == Set(["red", "green", "blue"])
        end
    end

        
    @testset_nokeep_data "French" begin
        embs_fr = load_embeddings(FastText_Text{:fr}; max_vocab_size=100)
        @test length(embs_fr.vocab)==100
        @test size(embs_fr.embeddings) == (300, 100)
    end
    

        
    @testset_nokeep_data "English file number 2" begin
        embs_specific =  load_embeddings(FastText_Text, 2; keep_words=Set(["red", "green", "blue"]))
        @test size(embs_specific.embeddings) == (300, 3)
        @test Set(embs_specific.vocab) == Set(["red", "green", "blue"])
    end   

end

@testset "ConceptNet" begin
    @testset_nokeep_data "Multilingual" begin
        embs_full = load_embeddings(ConceptNet{:multi})

        @test size(embs_full.embeddings) == (300, length(embs_full.vocab))

        embs_mini = load_embeddings(ConceptNet{:multi}; max_vocab_size=100)
        @test length(embs_mini.vocab)==100

        @test embs_mini.embeddings == embs_full.embeddings[:, 1:100]
        @test embs_mini.vocab == embs_full.vocab[1:100]
    end

    @testset_nokeep_data "English" begin
        embs_full = load_embeddings(ConceptNet{:en})

        @test size(embs_full.embeddings) == (300, length(embs_full.vocab))

        embs_mini = load_embeddings(ConceptNet{:en}; max_vocab_size=100)
        @test length(embs_mini.vocab)==100

        @test embs_mini.embeddings == embs_full.embeddings[:, 1:100]
        @test embs_mini.vocab == embs_full.vocab[1:100]

        embs_specific =  load_embeddings(ConceptNet{:en};
                                         keep_words=Set(["red", "green", "blue"]))

        @test size(embs_specific.embeddings) == (300, 3)
        @test Set(embs_specific.vocab) == Set(["red", "green", "blue"])
    end

    @testset_nokeep_data "Compressed" begin
        embs_full = load_embeddings(ConceptNet{:compressed})

        @test size(embs_full.embeddings) == (300, length(embs_full.vocab))

        embs_mini = load_embeddings(ConceptNet{:compressed}; max_vocab_size=100)
        @test length(embs_mini.vocab)==100

        @test embs_mini.embeddings == embs_full.embeddings[:, 1:100]
        @test embs_mini.vocab == embs_full.vocab[1:100]

        embs_specific =  load_embeddings(ConceptNet{:compressed};
                                         keep_words=Set(["/c/en/red", "/c/en/green", "/c/en/blue"]))

        @test size(embs_specific.embeddings) == (300, 3)
        @test Set(embs_specific.vocab) == Set(["/c/en/red", "/c/en/green", "/c/en/blue"])
    end
end
