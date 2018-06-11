using PretrainedEmbeddings
using Base.Test

"""
    tempdatadeps(fun)

Run the function and delete all created datadeps afterwards
"""
function tempdatadeps(fun)
    tempdir = mktempdir()
    try
        info("sending all datadeps to $tempdir")
        withenv("DATADEPS_LOAD_PATH"=>tempdir) do
            fun()
        end
    finally
        try
            info("removing $tempdir")
            rm(tempdir, recursive=true, force=true)
        catch err
            warn("Something went wrong with removing $tempdir")
            warn(err)
        end
        run(`df -h`)
    end
    
end

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
