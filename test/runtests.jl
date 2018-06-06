using PretrainedEmbeddings
using Base.Test

@testset "Word2Vec" begin
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
    @testset "Basic" begin
        embs1 = load_embeddings(FastText_Text; max_vocab_size=100)
        @test length(embs1.vocab)==100
        @test size(embs1.embeddings) == (300, 100)
    end
        
    @testset "French" begin
        embs_fr = load_embeddings(FastText_Text{:fr}; max_vocab_size=100)
        @test length(embs_fr.vocab)==100
        @test size(embs_fr.embeddings) == (300, 100)
    end
    
    @testset "Specific" begin
        @testset "basic" begin
            embs_specific =  load_embeddings(FastText_Text; keep_words=Set(["red", "green", "blue"]))
            @test size(embs_specific.embeddings) == (300, 3)
            @test Set(embs_specific.vocab) == Set(["red", "green", "blue"])
        end
        
        @testset "file number 2" begin
            embs_specific =  load_embeddings(FastText_Text, 2; keep_words=Set(["red", "green", "blue"]))
            @test size(embs_specific.embeddings) == (300, 3)
            @test Set(embs_specific.vocab) == Set(["red", "green", "blue"])
        end   
    end

end