abstract type FastText{LANG} <: EmbeddingSystem{LANG} end
abstract type FastText_Text{LANG} <: FastText{LANG} end
abstract type FastText_Bin{LANG} <: FastText{LANG} end



function _load_embeddings(::Type{<:FastText_Bin}, embedding_file, max_vocab_size, keep_words)
    error("FastText Binary Format not supported. If anyone knows how to parse it please feel encouraged to make a PR.")
end


function _load_embeddings(::Type{<:FastText_Text}, embedding_file, max_vocab_size, keep_words)
   #If there are any words in keep_words, then only those are kept, otherwise all are kept
    local LL, indexed_words, index
    
    if length(keep_words) > 0
        max_stored_vocab_size = length(keep_words)
    end
        
    open(embedding_file,"r") do fh
        vocab_size, vector_size = parse.(Int64, split(readline(fh)))
        max_stored_vocab_size = min(max_vocab_size, vocab_size)
        
        indexed_words = Vector{String}(undef, max_stored_vocab_size)
        LL = Array{Float32}(undef, vector_size, max_stored_vocab_size)
        index = 1
        @inbounds for _ in 1:vocab_size
            line = readline(fh)
            toks = split(line)
            word = first(toks)
            if length(keep_words)==0 || word in keep_words
                indexed_words[index]=word
                LL[:,index] .= parse.(Float32, @view toks[2:end])
            
                index+=1
                if index>max_stored_vocab_size
                    break
                end
            end
        end
    end

    LL = LL[:,1:index-1] #throw away unused columns
    indexed_words = indexed_words[1:index-1] #throw away unused columns
    LL, indexed_words
end




function init(::Type{FastText})
    
    #########
    # English
    for (source, name, hashstring) in [
            ("Common Crawl", "crawl-300d-2M.vec", "5bfffffbabdab299d4c9165c47275e8f982807a6eaca37ee1f71d3a79ddb544d"),
            ("Wiki News", "wiki-news-300d-1M.vec","bdeb85f44892c505953e3654183e9cb0d792ee51be0992460593e27198d746f8")
        ]
        
        push!(language_files(FastText_Text{:en}), "FastText $(source)/$(name)")
        register(DataDep("FastText $(source)",
            """
            Dataset: FastText Word Embeddings for English (original release)
            Author: Bojanowski et. al. (Facebook)
            License: CC-SA 3.0
            Website: https://fasttext.cc/docs/en/english-vectors.html

            1 million 300 dimentional  word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens)
            Citation: P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information

            Notice: this file is ~ 1GB
            """,
            "https://s3-us-west-1.amazonaws.com/fasttext-vectors/$(name).zip",
            hashstring,
            post_fetch_method=DataDeps.unpack
        ));
    end
    
    #########################
    # Common Crawl
    for (lang, text_hashstring, bin_hashstring) in fast_commoncrawl_languages_and_hashes
        push!(language_files(FastText_Bin{lang}), "FastText $lang CommonCrawl Binary/cc.$(lang).300.bin")
        push!(language_files(FastText_Text{lang}), "FastText $lang CommonCrawl Text/cc.$(lang).300.vec")

        for (mode, hashstring, ext) in [("Text", text_hashstring, "vec"), ("Binary", bin_hashstring, "bin")]
            register(DataDep("FastText $lang CommonCrawl $mode",
                """
                Dataset: 300 dimentional FastText Word Embeddings, for $lang trained on Wikipedia and the CommonCrawl
                Website: https://fasttext.cc/docs/en/crawl-vectors.html
                Author:  Grave et. al. (Facebook)
                License: CC-SA 3.0
                Citation: E. Grave*, P. Bojanowski*, P. Gupta, A. Joulin, T. Mikolov, Learning Word Vectors for 157 Languages
                """,
                "https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.$(lang).300.$(ext).gz",
                hashstring;
                post_fetch_method=DataDeps.unpack
            ));
        end
    end
    
    for (lang, hashstring) in fast_text_wiki_languages_and_hashes
        # TODO Add Binary files as well
        push!(language_files(FastText_Text{lang}), "FastText $lang Wiki Text/wiki.$(lang).vec")
        register(DataDep("FastText $lang Wiki Text",
            """
            Dataset: 300 dimentional FastText Word Embeddings for $lang, trained on Wikipedia
            Website: https://fasttext.cc/docs/en/pretrained-vectors.html
            Author:  Bojanowski et. al. (Facebook)
            License: CC-SA 3.0
            Citation: P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information
            """,
            "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.$(lang).vec",
            hashstring;
        ));
    end
    
    
    
end


# Lang, TextHash, BinHash
const fast_commoncrawl_languages_and_hashes = [
	(:af, nothing, nothing),
	(:als, nothing, nothing),
	(:am, nothing, nothing),
	(:an, nothing, nothing),
	(:ar, nothing, nothing),
	(:arz, nothing, nothing),
	(:as, nothing, nothing),
	(:ast, nothing, nothing),
	(:az, nothing, nothing),
	(:azb, nothing, nothing),
	(:ba, nothing, nothing),
	(:bar, nothing, nothing),
	(:bcl, nothing, nothing),
	(:be, nothing, nothing),
	(:bg, nothing, nothing),
	(:bh, nothing, nothing),
	(:bn, nothing, nothing),
	(:bo, nothing, nothing),
	(:bpy, nothing, nothing),
	(:br, nothing, nothing),
	(:bs, nothing, nothing),
	(:ca, nothing, nothing),
	(:ce, nothing, nothing),
	(:ceb, nothing, nothing),
	(:ckb, nothing, nothing),
	(:co, nothing, nothing),
	(:cs, nothing, nothing),
	(:cv, nothing, nothing),
	(:cy, nothing, nothing),
	(:da, nothing, nothing),
	(:de, nothing, nothing),
	(:diq, nothing, nothing),
	(:dv, nothing, nothing),
	(:el, nothing, nothing),
	(:eml, nothing, nothing),
	(:eo, nothing, nothing),
	(:es, nothing, nothing),
	(:et, nothing, nothing),
	(:eu, nothing, nothing),
	(:fa, nothing, nothing),
	(:fi, nothing, nothing),
	(:fr, "ab0dca4ef2a8a38d97ca119b491c701c57e17178cfc2f032f3de973e86fe87aa", nothing),
	(:frr, nothing, nothing),
	(:fy, nothing, nothing),
	(:ga, nothing, nothing),
	(:gd, nothing, nothing),
	(:gl, nothing, nothing),
	(:gom, nothing, nothing),
	(:gu, nothing, nothing),
	(:gv, nothing, nothing),
	(:he, nothing, nothing),
	(:hi, nothing, nothing),
	(:hif, nothing, nothing),
	(:hr, nothing, nothing),
	(:hsb, nothing, nothing),
	(:ht, nothing, nothing),
	(:hu, nothing, nothing),
	(:hy, nothing, nothing),
	(:ia, nothing, nothing),
	(:id, nothing, nothing),
	(:ilo, nothing, nothing),
	(:io, nothing, nothing),
	(:is, nothing, nothing),
	(:it, nothing, nothing),
	(:ja, nothing, nothing),
	(:jv, nothing, nothing),
	(:ka, nothing, nothing),
	(:kk, nothing, nothing),
	(:km, nothing, nothing),
	(:kn, nothing, nothing),
	(:ko, nothing, nothing),
	(:ku, nothing, nothing),
	(:ky, nothing, nothing),
	(:la, nothing, nothing),
	(:lb, nothing, nothing),
	(:li, nothing, nothing),
	(:lmo, nothing, nothing),
	(:lt, nothing, nothing),
	(:lv, nothing, nothing),
	(:mai, nothing, nothing),
	(:mg, nothing, nothing),
	(:mhr, nothing, nothing),
	(:min, nothing, nothing),
	(:mk, nothing, nothing),
	(:ml, nothing, nothing),
	(:mn, nothing, nothing),
	(:mr, nothing, nothing),
	(:mrj, nothing, nothing),
	(:ms, nothing, nothing),
	(:mt, nothing, nothing),
	(:mwl, nothing, nothing),
	(:my, nothing, nothing),
	(:myv, nothing, nothing),
	(:mzn, nothing, nothing),
	(:nah, nothing, nothing),
	(:nap, nothing, nothing),
	(:nds, nothing, nothing),
	(:ne, nothing, nothing),
	(:new, nothing, nothing),
	(:nl, nothing, nothing),
	(:nn, nothing, nothing),
	(:no, nothing, nothing),
	(:nso, nothing, nothing),
	(:oc, nothing, nothing),
	(:or, nothing, nothing),
	(:os, nothing, nothing),
	(:pa, nothing, nothing),
	(:pam, nothing, nothing),
	(:pfl, nothing, nothing),
	(:pl, nothing, nothing),
	(:pms, nothing, nothing),
	(:pnb, nothing, nothing),
	(:ps, nothing, nothing),
	(:pt, nothing, nothing),
	(:qu, nothing, nothing),
	(:rm, nothing, nothing),
	(:ro, nothing, nothing),
	(:ru, nothing, nothing),
	(:sa, nothing, nothing),
	(:sah, nothing, nothing),
	(:sc, nothing, nothing),
	(:scn, nothing, nothing),
	(:sco, nothing, nothing),
	(:sd, nothing, nothing),
	(:sh, nothing, nothing),
	(:si, nothing, nothing),
	(:sk, nothing, nothing),
	(:sl, nothing, nothing),
	(:so, nothing, nothing),
	(:sq, nothing, nothing),
	(:sr, nothing, nothing),
	(:su, nothing, nothing),
	(:sv, nothing, nothing),
	(:sw, nothing, nothing),
	(:ta, nothing, nothing),
	(:te, nothing, nothing),
	(:tg, nothing, nothing),
	(:th, nothing, nothing),
	(:tk, nothing, nothing),
	(:tl, nothing, nothing),
	(:tr, nothing, nothing),
	(:tt, nothing, nothing),
	(:ug, nothing, nothing),
	(:uk, nothing, nothing),
	(:ur, nothing, nothing),
	(:uz, nothing, nothing),
	(:vec, nothing, nothing),
	(:vi, nothing, nothing),
	(:vls, nothing, nothing),
	(:vo, nothing, nothing),
	(:wa, nothing, nothing),
	(:war, nothing, nothing),
	(:xmf, nothing, nothing),
	(:yi, nothing, nothing),
	(:yo, nothing, nothing),
	(:zea, nothing, nothing),
	(:zh, nothing, nothing),
]



const fast_text_wiki_languages_and_hashes = [
    (:en, "f4d87723baad28804f89c0ecf74fd0f52ac2ae194c270cb4c89b0a84f0bcf53b"),
    # PR Welcome to add Hashes for the ones below
    (:ab, nothing),
    (:ace, nothing),
    (:ady, nothing),
    (:aa, nothing),
    (:af, nothing),
    (:ak, nothing),
    (:sq, nothing),
    (:als, nothing),
    (:am, nothing),
    (:ang, nothing),
    (:ar, nothing),
    (:an, nothing),
    (:arc, nothing),
    (:hy, nothing),
    (:roa_rup, nothing),
    (:as, nothing),
    (:ast, nothing),
    (:av, nothing),
    (:ay, nothing),
    (:az, nothing),
    (:bm, nothing),
    (:bjn, nothing),
    (:map_bms, nothing),
    (:ba, nothing),
    (:eu, nothing),
    (:bar, nothing),
    (:be, nothing),
    (:bn, nothing),
    (:bh, nothing),
    (:bpy, nothing),
    (:bi, nothing),
    (:bs, nothing),
    (:br, nothing),
    (:bug, nothing),
    (:bg, nothing),
    (:my, nothing),
    (:bxr, nothing),
    (:zh_yue, nothing),
    (:ca, nothing),
    (:ceb, nothing),
    (:bcl, nothing),
    (:ch, nothing),
    (:cbk_zam, nothing),
    (:ce, nothing),
    (:chr, nothing),
    (:chy, nothing),
    (:ny, nothing),
    (:zh, nothing),
    (:cho, nothing),
    (:cv, nothing),
    (:zh_classical, nothing),
    (:kw, nothing),
    (:co, nothing),
    (:cr, nothing),
    (:crh, nothing),
    (:hr, nothing),
    (:cs, nothing),
    (:da, nothing),
    (:dv, nothing),
    (:nl, nothing),
    (:nds_nl, nothing),
    (:dz, nothing),
    (:pa, nothing),
    (:arz, nothing),
    (:eml, nothing),
    (:myv, nothing),
    (:eo, nothing),
    (:et, nothing),
    (:ee, nothing),
    (:ext, nothing),
    (:fo, nothing),
    (:hif, nothing),
    (:fj, nothing),
    (:fi, nothing),
    (:frp, nothing),
    (:fr, nothing),
    (:fur, nothing),
    (:ff, nothing),
    (:gag, nothing),
    (:gl, nothing),
    (:gan, nothing),
    (:ka, nothing),
    (:de, nothing),
    (:glk, nothing),
    (:gom, nothing),
    (:got, nothing),
    (:el, nothing),
    (:kl, nothing),
    (:gn, nothing),
    (:gu, nothing),
    (:ht, nothing),
    (:hak, nothing),
    (:ha, nothing),
    (:haw, nothing),
    (:he, nothing),
    (:hz, nothing),
    (:mrj, nothing),
    (:hi, nothing),
    (:ho, nothing),
    (:hu, nothing),
    (:is, nothing),
    (:io, nothing),
    (:ig, nothing),
    (:ilo, nothing),
    (:id, nothing),
    (:ia, nothing),
    (:ie, nothing),
    (:iu, nothing),
    (:ik, nothing),
    (:ga, nothing),
    (:it, nothing),
    (:jam, nothing),
    (:ja, nothing),
    (:jv, nothing),
    (:kbd, nothing),
    (:kab, nothing),
    (:xal, nothing),
    (:kn, nothing),
    (:kr, nothing),
    (:pam, nothing),
    (:krc, nothing),
    (:kaa, nothing),
    (:ks, nothing),
    (:csb, nothing),
    (:kk, nothing),
    (:km, nothing),
    (:ki, nothing),
    (:rw, nothing),
    (:ky, nothing),
    (:rn, nothing),
    (:kv, nothing),
    (:koi, nothing),
    (:kg, nothing),
    (:ko, nothing),
    (:kj, nothing),
    (:ku, nothing),
    (:ckb, nothing),
    (:lad, nothing),
    (:lbe, nothing),
    (:lo, nothing),
    (:ltg, nothing),
    (:la, nothing),
    (:lv, nothing),
    (:lez, nothing),
    (:lij, nothing),
    (:li, nothing),
    (:ln, nothing),
    (:lt, nothing),
    (:olo, nothing),
    (:jbo, nothing),
    (:lmo, nothing),
    (:nds, nothing),
    (:dsb, nothing),
    (:lg, nothing),
    (:lb, nothing),
    (:mk, nothing),
    (:mai, nothing),
    (:mg, nothing),
    (:ms, nothing),
    (:ml, nothing),
    (:mt, nothing),
    (:gv, nothing),
    (:mi, nothing),
    (:mr, nothing),
    (:mh, nothing),
    (:mzn, nothing),
    (:mhr, nothing),
    (:cdo, nothing),
    (:zh_min_nan, nothing),
    (:min, nothing),
    (:xmf, nothing),
    (:mwl, nothing),
    (:mdf, nothing),
    (:mo, nothing),
    (:mn, nothing),
    (:mus, nothing),
    (:nah, nothing),
    (:na, nothing),
    (:nv, nothing),
    (:ng, nothing),
    (:nap, nothing),
    (:ne, nothing),
    (:new, nothing),
    (:pih, nothing),
    (:nrm, nothing),
    (:frr, nothing),
    (:lrc, nothing),
    (:se, nothing),
    (:nso, nothing),
    (:no, nothing),
    (:nn, nothing),
    (:nov, nothing),
    (:ii, nothing),
    (:oc, nothing),
    (:cu, nothing),
    (:or, nothing),
    (:om, nothing),
    (:os, nothing),
    (:pfl, nothing),
    (:pi, nothing),
    (:pag, nothing),
    (:pap, nothing),
    (:ps, nothing),
    (:pdc, nothing),
    (:fa, nothing),
    (:pcd, nothing),
    (:pms, nothing),
    (:pl, nothing),
    (:pnt, nothing),
    (:pt, nothing),
    (:qu, nothing),
    (:ksh, nothing),
    (:rmy, nothing),
    (:ro, nothing),
    (:rm, nothing),
    (:ru, nothing),
    (:rue, nothing),
    (:sah, nothing),
    (:sm, nothing),
    (:bat_smg, nothing),
    (:sg, nothing),
    (:sa, nothing),
    (:sc, nothing),
    (:stq, nothing),
    (:sco, nothing),
    (:gd, nothing),
    (:sr, nothing),
    (:sh, nothing),
    (:st, nothing),
    (:sn, nothing),
    (:scn, nothing),
    (:szl, nothing),
    (:simple, nothing),
    (:sd, nothing),
    (:si, nothing),
    (:sk, nothing),
    (:sl, nothing),
    (:so, nothing),
    (:azb, nothing),
    (:es, nothing),
    (:srn, nothing),
    (:su, nothing),
    (:sw, nothing),
    (:ss, nothing),
    (:sv, nothing),
    (:tl, nothing),
    (:ty, nothing),
    (:tg, nothing),
    (:ta, nothing),
    (:roa_tara, nothing),
    (:tt, nothing),
    (:te, nothing),
    (:tet, nothing),
    (:th, nothing),
    (:bo, nothing),
    (:ti, nothing),
    (:tpi, nothing),
    (:to, nothing),
    (:ts, nothing),
    (:tn, nothing),
    (:tcy, nothing),
    (:tum, nothing),
    (:tr, nothing),
    (:tk, nothing),
    (:tyv, nothing),
    (:tw, nothing),
    (:udm, nothing),
    (:uk, nothing),
    (:hsb, nothing),
    (:ur, nothing),
    (:ug, nothing),
    (:uz, nothing),
    (:ve, nothing),
    (:vec, nothing),
    (:vep, nothing),
    (:vi, nothing),
    (:vo, nothing),
    (:fiu_vro, nothing),
    (:wa, nothing),
    (:war, nothing),
    (:cy, nothing),
    (:vls, nothing),
    (:fy, nothing),
    (:pnb, nothing),
    (:wo, nothing),
    (:wuu, nothing),
    (:xh, nothing),
    (:yi, nothing),
    (:yo, nothing),
    (:diq, nothing),
    (:zea, nothing),
    (:za, nothing),
    (:zu, nothing),
]
