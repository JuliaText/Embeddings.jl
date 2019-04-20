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
    
    for (lang, text_hashstring, bin_hashstring) in fast_wiki_languages_and_hashes
        push!(language_files(FastText_Text{lang}), "FastText $lang Wiki Text/wiki.$(lang).vec")
        push!(language_files(FastText_Bin{lang}), "FastText $lang Wiki Binary/wiki.$(lang).bin")
        for (mode, hashstring, ext) in [
                ("Text", text_hashstring, "vec"),
                ("Binary", bin_hashstring, "zip")
            ]
            
            register(DataDep("FastText $lang Wiki $mode",
                """
                Dataset: 300 dimentional FastText Word Embeddings for $lang, trained on Wikipedia
                Website: https://fasttext.cc/docs/en/pretrained-vectors.html
                Author:  Bojanowski et. al. (Facebook)
                License: CC-SA 3.0
                Citation: P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information
                """,
                "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.$(lang).$(ext)",
                hashstring;
                post_fetch_method = mode == "Binary" ? DataDeps.unpack : identity
            ));
        end
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



const fast_wiki_languages_and_hashes = [
    # PR Welcome to add Hashes for the ones below
	(:en, "f4d87723baad28804f89c0ecf74fd0f52ac2ae194c270cb4c89b0a84f0bcf53b", nothing),
    (:ab, nothing, nothing),
    (:ace, nothing, nothing),
    (:ady, nothing, nothing),
    (:aa, nothing, nothing),
    (:af, nothing, nothing),
    (:ak, nothing, nothing),
    (:sq, nothing, nothing),
    (:als, nothing, nothing),
    (:am, nothing, nothing),
    (:ang, nothing, nothing),
    (:ar, nothing, nothing),
    (:an, nothing, nothing),
    (:arc, nothing, nothing),
    (:hy, nothing, nothing),
    (:roa_rup, nothing, nothing),
    (:as, nothing, nothing),
    (:ast, nothing, nothing),
    (:av, nothing, nothing),
    (:ay, nothing, nothing),
    (:az, nothing, nothing),
    (:bm, nothing, nothing),
    (:bjn, nothing, nothing),
    (:map_bms, nothing, nothing),
    (:ba, nothing, nothing),
    (:eu, nothing, nothing),
    (:bar, nothing, nothing),
    (:be, nothing, nothing),
    (:bn, nothing, nothing),
    (:bh, nothing, nothing),
    (:bpy, nothing, nothing),
    (:bi, nothing, nothing),
    (:bs, nothing, nothing),
    (:br, nothing, nothing),
    (:bug, nothing, nothing),
    (:bg, nothing, nothing),
    (:my, nothing, nothing),
    (:bxr, nothing, nothing),
    (:zh_yue, nothing, nothing),
    (:ca, nothing, nothing),
    (:ceb, nothing, nothing),
    (:bcl, nothing, nothing),
    (:ch, nothing, nothing),
    (:cbk_zam, nothing, nothing),
    (:ce, nothing, nothing),
    (:chr, nothing, nothing),
    (:chy, nothing, nothing),
    (:ny, nothing, nothing),
    (:zh, nothing, nothing),
    (:cho, nothing, nothing),
    (:cv, nothing, nothing),
    (:zh_classical, nothing, nothing),
    (:kw, nothing, nothing),
    (:co, nothing, nothing),
    (:cr, nothing, nothing),
    (:crh, nothing, nothing),
    (:hr, nothing, nothing),
    (:cs, nothing, nothing),
    (:da, nothing, nothing),
    (:dv, nothing, nothing),
    (:nl, nothing, nothing),
    (:nds_nl, nothing, nothing),
    (:dz, nothing, nothing),
    (:pa, nothing, nothing),
    (:arz, nothing, nothing),
    (:eml, nothing, nothing),
    (:myv, nothing, nothing),
    (:eo, nothing, nothing),
    (:et, nothing, nothing),
    (:ee, nothing, nothing),
    (:ext, nothing, nothing),
    (:fo, nothing, nothing),
    (:hif, nothing, nothing),
    (:fj, nothing, nothing),
    (:fi, nothing, nothing),
    (:frp, nothing, nothing),
    (:fr, nothing, nothing),
    (:fur, nothing, nothing),
    (:ff, nothing, nothing),
    (:gag, nothing, nothing),
    (:gl, nothing, nothing),
    (:gan, nothing, nothing),
    (:ka, nothing, nothing),
    (:de, nothing, nothing),
    (:glk, nothing, nothing),
    (:gom, nothing, nothing),
    (:got, nothing, nothing),
    (:el, nothing, nothing),
    (:kl, nothing, nothing),
    (:gn, nothing, nothing),
    (:gu, nothing, nothing),
    (:ht, nothing, nothing),
    (:hak, nothing, nothing),
    (:ha, nothing, nothing),
    (:haw, nothing, nothing),
    (:he, nothing, nothing),
    (:hz, nothing, nothing),
    (:mrj, nothing, nothing),
    (:hi, nothing, nothing),
    (:ho, nothing, nothing),
    (:hu, nothing, nothing),
    (:is, nothing, nothing),
    (:io, nothing, nothing),
    (:ig, nothing, nothing),
    (:ilo, nothing, nothing),
    (:id, nothing, nothing),
    (:ia, nothing, nothing),
    (:ie, nothing, nothing),
    (:iu, nothing, nothing),
    (:ik, nothing, nothing),
    (:ga, nothing, nothing),
    (:it, nothing, nothing),
    (:jam, nothing, nothing),
    (:ja, nothing, nothing),
    (:jv, nothing, nothing),
    (:kbd, nothing, nothing),
    (:kab, nothing, nothing),
    (:xal, nothing, nothing),
    (:kn, nothing, nothing),
    (:kr, nothing, nothing),
    (:pam, nothing, nothing),
    (:krc, nothing, nothing),
    (:kaa, nothing, nothing),
    (:ks, nothing, nothing),
    (:csb, nothing, nothing),
    (:kk, nothing, nothing),
    (:km, nothing, nothing),
    (:ki, nothing, nothing),
    (:rw, nothing, nothing),
    (:ky, nothing, nothing),
    (:rn, nothing, nothing),
    (:kv, nothing, nothing),
    (:koi, nothing, nothing),
    (:kg, nothing, nothing),
    (:ko, nothing, nothing),
    (:kj, nothing, nothing),
    (:ku, nothing, nothing),
    (:ckb, nothing, nothing),
    (:lad, nothing, nothing),
    (:lbe, nothing, nothing),
    (:lo, nothing, nothing),
    (:ltg, nothing, nothing),
    (:la, nothing, nothing),
    (:lv, nothing, nothing),
    (:lez, nothing, nothing),
    (:lij, nothing, nothing),
    (:li, nothing, nothing),
    (:ln, nothing, nothing),
    (:lt, nothing, nothing),
    (:olo, nothing, nothing),
    (:jbo, nothing, nothing),
    (:lmo, nothing, nothing),
    (:nds, nothing, nothing),
    (:dsb, nothing, nothing),
    (:lg, nothing, nothing),
    (:lb, nothing, nothing),
    (:mk, nothing, nothing),
    (:mai, nothing, nothing),
    (:mg, nothing, nothing),
    (:ms, nothing, nothing),
    (:ml, nothing, nothing),
    (:mt, nothing, nothing),
    (:gv, nothing, nothing),
    (:mi, nothing, nothing),
    (:mr, nothing, nothing),
    (:mh, nothing, nothing),
    (:mzn, nothing, nothing),
    (:mhr, nothing, nothing),
    (:cdo, nothing, nothing),
    (:zh_min_nan, nothing, nothing),
    (:min, nothing, nothing),
    (:xmf, nothing, nothing),
    (:mwl, nothing, nothing),
    (:mdf, nothing, nothing),
    (:mo, nothing, nothing),
    (:mn, nothing, nothing),
    (:mus, nothing, nothing),
    (:nah, nothing, nothing),
    (:na, nothing, nothing),
    (:nv, nothing, nothing),
    (:ng, nothing, nothing),
    (:nap, nothing, nothing),
    (:ne, nothing, nothing),
    (:new, nothing, nothing),
    (:pih, nothing, nothing),
    (:nrm, nothing, nothing),
    (:frr, nothing, nothing),
    (:lrc, nothing, nothing),
    (:se, nothing, nothing),
    (:nso, nothing, nothing),
    (:no, nothing, nothing),
    (:nn, nothing, nothing),
    (:nov, nothing, nothing),
    (:ii, nothing, nothing),
    (:oc, nothing, nothing),
    (:cu, nothing, nothing),
    (:or, nothing, nothing),
    (:om, nothing, nothing),
    (:os, nothing, nothing),
    (:pfl, nothing, nothing),
    (:pi, nothing, nothing),
    (:pag, nothing, nothing),
    (:pap, nothing, nothing),
    (:ps, nothing, nothing),
    (:pdc, nothing, nothing),
    (:fa, nothing, nothing),
    (:pcd, nothing, nothing),
    (:pms, nothing, nothing),
    (:pl, nothing, nothing),
    (:pnt, nothing, nothing),
    (:pt, nothing, nothing),
    (:qu, nothing, nothing),
    (:ksh, nothing, nothing),
    (:rmy, nothing, nothing),
    (:ro, nothing, nothing),
    (:rm, nothing, nothing),
    (:ru, nothing, nothing),
    (:rue, nothing, nothing),
    (:sah, nothing, nothing),
    (:sm, nothing, nothing),
    (:bat_smg, nothing, nothing),
    (:sg, nothing, nothing),
    (:sa, nothing, nothing),
    (:sc, nothing, nothing),
    (:stq, nothing, nothing),
    (:sco, nothing, nothing),
    (:gd, nothing, nothing),
    (:sr, nothing, nothing),
    (:sh, nothing, nothing),
    (:st, nothing, nothing),
    (:sn, nothing, nothing),
    (:scn, nothing, nothing),
    (:szl, nothing, nothing),
    (:simple, nothing, nothing),
    (:sd, nothing, nothing),
    (:si, nothing, nothing),
    (:sk, nothing, nothing),
    (:sl, nothing, nothing),
    (:so, nothing, nothing),
    (:azb, nothing, nothing),
    (:es, nothing, nothing),
    (:srn, nothing, nothing),
    (:su, nothing, nothing),
    (:sw, nothing, nothing),
    (:ss, nothing, nothing),
    (:sv, nothing, nothing),
    (:tl, nothing, nothing),
    (:ty, nothing, nothing),
    (:tg, nothing, nothing),
    (:ta, nothing, nothing),
    (:roa_tara, nothing, nothing),
    (:tt, nothing, nothing),
    (:te, nothing, nothing),
    (:tet, nothing, nothing),
    (:th, nothing, nothing),
    (:bo, nothing, nothing),
    (:ti, nothing, nothing),
    (:tpi, nothing, nothing),
    (:to, nothing, nothing),
    (:ts, nothing, nothing),
    (:tn, nothing, nothing),
    (:tcy, nothing, nothing),
    (:tum, nothing, nothing),
    (:tr, nothing, nothing),
    (:tk, nothing, nothing),
    (:tyv, nothing, nothing),
    (:tw, nothing, nothing),
    (:udm, nothing, nothing),
    (:uk, nothing, nothing),
    (:hsb, nothing, nothing),
    (:ur, nothing, nothing),
    (:ug, nothing, nothing),
    (:uz, nothing, nothing),
    (:ve, nothing, nothing),
    (:vec, nothing, nothing),
    (:vep, nothing, nothing),
    (:vi, nothing, nothing),
    (:vo, nothing, nothing),
    (:fiu_vro, nothing, nothing),
    (:wa, nothing, nothing),
    (:war, nothing, nothing),
    (:cy, nothing, nothing),
    (:vls, nothing, nothing),
    (:fy, nothing, nothing),
    (:pnb, nothing, nothing),
    (:wo, nothing, nothing),
    (:wuu, nothing, nothing),
    (:xh, nothing, nothing),
    (:yi, nothing, nothing),
    (:yo, nothing, nothing),
    (:diq, nothing, nothing),
    (:zea, nothing, nothing),
    (:za, nothing, nothing),
    (:zu, nothing, nothing),
]
