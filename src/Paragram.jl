abstract type Paragram{LANG} <: EmbeddingSystem{LANG} end
using Dates
using Random: randstring
using HTTP 
"""
    unshortlink(url)
return unshorten url or the url if it is not a short link
"""
function unshortlink(url; kw...)
    rq = HTTP.request("HEAD", url; redirect=false, status_exception=false, kw...)
    while rq.status ÷ 100 == 3
        url = HTTP.header(rq, "Location")
        rq = HTTP.request("HEAD", url; redirect=false, status_exception=false, kw...)
    end
    url
end

isgooglesheet(url) = occursin("docs.google.com/spreadsheets", url)
isgoogledrive(url) = occursin("drive.google.com", url)

function googlesheet_handler(url; format=:csv)
    link, expo = splitdir(url)
    if startswith(expo, "edit") || expo == ""
        url = link * "/export?format=$format"
    elseif startswith(expo, "export")
        url = replace(url, r"format=([a-zA-Z]*)(.*)"=>SubstitutionString("format=$format\\2"))
    end
    url
end

function maybegoogle_download(url, localdir)
    long_url = unshortlink(url)
    if isgooglesheet(long_url)
        long_url = googlesheet_handler(long_url)
    end

    if isgoogledrive(long_url)
        download_gdrive(long_url, localdir)
    else
        DataDeps.fetch_http(long_url, localdir)
    end
end

function find_gcode(ckj)
    for cookie ∈ ckj
        if match(r"_warning_", cookie.name) !== nothing
            return cookie.value
        end
    end

    nothing
end

function download_gdrive(url, localdir)
    ckjar = copy(HTTP.CookieRequest.default_cookiejar)
    rq = HTTP.request("HEAD", url; cookies=true, cookiejar=ckjar)
    ckj = ckjar["drive.google.com"]
    gcode = find_gcode(ckj)
    @assert gcode !== nothing

    format_progress(x) = round(x, digits=4)
    format_bytes(x) = !isfinite(x) ? "∞ B" : Base.format_bytes(x)
    format_seconds(x) = "$(round(x; digits=2)) s"
    format_bytes_per_second(x) = format_bytes(x) * "/s"

    local filepath
    newurl = unshortlink("$url&confirm=$gcode"; cookies=true, cookiejar=ckjar)


    HTTP.open("GET", newurl, ["Range"=>"bytes=0-"]; cookies=true, cookiejar=ckjar) do stream
        resp = HTTP.startread(stream)
        hcd = HTTP.header(resp, "Content-Disposition")
        m = match(r"filename=\\\"(.*)\\\"", hcd)
        if m === nothing
            filename = "gdrive_downloaded-$(randstring())"
        else
            filename = m.captures[]
        end

        filepath = joinpath(localdir, filename)

        total_bytes = tryparse(Float64, split(HTTP.header(resp, "Content-Range"), '/')[end])
        total_bytes === nothing && (total_bytes = NaN)
        downloaded_bytes = 0
        start_time = now()
        prev_time = now()
        period = DataDeps.progress_update_period()

        function report_callback()
            prev_time = now()
            taken_time = (prev_time - start_time).value / 1000 # in seconds
            average_speed = downloaded_bytes / taken_time
            remaining_bytes = total_bytes - downloaded_bytes
            remaining_time = remaining_bytes / average_speed
            completion_progress = downloaded_bytes / total_bytes

            @info("Downloading",
                  source=url,
                  dest = filepath,
                  progress = completion_progress |> format_progress,
                  time_taken = taken_time |> format_seconds,
                  time_remaining = remaining_time |> format_seconds,
                  average_speed = average_speed |> format_bytes_per_second,
                  downloaded = downloaded_bytes |> format_bytes,
                  remaining = remaining_bytes |> format_bytes,
                  total = total_bytes |> format_bytes,
                  )
        end


        Base.open(filepath, "w") do fh
            while(!eof(stream))
                downloaded_bytes += write(fh, readavailable(stream))
                if !isinf(period)
                  if now() - prev_time > Millisecond(1000*period)
                    report_callback()
                  end
                end
            end
        end
        report_callback()
    end
    filepath
end
#const Paragram_max_size = 4_000_000

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
                         fetch_method = download_gdrive))
       
            append!(language_files(Paragram{:en}), ["$(depname)/$(depname).txt"])
       
            
      
    end
end

function _load_embeddings(::Type{<:Paragram}, embedding_file, max_vocab_size, keep_words)
    local LL, indexed_words, index
    if length(keep_words) > 0
        max_vocab_size = length(keep_words)
    end
    indexed_words = Vector{String}()
    LL = Vector{Vector{Float32}}()
    open(embedding_file) do f
        index = 1
        for line in eachline(f)
            xs = split(line, ' ')
            word = xs[1]
            if length(keep_words) == 0 || (word in keep_words)
                index > max_vocab_size && break
                push!(indexed_words, word)
                try
                    push!(LL, parse.(Float32, @view(xs[2:end])))
                catch err
                    err isa ArgumentError || rethrow()
                    @warn "Could not parse word vector" index word exception=err
                end
                index += 1
            end
        end
    end
    return reduce(hcat, LL), indexed_words
end
