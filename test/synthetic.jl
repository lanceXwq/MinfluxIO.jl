# Writers for small synthetic OBF/MSR files with embedded MINFLUX data blocks, mimicking what Imspector produces, so the readers can be tested without real data.

using CodecZlib: ZlibCompressor
using Blosc: Blosc

le(x) = Vector{UInt8}(reinterpret(UInt8, [htol(x)]))
lestr(s::AbstractString) = vcat(le(UInt32(ncodeunits(s))), Vector{UInt8}(codeunits(s)))

const DT_CODES = Dict(UInt8 => 0x01, Int8 => 0x02, UInt16 => 0x04, Int16 => 0x08, UInt32 => 0x10,
    Int32 => 0x20, Float32 => 0x40, Float64 => 0x80, UInt64 => 0x1000, Int64 => 0x2000, Bool => 0x10000)

function siunit_bytes(meters::Integer)
    out = UInt8[]
    for i in 1:9
        append!(out, le(Int32(i == 1 ? meters : 0)), le(Int32(1)))
    end
    append!(out, le(1.0))
end

function kv_bytes(d::AbstractDict)
    out = UInt8[]
    for (k, v) in d
        append!(out, lestr(k), lestr(v))
    end
    append!(out, le(UInt32(0)))
end

# One OBF stack (format version 7-like footer, including 60 bytes of unknown trailing footer fields as seen in real files). The `next` pointer is at byte 360.
function stack_blob(name, data::AbstractArray{T}; compress = true, labels = nothing,
        pixel = 1e-7, tags = Dict{String,String}(), raw_bytes = nothing) where {T}
    rank = ndims(data)
    raw = raw_bytes === nothing ? Vector{UInt8}(reinterpret(UInt8, vec(collect(data)))) : raw_bytes
    payload = compress ? transcode(ZlibCompressor, raw) : raw
    out = UInt8[]
    append!(out, b"OMAS_BF_STACK\n\xff\xff", le(UInt32(7)), le(UInt32(rank)))
    for i in 1:15
        append!(out, le(UInt32(i <= rank ? size(data, i) : 0)))
    end
    for i in 1:15
        append!(out, le(i <= rank ? pixel * size(data, i) : 0.0))
    end
    for i in 1:15
        append!(out, le(0.0))
    end
    append!(out, le(UInt32(DT_CODES[T])), le(UInt32(compress ? 1 : 0)), le(UInt32(compress ? 6 : 0)),
        le(UInt32(ncodeunits(name))), le(UInt32(0)), le(UInt64(0)), le(UInt64(length(payload))), le(UInt64(0)))
    append!(out, codeunits(name), payload)

    tagbytes = kv_bytes(tags)
    footer = UInt8[]
    append!(footer, zeros(UInt8, 60), zeros(UInt8, 60), le(UInt32(0)))   # col flags, metadata_length
    append!(footer, siunit_bytes(0))
    for i in 1:15
        append!(footer, siunit_bytes(i <= min(rank, 2) ? 1 : 0))
    end
    append!(footer, le(UInt64(0)), le(UInt64(0)), le(UInt64(length(tagbytes))), le(UInt64(0)),
        le(UInt32(1)), le(UInt64(0)), le(UInt64(length(data))), le(UInt64(0)))
    append!(footer, zeros(UInt8, 60))
    append!(out, le(UInt32(length(footer) + 4)), footer)
    labs = labels === nothing ? ["Ax$i" for i in 1:rank] : labels
    for l in labs
        append!(out, lestr(l))
    end
    append!(out, tagbytes)
    return out
end

function obf_bytes(blobs::Vector{Vector{UInt8}}; metadata = Dict("ome_xml" => "<OME/>"))
    header_len = 10 + 4 + 8 + 4 + 8
    positions = cumsum([header_len; length.(blobs)])[1:end-1]
    meta_pos = header_len + sum(length, blobs)
    out = UInt8[]
    append!(out, b"OMAS_BF\n\xff\xff", le(UInt32(2)), le(UInt64(positions[1])), le(UInt32(0)), le(UInt64(meta_pos)))
    for (i, b) in enumerate(blobs)
        b = copy(b)
        next = i < length(blobs) ? positions[i+1] : 0
        b[361:368] = le(UInt64(next))
        append!(out, b)
    end
    append!(out, kv_bytes(metadata))
    append!(out, b"trailing msr-specific data")
    return out
end

# A serialized MFXDTA block. `entries` is a vector of path => bytes (or `nothing` for a directory).
function mfxdta_bytes(entries)
    out = UInt8[]
    append!(out, b"MFXDTA\0", le(UInt32(3)), le(UInt32(1_794_585_388)))
    append!(out, zeros(UInt8, 147 - length(out)))
    for (path, data) in entries
        append!(out, le(UInt64(ncodeunits(path))), codeunits(path))
        if data === nothing
            push!(out, 0x01)
        else
            append!(out, [0x00], le(UInt64(length(data))), data)
        end
    end
    append!(out, zeros(UInt8, 100))
end

# Record bytes for a small MINFLUX table with the real Imspector dtype subset below.
const MFX_DTYPE_JSON = """[["vld","|b1"],["fnl","|b1"],["tim","<f8"],["tid","<u4"],["thi","|u1"],["itr","<i4"],["loc","<f8",[3]],["cfr","<f2"],["dcr","<f2",[1]]]"""

function mfx_records(n)
    vld = isodd.(1:n)
    fnl = (1:n) .% 3 .== 0
    tim = collect(0.5 .* (1:n))
    tid = UInt32.(div.(1:n, 4))
    thi = UInt8.((1:n) .% 2)
    itr = Int32.((1:n) .% 5)
    loc = [Float64(1e-9 * (100i + j)) for i in 1:n, j in 1:3]
    cfr = Float16.(0.25 .* ((1:n) .% 4))
    dcr = Float16.(0.5 .* ((1:n) .% 2))
    buf = UInt8[]
    for i in 1:n
        append!(buf, [UInt8(vld[i]), UInt8(fnl[i])], le(tim[i]), le(tid[i]), [thi[i]], le(itr[i]))
        for j in 1:3
            append!(buf, le(loc[i, j]))
        end
        append!(buf, le(cfr[i]), le(dcr[i]))
    end
    return buf, (; vld, fnl, tim, tid, thi, itr, loc, cfr, dcr)
end

const MFX_ITEMSIZE = 2 + 8 + 4 + 1 + 4 + 24 + 2 + 2

const MFX_DTYPE_PY = "[('vld', '|b1'), ('fnl', '|b1'), ('tim', '<f8'), ('tid', '<u4'), ('thi', '|u1'), " *
                     "('itr', '<i4'), ('loc', '<f8', (3,)), ('cfr', '<f2'), ('dcr', '<f2', (1,))]"

# A .npy file (format version 1.0 or 2.0) holding `n` packed records.
function npy_bytes(descr::AbstractString, n::Integer, data::Vector{UInt8}; version = 1, shape = "($n,)")
    header = "{'descr': $descr, 'fortran_order': False, 'shape': $shape, }"
    prefix = 8 + (version == 1 ? 2 : 4)
    pad = mod(-(prefix + ncodeunits(header) + 1), 64)
    header = header * " "^pad * "\n"
    hlen = version == 1 ? le(UInt16(ncodeunits(header))) : le(UInt32(ncodeunits(header)))
    return vcat(b"\x93NUMPY", [UInt8(version), 0x00], hlen, Vector{UInt8}(codeunits(header)), data)
end

function zarray_json(dtype_json, n, chunk, compressor_json, itemsize)
    fill = repr(String(Base64.base64encode(zeros(UInt8, itemsize))))
    """{"chunks":[$chunk],"compressor":$compressor_json,"dtype":$dtype_json,"fill_value":$fill,"filters":null,"order":"C","shape":[$n],"zarr_format":2}"""
end

# MINFLUX store with an `mfx` table (zlib chunks, one chunk left out on purpose when `drop_chunk`), a bead grid with blosc-compressed points, and attributes.
function minflux_store(label; n = 25, chunk = 10, drop_chunk = false)
    buf, cols = mfx_records(n)
    entries = Any["zarr" => nothing,
        "zarr/.zattrs" => Vector{UInt8}("""{"rois":[],"version":"2.1"}"""),
        "zarr/.zgroup" => Vector{UInt8}("""{"zarr_format":2}"""),
        "zarr/mfx" => nothing,
        "zarr/mfx/.zarray" => Vector{UInt8}(zarray_json(MFX_DTYPE_JSON, n, chunk, """{"id":"zlib","level":1}""", MFX_ITEMSIZE)),
        "zarr/mfx/.zattrs" => Vector{UInt8}("""{"label":"$label","acquisition_date":"2026-01-01T00:00:00+01:00","measurement":{"grids":{"mbm_0":{"type":"mbm"}}}}""")]
    for ci in 0:cld(n, chunk)-1
        drop_chunk && ci == 1 && continue
        lo = ci * chunk
        k = min(chunk, n - lo)
        push!(entries, "zarr/mfx/$ci" => transcode(ZlibCompressor, buf[lo*MFX_ITEMSIZE+1:(lo+k)*MFX_ITEMSIZE]))
    end
    # bead grid: gri u4, xyz f8[3], tim f4, str f4 (blosc, as in real files)
    nb = 4
    pbuf = UInt8[]
    for i in 1:nb
        append!(pbuf, le(UInt32(60 + i)), le(1e-6i), le(2e-6i), le(0.0), le(Float32(10i)), le(Float32(i)))
    end
    pdtype = """[["gri","<u4"],["xyz","<f8",[3]],["tim","<f4"],["str","<f4"]]"""
    append!(entries, Any["zarr/grd" => nothing, "zarr/grd/mbm_0" => nothing,
        "zarr/grd/mbm_0/.zattrs" => Vector{UInt8}("""{"controlling_thi":2}"""),
        "zarr/grd/mbm_0/points" => nothing,
        "zarr/grd/mbm_0/points/.zarray" => Vector{UInt8}(zarray_json(pdtype, nb, 10000,
            """{"blocksize":0,"clevel":5,"cname":"lz4","id":"blosc","shuffle":1}""", 36)),
        "zarr/grd/mbm_0/points/0" => Blosc.compress(pbuf; level = 5, shuffle = true, itemsize = 4),
        "zarr/mbm" => nothing, "zarr/mbm/mbm_0" => nothing,
        "zarr/mbm/mbm_0/.zattrs" => Vector{UInt8}("""{"used":[],"version":"2.2"}""")])
    return mfxdta_bytes(entries), cols
end
