# NumPy dtype handling (shared with the .npy reader) and a minimal reader for the Zarr v2 arrays inside a MINFLUX data block.
#
# Supports what Imspector writes: 1-D arrays of plain or structured (record) NumPy dtypes, little-endian, compressed with zlib, gzip or blosc (or not at all), without filters. A general Zarr implementation is not needed for these files.

struct ZarrField
    name::Symbol
    T::DataType
    shape::Tuple{Vararg{Int}}
    offset::Int
    nbytes::Int
end

function _numpy_scalar_type(s::AbstractString)
    length(s) >= 3 || error("unsupported NumPy dtype \"$s\"")
    endian, kind, n = s[1], s[2], parse(Int, s[3:end])
    endian == '>' && n > 1 && error("big-endian NumPy dtype \"$s\" is not supported")
    T = if kind == 'b' && n == 1
        Bool
    elseif kind == 'u'
        get(Dict(1 => UInt8, 2 => UInt16, 4 => UInt32, 8 => UInt64), n, nothing)
    elseif kind == 'i'
        get(Dict(1 => Int8, 2 => Int16, 4 => Int32, 8 => Int64), n, nothing)
    elseif kind == 'f'
        get(Dict(2 => Float16, 4 => Float32, 8 => Float64), n, nothing)
    elseif kind == 'c'
        get(Dict(8 => ComplexF32, 16 => ComplexF64), n, nothing)
    else
        nothing
    end
    T === nothing && error("unsupported NumPy dtype \"$s\"")
    return T
end

# Returns (fields, itemsize); `fields === nothing` for a plain (non-record) dtype.
function _parse_dtype(dt)
    if dt isa AbstractString
        T = _numpy_scalar_type(dt)
        return nothing, sizeof(T), T
    end
    fields = ZarrField[]
    offset = 0
    for f in dt
        f[1] isa AbstractString || error("structured dtype fields with titles are not supported")
        f[2] isa AbstractString ||
            error("nested structured dtypes (field \"$(f[1])\") are not supported; " *
                  "older Imspector MINFLUX export layouts used them")
        shape = length(f) >= 3 ? Tuple(Int.(f[3])) : ()
        if length(f[2]) >= 3 && f[2][2] == 'V'    # padding / opaque bytes: skip
            offset += parse(Int, f[2][3:end]) * prod(shape)
            continue
        end
        T = _numpy_scalar_type(f[2])
        nb = sizeof(T) * prod(shape)
        push!(fields, ZarrField(Symbol(f[1]), T, shape, offset, nb))
        offset += nb
    end
    return fields, offset, nothing
end

function _decode_chunk(raw::Vector{UInt8}, compressor)
    compressor === nothing && return raw
    id = compressor["id"]
    id == "zlib" && return transcode(ZlibDecompressor, raw)
    id == "gzip" && return transcode(GzipDecompressor, raw)
    id == "blosc" && return Blosc.decompress(UInt8, raw)
    error("unsupported Zarr compressor \"$id\"")
end

function _fill_bytes(fill, itemsize::Int, T)
    fill === nothing && return zeros(UInt8, itemsize)
    if T === nothing                          # structured: base64 of one record
        b = base64decode(fill)
        length(b) == itemsize || error("bad Zarr fill_value for structured dtype")
        return b
    end
    v = fill isa AbstractString ?
        (fill == "NaN" ? T(NaN) : fill == "Infinity" ? T(Inf) : fill == "-Infinity" ? T(-Inf) :
         error("unsupported Zarr fill_value \"$fill\"")) : T(fill)
    return Vector{UInt8}(reinterpret(UInt8, [htol(v)]))
end

# Copy rows `lo:hi` of a field (starting `offset` bytes into each record, `k` values per record) from the packed records in `buf` into `out`, which holds the column as an n × k matrix in column-major order (sub-arrays flattened in C order). Values are read directly from the record buffer; unaligned loads are fine because `unsafe_load` uses alignment 1.
function _extract_rows!(out::Vector{T}, buf::Vector{UInt8}, itemsize::Int, n::Int,
        offset::Int, k::Int, lo::Int, hi::Int) where {T}
    GC.@preserve buf begin
        base = pointer(buf) + offset
        for j in 1:k
            p = Ptr{T}(base + (j - 1) * sizeof(T))
            o = (j - 1) * n
            @inbounds for i in lo:hi
                out[o+i] = ltoh(unsafe_load(p + (i - 1) * itemsize))
            end
        end
    end
    return out
end

function _extract_rows!(out::Vector{Bool}, buf::Vector{UInt8}, itemsize::Int, n::Int,
        offset::Int, k::Int, lo::Int, hi::Int)
    for j in 1:k
        o = (j - 1) * n
        s = offset + j
        @inbounds for i in lo:hi
            out[o+i] = buf[s+(i-1)*itemsize] != 0x00
        end
    end
    return out
end

function _extract_field(::Type{T}, buf::Vector{UInt8}, itemsize::Int, n::Int, offset::Int, k::Int) where {T}
    n * itemsize <= length(buf) || throw(BoundsError(buf, n * itemsize))
    return _extract_rows!(Vector{T}(undef, n * k), buf, itemsize, n, offset, k, 1, n)
end

"""
    read_zarr_attrs(c::MfxContainer, path) -> Union{Nothing, AbstractDict}

The parsed `.zattrs` JSON of a Zarr group or array, or `nothing` if absent.
"""
function read_zarr_attrs(c::MfxContainer, path::AbstractString)
    key = string(path, "/.zattrs")
    haskey(c.files, key) || return nothing
    return JSON.parse(String(copy(c.files[key])))
end

"""
    read_zarr_array(c::MfxContainer, path)

Read a 1-D Zarr array from a MINFLUX data block, with its native element types. A plain dtype gives a `Vector`; a structured (record) dtype gives a `NamedTuple` of columns, where fields with a sub-array shape `(k,)` become `n × k` matrices.
"""
function read_zarr_array(c::MfxContainer, path::AbstractString)
    key = string(path, "/.zarray")
    haskey(c.files, key) || throw(KeyError(key))
    meta = JSON.parse(String(copy(c.files[key])))
    get(meta, "zarr_format", 2) == 2 || error("$path: only Zarr format 2 is supported")
    shape = Int.(meta["shape"])
    length(shape) == 1 || error("$path: only 1-D Zarr arrays are supported (shape $shape)")
    filters = get(meta, "filters", nothing)
    (filters === nothing || isempty(filters)) || error("$path: Zarr filters are not supported")
    fields, itemsize, T = _parse_dtype(meta["dtype"])
    n = shape[1]
    chunk = Int(meta["chunks"][1])
    compressor = get(meta, "compressor", nothing)
    fill = _fill_bytes(get(meta, "fill_value", nothing), itemsize, T)

    # Zarr v2 stores every chunk at full size (the last one padded), so decode into a buffer padded to whole chunks; zlib chunks inflate straight into it, in parallel when Julia runs with several threads.
    nchunks = cld(n, chunk)
    csize = chunk * itemsize
    buf = Vector{UInt8}(undef, nchunks * csize)
    zlib = compressor !== nothing && compressor["id"] == "zlib"
    _foreach_maybe_threaded(0:nchunks-1) do ci
        ckey = string(path, "/", ci)
        lo = ci * csize
        if !haskey(c.files, ckey)
            for j in 0:chunk-1
                copyto!(buf, lo + j * itemsize + 1, fill, 1, itemsize)
            end
        elseif zlib
            m = _inflate!(buf, lo, csize, c.files[ckey])
            m < 0 && error("$path: chunk $ci is larger than the chunk size")
            m >= min(chunk, n - ci * chunk) * itemsize || error("$path: chunk $ci is too short")
        else
            dec = _decode_chunk(c.files[ckey], compressor)
            length(dec) >= min(csize, (n - ci * chunk) * itemsize) || error("$path: chunk $ci is too short")
            copyto!(buf, lo + 1, dec, 1, min(csize, length(dec)))
        end
    end

    return _decode_records(buf, fields, itemsize, n, T)
end

# Run `f` on each element, using threads only if Julia has more than one (so that, single-threaded, errors are not wrapped in a TaskFailedException).
function _foreach_maybe_threaded(f, itr)
    if Threads.nthreads() > 1 && length(itr) > 1
        Threads.@threads for x in itr
            f(x)
        end
    else
        foreach(f, itr)
    end
    return nothing
end

# Decode `n` packed records of a (plain or structured) dtype from the start of `buf`.
function _decode_records(buf::Vector{UInt8}, fields, itemsize::Int, n::Int, T)
    length(buf) >= n * itemsize || error("not enough data: expected $(n * itemsize) bytes, got $(length(buf))")
    fields === nothing && return _extract_field(T, buf, itemsize, n, 0, 1)
    ks = [prod(f.shape) for f in fields]
    cols = [Vector{f.T}(undef, n * k) for (f, k) in zip(fields, ks)]
    # Work through the records in cache-sized blocks, extracting every field from one block before moving on, instead of one full pass over `buf` per field.
    block = max(1, (256 * 1024) ÷ itemsize)
    for lo in 1:block:n
        hi = min(n, lo + block - 1)
        for (f, k, col) in zip(fields, ks, cols)
            _extract_rows!(col, buf, itemsize, n, f.offset, k, lo, hi)
        end
    end
    names = Tuple(f.name for f in fields)
    return NamedTuple{names}(Tuple(f.shape == () ? col : reshape(col, n, k)
                                   for (f, k, col) in zip(fields, ks, cols)))
end
