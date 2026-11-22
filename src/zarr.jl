# NumPy dtype handling (shared with the .npy reader) and a minimal reader for the
# Zarr v2 arrays inside a MINFLUX data block.
#
# Supports what Imspector writes: 1-D arrays of plain or structured (record)
# NumPy dtypes, little-endian, compressed with zlib, gzip or blosc (or not at all),
# without filters. A general Zarr implementation is not needed for these files.

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

function _extract_field(M::AbstractMatrix{UInt8}, f::ZarrField)
    n = size(M, 2)
    bytes = vec(M[f.offset+1:f.offset+f.nbytes, :])
    vals = f.T === Bool ? map(!iszero, bytes) : copy(reinterpret(f.T, bytes))
    ENDIAN_BOM == 0x04030201 || f.T === Bool || (vals = ltoh.(vals))
    f.shape == () && return vals
    k = prod(f.shape)
    return permutedims(reshape(vals, k, n))  # n × k (sub-array flattened, C order)
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

Read a 1-D Zarr array from a MINFLUX data block, with its native element types.
A plain dtype gives a `Vector`; a structured (record) dtype gives a `NamedTuple`
of columns, where fields with a sub-array shape `(k,)` become `n × k` matrices.
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

    # Zarr v2 stores every chunk at full size (the last one padded), so decode
    # into a buffer padded to whole chunks; zlib chunks inflate straight into it.
    nchunks = cld(n, chunk)
    csize = chunk * itemsize
    buf = Vector{UInt8}(undef, nchunks * csize)
    zlib = compressor !== nothing && compressor["id"] == "zlib"
    for ci in 0:nchunks-1
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

# Decode `n` packed records of a (plain or structured) dtype from `buf`.
function _decode_records(buf::AbstractVector{UInt8}, fields, itemsize::Int, n::Int, T)
    length(buf) >= n * itemsize || error("not enough data: expected $(n * itemsize) bytes, got $(length(buf))")
    buf = view(buf, 1:n*itemsize)
    if fields === nothing
        v = T === Bool ? map(!iszero, buf) : copy(reinterpret(T, buf))
        return ENDIAN_BOM == 0x04030201 || T === Bool ? v : ltoh.(v)
    end
    M = reshape(buf, itemsize, n)
    names = Tuple(f.name for f in fields)
    return NamedTuple{names}(Tuple(_extract_field(M, f) for f in fields))
end
