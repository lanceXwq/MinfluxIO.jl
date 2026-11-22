# Reader for the OBF container used by Abberior Imspector (.obf / .msr).
#
# The layout follows the official specification:
#   https://imspectordocs.readthedocs.io/en/latest/fileformat.html
# All structures are packed (no padding) and little-endian. An .msr file is an OBF file with extra Imspector-specific data appended after the OBF metadata section; that extra data is not interpreted here.

const OBF_FILE_MAGIC = b"OMAS_BF\n\xff\xff"
const OBF_STACK_MAGIC = b"OMAS_BF_STACK\n\xff\xff"
const OBF_MAX_DIMS = 15
const OMAS_DT_COMPLEX = 0x40000000

const OBF_ELTYPES = Dict{UInt32,DataType}(
    0x00000001 => UInt8,
    0x00000002 => Int8,
    0x00000004 => UInt16,
    0x00000008 => Int16,
    0x00000010 => UInt32,
    0x00000020 => Int32,
    0x00000040 => Float32,
    0x00000080 => Float64,
    0x00001000 => UInt64,
    0x00002000 => Int64,
    0x00010000 => Bool,
)

_rd(io::IO, ::Type{T}) where {T} = ltoh(read(io, T))
_rdstr(io::IO) = String(read(io, _rd(io, UInt32)))

function _obf_eltype(dt::UInt32)
    T = get(OBF_ELTYPES, dt & ~OMAS_DT_COMPLEX, nothing)
    T === nothing && return nothing          # e.g. RGB / RGB4 (not supported)
    return dt & OMAS_DT_COMPLEX != 0 ? Complex{T} : T
end

"""
    SIUnit

Physical unit of an OBF stack axis or of its values. Holds the rational exponents of the SI base units (m, kg, s, A, K, mol, cd, rad, sr) and a scale factor.
"""
struct SIUnit
    exponents::NTuple{9,Rational{Int}}
    scalefactor::Float64
end

const SI_BASE_SYMBOLS = ("m", "kg", "s", "A", "K", "mol", "cd", "rad", "sr")

function _read_siunit(io::IO)
    ex = Vector{Rational{Int}}(undef, 9)
    for i in 1:9
        num = _rd(io, Int32)
        den = _rd(io, Int32)
        ex[i] = den == 0 ? 0 // 1 : Rational{Int}(num, den)
    end
    return SIUnit(Tuple(ex), _rd(io, Float64))
end

function Base.show(io::IO, u::SIUnit)
    parts = String[]
    for (e, sym) in zip(u.exponents, SI_BASE_SYMBOLS)
        iszero(e) && continue
        exp_str = denominator(e) == 1 ? string(numerator(e)) : string(numerator(e), "/", denominator(e))
        push!(parts, isone(e) ? sym : string(sym, "^", exp_str))
    end
    str = isempty(parts) ? "1" : join(parts, "·")
    if u.scalefactor == 1 || u.scalefactor == 0
        print(io, str)
    else
        print(io, u.scalefactor, " ", str)
    end
end

"""
    ObfStack

Header information of one data stack in an OBF/MSR file (the data itself is read on demand with [`read_stack`](@ref)).

Useful fields:
- `index`, `name`, `description`
- `size`: number of samples along each dimension (first dimension varies fastest, which matches Julia's column-major order)
- `lengths`, `offsets`: physical extent and offset of each dimension
- `labels`: dimension labels, e.g. `"ExpControl X"`
- `units`: [`SIUnit`](@ref) of each dimension; `value_unit`: unit of the values
- `eltype`: Julia element type (`nothing` if unsupported)
- `tags`: tag dictionary (e.g. `"imspector"` → Imspector XML metadata)
- `kind`: `:image`, or `:minflux` if the stack holds an embedded MINFLUX data block
"""
struct ObfStack
    index::Int
    name::String
    description::String
    format_version::Int
    size::Vector{Int}
    lengths::Vector{Float64}
    offsets::Vector{Float64}
    dtype::UInt32
    eltype::Union{DataType,Nothing}
    compression::Int
    labels::Vector{String}
    value_unit::Union{SIUnit,Nothing}
    units::Vector{SIUnit}
    col_positions::Vector{Union{Nothing,Vector{Float64}}}
    col_labels::Vector{Union{Nothing,Vector{String}}}
    metadata::String
    tags::Dict{String,String}
    kind::Symbol
    # on-disk layout
    data_offset::Int
    data_length::Int
    samples_written::Int
    flush_block_size::Int
    flush_points::Vector{Int}
    chunk_positions::Vector{Tuple{Int,Int}}
end

Base.size(s::ObfStack) = Tuple(s.size)
Base.ndims(s::ObfStack) = length(s.size)

"""
    pixelsize(stack::ObfStack)

Physical size of one sample along each dimension (`lengths ./ size`), in the units given by `stack.units`.
"""
pixelsize(s::ObfStack) = s.lengths ./ s.size

"""
    ObfFile

An opened OBF/MSR file: file-level header information plus the list of [`ObfStack`](@ref)s. Index it with an integer or a stack name to get a stack.
"""
struct ObfFile
    path::String
    format_version::Int
    description::String
    metadata::Dict{String,String}
    stacks::Vector{ObfStack}
end

Base.length(f::ObfFile) = length(f.stacks)
Base.iterate(f::ObfFile, state...) = iterate(f.stacks, state...)
Base.getindex(f::ObfFile, i::Integer) = f.stacks[i]
Base.getindex(f::ObfFile, name::AbstractString) = _find_stack(f, name)

function _find_stack(f::ObfFile, name::AbstractString)
    hits = findall(s -> s.name == name, f.stacks)
    isempty(hits) && throw(KeyError(name))
    length(hits) > 1 && error("several stacks are named \"$name\" (indices $hits); use an index")
    return f.stacks[only(hits)]
end

# Key/value list: <u32 key len><key><u32 value len><value> ..., terminated by a zero u32 or by `stop`. Used for the stack tag dictionary and (empirically) for the file-level metadata section.
function _read_kv(io::IO, stop::Integer)
    d = Dict{String,String}()
    while stop - position(io) >= 4
        kl = _rd(io, UInt32)
        kl == 0 && break
        stop - position(io) < kl + 4 && break
        key = String(read(io, kl))
        vl = _rd(io, UInt32)
        stop - position(io) < vl && break
        d[key] = String(read(io, vl))
    end
    return d
end

function _peek_kind(io::IO, offset::Int, len::Int, compression::Int, T)
    (T === UInt8 && len >= length(MFXDTA_MAGIC)) || return :image
    seek(io, offset)
    head = read(io, min(len, 1 << 16))
    bytes = if compression == 1
        try
            read(ZlibDecompressorStream(IOBuffer(head)), length(MFXDTA_MAGIC))
        catch
            return :image
        end
    else
        head
    end
    return (length(bytes) >= length(MFXDTA_MAGIC) &&
            view(bytes, 1:length(MFXDTA_MAGIC)) == MFXDTA_MAGIC) ? :minflux : :image
end

function _read_stack_header(io::IO, pos::Int, index::Int)
    seek(io, pos)
    read(io, 16) == OBF_STACK_MAGIC || error("no OBF stack header at byte $pos")
    fv = Int(_rd(io, UInt32))
    rank = Int(_rd(io, UInt32))
    res = [Int(_rd(io, UInt32)) for _ in 1:OBF_MAX_DIMS]
    lens = [_rd(io, Float64) for _ in 1:OBF_MAX_DIMS]
    offs = [_rd(io, Float64) for _ in 1:OBF_MAX_DIMS]
    dt = _rd(io, UInt32)
    compression = Int(_rd(io, UInt32))
    _rd(io, UInt32)                           # compression level
    name_len = _rd(io, UInt32)
    descr_len = _rd(io, UInt32)
    _rd(io, UInt64)                           # reserved
    data_len = Int(_rd(io, UInt64))
    next_pos = _rd(io, UInt64)
    name = String(read(io, name_len))
    descr = String(read(io, descr_len))
    data_off = Int(position(io))
    T = _obf_eltype(dt)

    labels = ["dim$i" for i in 1:rank]
    value_unit = nothing
    units = SIUnit[]
    colpos = Vector{Union{Nothing,Vector{Float64}}}(nothing, rank)
    collab = Vector{Union{Nothing,Vector{String}}}(nothing, rank)
    metadata = ""
    tags = Dict{String,String}()
    samples_written = 0
    flush_block = 0
    flush = Int[]
    chunks = Tuple{Int,Int}[]

    if fv >= 1
        # Stack footer. Fields were appended over format versions, so read each one only if the footer (whose size is stored first) is large enough.
        footer_start = data_off + data_len
        seek(io, footer_start)
        fsz = Int(_rd(io, UInt32))
        fb = IOBuffer(read(io, fsz - 4))
        avail(n) = bytesavailable(fb) >= n
        has_pos = avail(4OBF_MAX_DIMS) ? [_rd(fb, UInt32) for _ in 1:OBF_MAX_DIMS] : zeros(UInt32, OBF_MAX_DIMS)
        has_lab = avail(4OBF_MAX_DIMS) ? [_rd(fb, UInt32) for _ in 1:OBF_MAX_DIMS] : zeros(UInt32, OBF_MAX_DIMS)
        meta_len = avail(4) ? Int(_rd(fb, UInt32)) : 0
        if avail(80 * (1 + OBF_MAX_DIMS))
            value_unit = _read_siunit(fb)
            units = [_read_siunit(fb) for _ in 1:OBF_MAX_DIMS][1:rank]
        end
        nflush = avail(8) ? Int(_rd(fb, UInt64)) : 0
        flush_block = avail(8) ? Int(_rd(fb, UInt64)) : 0
        tag_len = avail(8) ? Int(_rd(fb, UInt64)) : 0
        avail(8) && _rd(fb, UInt64)           # stack_end_disk
        avail(4) && _rd(fb, UInt32)           # min_format_version
        avail(8) && _rd(fb, UInt64)           # stack_end_used_disk
        samples_written = avail(8) ? Int(_rd(fb, UInt64)) : 0
        nchunks = avail(8) ? Int(_rd(fb, UInt64)) : 0

        # Variable-length data after the footer.
        seek(io, footer_start + fsz)
        labels = [_rdstr(io) for _ in 1:rank]
        for i in 1:rank
            has_pos[i] != 0 && (colpos[i] = [reinterpret(Float64, _rd(io, UInt64)) for _ in 1:res[i]])
        end
        for i in 1:rank
            has_lab[i] != 0 && (collab[i] = [_rdstr(io) for _ in 1:res[i]])
        end
        metadata = String(read(io, meta_len))
        flush = [Int(_rd(io, UInt64)) for _ in 1:nflush]
        tag_start = Int(position(io))
        tags = _read_kv(io, tag_start + tag_len)
        seek(io, tag_start + tag_len)
        chunks = [(Int(_rd(io, UInt64)), Int(_rd(io, UInt64))) for _ in 1:nchunks]
    end

    kind = _peek_kind(io, data_off, data_len, compression, T)
    stack = ObfStack(index, name, descr, fv, res[1:rank], lens[1:rank], offs[1:rank], dt, T,
        compression, labels, value_unit, units, colpos, collab, metadata, tags, kind,
        data_off, data_len, samples_written, flush_block, flush, chunks)
    return stack, next_pos
end

"""
    read_obf(path) -> ObfFile

Read the header information of an Imspector `.obf` or `.msr` file: the file-level metadata (e.g. `"ome_xml"`) and the list of data stacks. Stack data is not loaded; use [`read_stack`](@ref) for that.
"""
function read_obf(path::AbstractString)
    fsize = filesize(path)
    open(path, "r") do io
        read(io, 10) == OBF_FILE_MAGIC ||
            throw(ArgumentError("$path is not an OBF/MSR file (bad magic bytes)"))
        fv = Int(_rd(io, UInt32))
        first_pos = _rd(io, UInt64)
        descr = String(read(io, _rd(io, UInt32)))
        meta_pos = fv >= 2 ? _rd(io, UInt64) : UInt64(0)

        stacks = ObfStack[]
        seen = Set{UInt64}()
        pos = first_pos
        while pos != 0
            (pos in seen || pos >= fsize) && error("corrupt stack chain in $path (position $pos)")
            push!(seen, pos)
            stack, pos = _read_stack_header(io, Int(pos), length(stacks) + 1)
            push!(stacks, stack)
        end

        metadata = Dict{String,String}()
        if 0 < meta_pos < fsize
            seek(io, meta_pos)
            metadata = _read_kv(io, fsize)
        end
        return ObfFile(abspath(path), fv, descr, metadata, stacks)
    end
end

"""
    read_stack_bytes(f::ObfFile, stack) -> Vector{UInt8}

Raw (decompressed) bytes of a stack. `stack` may be an [`ObfStack`](@ref), an index or a name.
"""
function read_stack_bytes(f::ObfFile, s::ObfStack)
    isempty(s.chunk_positions) ||
        error("stack $(s.index) (\"$(s.name)\") is stored in chunks (OBF v6), which is not supported yet")
    raw = open(f.path, "r") do io
        seek(io, s.data_offset)
        read(io, s.data_length)
    end
    s.compression == 0 && return raw
    if s.compression == 1
        expected = s.eltype === nothing ? 0 : prod(s.size) * sizeof(s.eltype)
        return _zlib_inflate(raw, expected)
    end
    error("unknown OBF compression type $(s.compression)")
end
read_stack_bytes(f::ObfFile, which::Union{Integer,AbstractString}) = read_stack_bytes(f, f[which])

"""
    read_stack(f::ObfFile, stack) -> Array
    read_stack(path, stack) -> Array

Read the data of a stack as an `Array` with `size(stack)`. `stack` may be an [`ObfStack`](@ref), an index or a name. The first array dimension is the first OBF dimension (usually x). Trailing singleton dimensions are kept; use `dropdims` to remove them.
"""
function read_stack(f::ObfFile, s::ObfStack)
    T = s.eltype
    T === nothing && error("stack $(s.index): unsupported OBF data type 0x$(string(s.dtype, base = 16))")
    raw = read_stack_bytes(f, s)
    n = prod(s.size)
    nb = n * sizeof(T)
    if length(raw) < nb
        @warn "stack $(s.index) (\"$(s.name)\") holds fewer samples than its size; missing samples are set to zero"
        append!(raw, zeros(UInt8, nb - length(raw)))
    end
    data = if T === Bool
        map(!iszero, view(raw, 1:nb))
    else
        v = copy(reinterpret(T, view(raw, 1:nb)))
        ENDIAN_BOM == 0x04030201 ? v : ltoh.(v)
    end
    return reshape(data, Tuple(s.size))
end
read_stack(f::ObfFile, which::Union{Integer,AbstractString}) = read_stack(f, f[which])
read_stack(path::AbstractString, which::Union{Integer,AbstractString}) = read_stack(read_obf(path), which)

function Base.show(io::IO, s::ObfStack)
    print(io, "ObfStack(", s.index, ", \"", s.name, "\", ", join(s.size, "×"), ", ",
        something(s.eltype, "unsupported"), s.kind === :minflux ? ", MINFLUX data" : "", ")")
end

function Base.show(io::IO, ::MIME"text/plain", f::ObfFile)
    println(io, "ObfFile \"", basename(f.path), "\" (OBF format version ", f.format_version, ", ",
        length(f.stacks), " stacks)")
    for s in f.stacks
        desc = s.kind === :minflux ? "MINFLUX data block" :
               string(join(s.size, "×"), " ", something(s.eltype, "unsupported"))
        name = isempty(s.name) ? "(unnamed)" : s.name
        print(io, "  [", lpad(s.index, ndigits(length(f.stacks))), "] ", name, "  ", desc)
        s === last(f.stacks) || println(io)
    end
end
Base.show(io::IO, f::ObfFile) = print(io, "ObfFile(\"", basename(f.path), "\", ", length(f.stacks), " stacks)")
