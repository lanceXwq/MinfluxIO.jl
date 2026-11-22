# Reader for the MINFLUX data block ("MFXDTA") that Imspector embeds in .msr files.
#
# This format is not publicly documented; the layout below was inferred from
# example files written by Imspector 2026.7 (MINFLUX build) and may change
# between Imspector versions. The block is stored as an
# (otherwise meaningless) UInt8 OBF stack and contains a serialized Zarr v2
# directory tree:
#
#   "MFXDTA\0"          7-byte magic
#   UInt32              version (3 in the files seen so far)
#   UInt32              Unix timestamp; matches the end of acquisition/saving
#   zero padding        up to the first record (offset 147 for version 3)
#   records             repeated until a zero name length:
#     UInt64 n, n bytes path (e.g. "zarr/mfx/.zarray"), UInt8 is_directory,
#     and, for files only: UInt64 m, m bytes content
#   zero padding

const MFXDTA_MAGIC = b"MFXDTA\0"

"""
    MfxContainer

The unpacked contents of an embedded MINFLUX data block: the files of the Zarr v2
store (`files`, path => bytes), its directories (`dirs`), the block `version` and
`timestamp` (UTC; appears to be the time the data were written).

Arrays in it can be read with [`read_zarr_array`](@ref) and attributes with
[`read_zarr_attrs`](@ref), e.g. `read_zarr_array(c, "zarr/mfx")`.
"""
struct MfxContainer
    version::Int
    timestamp::DateTime
    files::Dict{String,Vector{UInt8}}
    dirs::Vector{String}
end

"""
    parse_mfxdta(bytes) -> MfxContainer

Unpack an embedded MINFLUX data block (the decompressed bytes of an OBF stack whose
`kind` is `:minflux`).
"""
function parse_mfxdta(bytes::AbstractVector{UInt8})
    bytes isa Vector{UInt8} || (bytes = Vector{UInt8}(bytes))
    nm = length(MFXDTA_MAGIC)
    (length(bytes) >= nm + 8 && view(bytes, 1:nm) == MFXDTA_MAGIC) ||
        throw(ArgumentError("not a MINFLUX data block (missing MFXDTA magic)"))
    io = IOBuffer(bytes)
    seek(io, nm)
    version = Int(_rd(io, UInt32))
    timestamp = unix2datetime(_rd(io, UInt32))

    # The first record is the "zarr" root directory; locate it rather than relying
    # on a fixed header length.
    marker = vcat(reinterpret(UInt8, [htol(UInt64(4))]), Vector{UInt8}("zarr"))
    r = findnext(marker, bytes, nm + 9)
    r === nothing && error("MINFLUX data block (version $version): no Zarr store found")
    seek(io, first(r) - 1)

    files = Dict{String,Vector{UInt8}}()
    dirs = String[]
    while bytesavailable(io) >= 8
        n = _rd(io, UInt64)
        n == 0 && break
        bytesavailable(io) >= n + 1 || error("truncated MINFLUX data block")
        path = String(read(io, n))
        isdir = read(io, UInt8)
        if isdir == 0x01
            push!(dirs, path)
        elseif isdir == 0x00
            m = _rd(io, UInt64)
            bytesavailable(io) >= m || error("truncated MINFLUX data block (in \"$path\")")
            files[path] = read(io, m)
        else
            error("unexpected record flag $isdir for \"$path\" in MINFLUX data block")
        end
    end
    return MfxContainer(version, timestamp, files, dirs)
end

# Names of the sub-groups (directories) directly below `path`.
_subgroups(c::MfxContainer, path::AbstractString) =
    sort!([basename(d) for d in c.dirs if dirname(d) == path])

function Base.show(io::IO, c::MfxContainer)
    print(io, "MfxContainer(version ", c.version, ", ", c.timestamp, " UTC, ",
        length(c.files), " files)")
end
