# Fast inflation of zlib data into preallocated memory.
#
# OBF stacks and Imspector's Zarr chunks have a known decompressed size, so we
# use libdeflate's one-shot decompressor directly into the destination. It is
# several times faster than streaming zlib. When the size is not known, we fall
# back to CodecZlib's streaming decoder.

const LIBDEFLATE_SUCCESS = Cint(0)
const LIBDEFLATE_INSUFFICIENT_SPACE = Cint(3)

"""
    _inflate!(dest, offset, maxlen, src) -> Int

Inflate the zlib stream `src` into `dest[offset+1 : offset+maxlen]`. Returns the
number of bytes written, or `-1` if the output does not fit in `maxlen` bytes.
Thread-safe: every call uses its own decompressor.
"""
function _inflate!(dest::Vector{UInt8}, offset::Int, maxlen::Int, src::Vector{UInt8})
    offset + maxlen <= length(dest) || throw(BoundsError(dest, offset + maxlen))
    d = ccall((:libdeflate_alloc_decompressor, libdeflate), Ptr{Cvoid}, ())
    d == C_NULL && throw(OutOfMemoryError())
    try
        actual = Ref{Csize_t}(0)
        ret = GC.@preserve dest src ccall((:libdeflate_zlib_decompress, libdeflate), Cint,
            (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, Ref{Csize_t}),
            d, pointer(src), length(src), pointer(dest, offset + 1), maxlen, actual)
        ret == LIBDEFLATE_SUCCESS && return Int(actual[])
        ret == LIBDEFLATE_INSUFFICIENT_SPACE && return -1
        error("corrupt zlib data (libdeflate error $ret)")
    finally
        ccall((:libdeflate_free_decompressor, libdeflate), Cvoid, (Ptr{Cvoid},), d)
    end
end

# Inflate `src` when the decompressed size is expected to be `expected` bytes;
# falls back to the streaming decoder if the guess is too small.
function _zlib_inflate(src::Vector{UInt8}, expected::Int)
    if expected > 0
        dest = Vector{UInt8}(undef, expected)
        n = _inflate!(dest, 0, expected, src)
        n >= 0 && return resize!(dest, n)
    end
    return transcode(ZlibDecompressor, src)
end
