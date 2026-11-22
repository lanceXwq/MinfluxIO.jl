# Reader for NumPy .npy files, as written by Imspector's MINFLUX "export as NumPy" (a 1-D structured array with one record per MINFLUX iteration).
#
# Format (documented by NumPy):
#   https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
#   "\x93NUMPY", UInt8 major, UInt8 minor,
#   header length (UInt16 for v1.0, UInt32 for v2.0/v3.0),
#   header: a Python dict literal {'descr': ..., 'fortran_order': ..., 'shape': ...}
#   (Latin-1 for v1/v2, UTF-8 for v3), padded with spaces and a final newline,
#   then the raw array data.

const NPY_MAGIC = b"\x93NUMPY"

# --- tiny parser for the Python literals that appear in .npy headers ---------

mutable struct _PyLiteralParser
    s::Vector{Char}
    i::Int
end

function _skipws!(p::_PyLiteralParser)
    while p.i <= length(p.s) && isspace(p.s[p.i])
        p.i += 1
    end
end

_peek(p::_PyLiteralParser) = (_skipws!(p); p.i <= length(p.s) ? p.s[p.i] : '\0')

function _expect!(p::_PyLiteralParser, c::Char)
    _peek(p) == c || error("malformed .npy header: expected '$c' at position $(p.i)")
    p.i += 1
end

function _parse_pyvalue!(p::_PyLiteralParser)
    c = _peek(p)
    if c == '{'
        p.i += 1
        d = Dict{String,Any}()
        while _peek(p) != '}'
            key = _parse_pyvalue!(p)
            _expect!(p, ':')
            d[string(key)] = _parse_pyvalue!(p)
            _peek(p) == ',' && (p.i += 1)
        end
        p.i += 1
        return d
    elseif c == '[' || c == '('
        close = c == '[' ? ']' : ')'
        p.i += 1
        v = Any[]
        while _peek(p) != close
            push!(v, _parse_pyvalue!(p))
            _peek(p) == ',' && (p.i += 1)
        end
        p.i += 1
        return v
    elseif c == '\'' || c == '"'
        p.i += 1
        buf = IOBuffer()
        while p.i <= length(p.s) && p.s[p.i] != c
            if p.s[p.i] == '\\' && p.i < length(p.s)
                p.i += 1
            end
            write(buf, p.s[p.i])
            p.i += 1
        end
        p.i += 1
        return String(take!(buf))
    elseif c == '-' || isdigit(c)
        j = p.i
        p.i += 1
        while p.i <= length(p.s) && isdigit(p.s[p.i])
            p.i += 1
        end
        val = parse(Int, String(p.s[j:p.i-1]))
        p.i <= length(p.s) && p.s[p.i] == 'L' && (p.i += 1)    # Python 2 long suffix
        return val
    elseif isletter(c)
        j = p.i
        while p.i <= length(p.s) && (isletter(p.s[p.i]) || isdigit(p.s[p.i]) || p.s[p.i] == '_')
            p.i += 1
        end
        word = String(p.s[j:p.i-1])
        word == "True" && return true
        word == "False" && return false
        word == "None" && return nothing
        error("malformed .npy header: unexpected \"$word\"")
    end
    error("malformed .npy header: unexpected '$c' at position $(p.i)")
end

function _read_npy_header(io::IO)
    read(io, 6) == NPY_MAGIC || throw(ArgumentError("not a .npy file (bad magic bytes)"))
    major = read(io, UInt8)
    minor = read(io, UInt8)
    hlen = major == 1 ? Int(_rd(io, UInt16)) :
           major in (2, 3) ? Int(_rd(io, UInt32)) :
           error("unsupported .npy format version $major.$minor")
    raw = read(io, hlen)
    text = major == 3 ? String(raw) : String(map(Char, raw))    # v1/v2 headers are Latin-1
    hdr = _parse_pyvalue!(_PyLiteralParser(collect(text), 1))
    (hdr isa Dict && all(k -> haskey(hdr, k), ("descr", "fortran_order", "shape"))) ||
        error("malformed .npy header: $text")
    return hdr
end

"""
    read_npy(path)

Read a 1-D NumPy `.npy` array with its native element types. A plain dtype gives a `Vector`; a structured (record) dtype, as in Imspector's MINFLUX export, gives a `NamedTuple` of columns, where fields with a sub-array shape `(k,)` become `n × k` matrices. Padding fields are skipped.
"""
function read_npy(path::AbstractString)
    open(path, "r") do io
        hdr = _read_npy_header(io)
        shape = Int.(hdr["shape"])
        length(shape) == 1 || error("$path: only 1-D .npy arrays are supported (shape $(Tuple(shape)))")
        fields, itemsize, T = _parse_dtype(hdr["descr"])
        n = shape[1]
        buf = read(io, n * itemsize)
        length(buf) == n * itemsize || error("$path: file is truncated")
        return _decode_records(buf, fields, itemsize, n, T)
    end
end

function _load_npy(path::AbstractString)
    cols = read_npy(path)
    cols isa NamedTuple ||
        error("$path holds a plain array, not a structured MINFLUX table")
    haskey(cols, :tim) || error("$path does not look like an Imspector MINFLUX export (no \"tim\" field)")
    table = MinfluxTable(_canonical_columns(cols))
    return MinfluxData(splitext(basename(path))[1], abspath(path), table, nothing,
        Dict{String,MinfluxGrid}(), Dict{String,Any}(), nothing, nothing)
end
