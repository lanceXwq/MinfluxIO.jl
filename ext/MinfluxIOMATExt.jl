# Reading of Imspector MINFLUX .mat exports (loaded together with MAT.jl).
#
# Imspector writes one MATLAB variable per MINFLUX field (vld, fnl, tim, loc, ...), each with one row per record; vector-valued fields such as `loc` are n × 3.
module MinfluxIOMATExt

using MinfluxIO
using MAT: MAT

# Bring a MATLAB variable into "one row per record" shape.
function _as_rows(name, v, n)
    v isa AbstractArray{<:Number} ||
        error("unsupported variable \"$name\" ($(typeof(v))) in .mat file; " *
              "older (pre-2025) Imspector export layouts are not supported")
    v isa AbstractVector && length(v) == n && return v
    v isa AbstractMatrix || error("variable \"$name\" has unexpected shape $(size(v))")
    size(v, 1) == n && return size(v, 2) == 1 ? vec(v) : v
    size(v, 2) == n && return size(v, 1) == 1 ? vec(v) : permutedims(v)
    error("variable \"$name\" has shape $(size(v)), expected $n rows")
end

function MinfluxIO._load_mat(path::AbstractString)
    vars = MAT.matread(path)
    haskey(vars, "tim") || error("$path does not look like an Imspector MINFLUX export (no \"tim\" variable)")
    n = length(vars["tim"])
    standard = [String(k) for k in keys(MinfluxIO.CANONICAL_TYPES) if haskey(vars, String(k))]
    extra = sort!([k for (k, v) in vars if !(k in standard) && v isa AbstractArray])
    names = vcat(standard, extra)
    cols = Tuple(_as_rows(k, vars[k], n) for k in names)
    table = MinfluxTable(MinfluxIO._canonical_columns(NamedTuple{Tuple(Symbol.(names))}(cols)))
    label = splitext(basename(path))[1]
    return MinfluxData(label, abspath(path), table, nothing, Dict{String,MinfluxGrid}(),
        Dict{String,Any}(), nothing, nothing)
end

end
