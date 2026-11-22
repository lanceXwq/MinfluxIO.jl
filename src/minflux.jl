# User-facing MINFLUX data types and loaders.

# Element types used for the standard MINFLUX fields, whatever the source file. Integer fields become `Int` (so that e.g. `diff(tid)` cannot wrap around) and Float16 fields become Float32 (Float16 sums overflow at 65504).
const CANONICAL_TYPES = (
    vld = Bool, fnl = Bool, bot = Bool, eot = Bool,
    sta = Int, tim = Float64, tid = Int, gri = Int, thi = Int, sqi = Int, itr = Int,
    loc = Float64, lnc = Float64, eco = Int, ecc = Int,
    efo = Float32, efc = Float32, fbg = Float32, cfr = Float32, dcr = Float32,
)

_default_type(::Type{Bool}) = Bool
_default_type(::Type{<:Integer}) = Int
_default_type(::Type{Float16}) = Float32
_default_type(::Type{T}) where {T} = T

function _canonical_column(name::Symbol, col::AbstractArray)
    col isa AbstractMatrix && size(col, 2) == 1 && (col = vec(col))
    T = haskey(CANONICAL_TYPES, name) ? CANONICAL_TYPES[name] : _default_type(eltype(col))
    col isa Array{T} && return col                      # already the right type: no copy
    T === Bool && return Array{Bool}(map(!iszero, col))
    return Array{T}(col)
end

_canonical_columns(cols::NamedTuple) =
    NamedTuple{keys(cols)}(map(_canonical_column, keys(cols), values(cols)))

"""
    MinfluxTable

Column table of MINFLUX localizations (one row per iteration record). Columns are accessed as properties (`t.tim`, `t.loc`, ...); vector-valued fields such as `loc` are `n × 3` matrices. Rows can be selected with `t[mask]` or `t[indices]`, and `t[i]` returns one row as a `NamedTuple`.

Implements the Tables.jl interface (matrix columns are split into `loc_x`, `loc_y`, `loc_z`, ...), so `DataFrame(t)` works.
"""
struct MinfluxTable{NT<:NamedTuple}
    columns::NT
end

_columns(t::MinfluxTable) = getfield(t, :columns)
Base.getproperty(t::MinfluxTable, s::Symbol) = getproperty(_columns(t), s)
Base.propertynames(t::MinfluxTable) = keys(_columns(t))
Base.haskey(t::MinfluxTable, s::Symbol) = haskey(_columns(t), s)
Base.length(t::MinfluxTable) = isempty(_columns(t)) ? 0 : size(first(_columns(t)), 1)
Base.lastindex(t::MinfluxTable) = length(t)

_rows(c::AbstractVector, r) = c[r]
_rows(c::AbstractMatrix, r) = c[r, :]
Base.getindex(t::MinfluxTable, r::Union{AbstractVector,Colon}) =
    MinfluxTable(map(c -> _rows(c, r), _columns(t)))
Base.getindex(t::MinfluxTable, i::Integer) = map(c -> _rows(c, i), _columns(t))

function Base.show(io::IO, t::MinfluxTable)
    print(io, "MinfluxTable(", length(t), " rows; ", join(keys(_columns(t)), ", "), ")")
end

"""
    MinfluxGrid

A grid from a MINFLUX measurement (e.g. the hexagonal search grid `search_0` or the bead grids `mbm_0`, `mbm_1` used for drift correction): its `attributes` and its `points` (a [`MinfluxTable`](@ref), or `nothing` if the grid has none). For bead grids the points are the registered bead positions (`gri` bead id, `xyz`, `tim`, `str`).
"""
struct MinfluxGrid
    name::String
    attributes::Union{Nothing,AbstractDict}
    points::Union{Nothing,MinfluxTable}
end

Base.show(io::IO, g::MinfluxGrid) = print(io, "MinfluxGrid(\"", g.name, "\", ",
    g.points === nothing ? "no" : length(g.points), " points)")

"""
    MinfluxData

One MINFLUX measurement.

- `label`: measurement name (for `.npy`/`.mat` exports: the file name)
- `source`: path of the file it was read from
- `table`: the localization records, a [`MinfluxTable`](@ref)
- `attributes`: measurement metadata (acquisition date, sequence, lasers, ...)
- `grids`: `Dict` of [`MinfluxGrid`](@ref)s (search grid, bead grids)
- `mbm`: bead-based drift-correction settings, per bead grid
- `root_attributes`: top-level metadata (e.g. regions of interest)
- `container`: the raw [`MfxContainer`](@ref), for anything not exposed above

Only `label`, `source` and `table` are available from `.npy`/`.mat` exports; the other fields are then `nothing` or empty.
"""
struct MinfluxData
    label::String
    source::String
    table::MinfluxTable
    attributes::Union{Nothing,AbstractDict}
    grids::Dict{String,MinfluxGrid}
    mbm::Dict{String,Any}
    root_attributes::Union{Nothing,AbstractDict}
    container::Union{Nothing,MfxContainer}
end

"""
    final_localizations(d::MinfluxData)
    final_localizations(t::MinfluxTable)

The valid localizations of the final iteration (rows with `vld .& fnl`).
"""
final_localizations(t::MinfluxTable) = t[t.vld .& t.fnl]
final_localizations(d::MinfluxData) = final_localizations(d.table)

"""
    beads(d::MinfluxData) -> Dict{String,MinfluxTable}

The points of the bead (MBM, drift-correction) grids, per grid name: registered bead positions with columns typically `gri` (bead id), `xyz`, `tim` and `str`. Whether beads were actually used for correction is recorded in `d.mbm`.
"""
function beads(d::MinfluxData)
    out = Dict{String,MinfluxTable}()
    for (name, g) in d.grids
        g.points === nothing && continue
        type = g.attributes === nothing ? nothing : get(g.attributes, "type", nothing)
        (type == "mbm" || (type === nothing && startswith(name, "mbm"))) && (out[name] = g.points)
    end
    return out
end

function _minflux_from_container(c::MfxContainer, source::AbstractString)
    haskey(c.files, "zarr/mfx/.zarray") || error("MINFLUX data block has no \"zarr/mfx\" array")
    attrs = read_zarr_attrs(c, "zarr/mfx")
    table = MinfluxTable(_canonical_columns(read_zarr_array(c, "zarr/mfx")))
    label = attrs === nothing ? "" : string(get(attrs, "label", ""))

    grids = Dict{String,MinfluxGrid}()
    for g in _subgroups(c, "zarr/grd")
        gpath = "zarr/grd/$g"
        gattrs = read_zarr_attrs(c, gpath)
        mattrs = attrs === nothing ? nothing : get(get(get(attrs, "measurement", Dict()), "grids", Dict()), g, nothing)
        if mattrs !== nothing && (gattrs === nothing || !haskey(gattrs, "type"))
            gattrs = merge(Dict{String,Any}(mattrs), gattrs === nothing ? Dict{String,Any}() : Dict{String,Any}(gattrs))
        end
        points = haskey(c.files, "$gpath/points/.zarray") ?
                 MinfluxTable(_canonical_columns(read_zarr_array(c, "$gpath/points"))) : nothing
        grids[g] = MinfluxGrid(g, gattrs, points)
    end

    mbm = Dict{String,Any}()
    for m in _subgroups(c, "zarr/mbm")
        mbm[m] = read_zarr_attrs(c, "zarr/mbm/$m")
    end
    return MinfluxData(label, abspath(source), table, attrs, grids, mbm,
        read_zarr_attrs(c, "zarr"), c)
end

function _msr_containers(path::AbstractString)
    f = read_obf(path)
    stacks = filter(s -> s.kind === :minflux, f.stacks)
    return [parse_mfxdta(read_stack_bytes(f, s)) for s in stacks]
end

_container_label(c::MfxContainer) =
    string(get(something(read_zarr_attrs(c, "zarr/mfx"), Dict()), "label", ""))

"""
    minflux_measurements(path) -> Vector{String}

Labels of the MINFLUX measurements stored in an `.msr` file.
"""
minflux_measurements(path::AbstractString) = map(_container_label, _msr_containers(path))

function _load_msr(path::AbstractString; measurement = nothing)
    cs = _msr_containers(path)
    isempty(cs) && error("$path contains no MINFLUX data")
    labels = map(_container_label, cs)
    i = if measurement === nothing
        length(cs) == 1 || error("$path contains $(length(cs)) MINFLUX measurements; pick one with " *
                                 "`measurement` (an index or one of: $(join(repr.(labels), ", ")))")
        1
    elseif measurement isa Integer
        measurement
    else
        j = findfirst(==(measurement), labels)
        j === nothing && error("no MINFLUX measurement \"$measurement\" in $path; available: $(join(repr.(labels), ", "))")
        j
    end
    return _minflux_from_container(cs[i], path)
end

# Implemented in the MAT.jl package extension (ext/MinfluxIOMATExt.jl).
function _load_mat end

"""
    load_minflux(path; measurement = nothing) -> MinfluxData

Load MINFLUX data from an Imspector `.msr`/`.obf` file, or from an Imspector MINFLUX export as `.npy` or `.mat` (the latter requires `using MAT`).

If an `.msr` file holds several MINFLUX measurements, select one with `measurement` (an index or a label, see [`minflux_measurements`](@ref)).
"""
function load_minflux(path::AbstractString; measurement = nothing)
    isfile(path) || throw(ArgumentError("no such file: $path"))
    ext = lowercase(splitext(path)[2])
    ext in (".msr", ".obf") && return _load_msr(path; measurement)
    ext == ".npy" && return _load_npy(path)
    if ext == ".mat"
        Base.get_extension(@__MODULE__, :MinfluxIOMATExt) === nothing &&
            error("reading .mat files requires MAT.jl; run `using MAT` first")
        return _load_mat(path)
    end
    throw(ArgumentError("unsupported file type \"$ext\" (expected .msr, .obf, .npy or .mat)"))
end

function Base.show(io::IO, d::MinfluxData)
    print(io, "MinfluxData(\"", d.label, "\", ", length(d.table), " rows)")
end

function Base.show(io::IO, ::MIME"text/plain", d::MinfluxData)
    t = d.table
    println(io, "MinfluxData \"", d.label, "\"")
    print(io, "  source: ", basename(d.source))
    if d.attributes !== nothing && haskey(d.attributes, "acquisition_date")
        print(io, "\n  acquired: ", d.attributes["acquisition_date"])
    end
    print(io, "\n  records: ", length(t))
    if haskey(t, :vld) && haskey(t, :fnl)
        fin = t.vld .& t.fnl
        print(io, " (", count(t.vld), " valid, ", count(fin), " valid final)")
        if haskey(t, :thi) && any(fin)
            ths = sort!(unique(t.thi[fin]))
            print(io, "\n  final localizations per thread (thi): ",
                join(("$th => $(count(fin .& (t.thi .== th)))" for th in ths), ", "))
        end
        if haskey(t, :tid) && any(fin)
            print(io, "\n  traces (tid) with final localizations: ", length(unique(t.tid[fin])))
        end
    end
    if haskey(t, :tim) && length(t) > 0
        print(io, "\n  time: ", round(minimum(t.tim), digits = 2), " – ",
            round(maximum(t.tim), digits = 2), " s")
    end
    if !isempty(d.grids)
        print(io, "\n  grids: ", join((sprint(show, d.grids[k]) for k in sort!(collect(keys(d.grids)))), ", "))
    end
end
