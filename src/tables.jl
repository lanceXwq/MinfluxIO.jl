# Tables.jl interface for MinfluxTable: matrix-valued columns (e.g. `loc`, n × 3) are split into one column per component.

const _XYZ = ("x", "y", "z")

function _flat_columns(t::MinfluxTable)
    names = Symbol[]
    cols = AbstractVector[]
    for (name, col) in pairs(_columns(t))
        if col isa AbstractVector
            push!(names, name)
            push!(cols, col)
        else
            k = size(col, 2)
            for j in 1:k
                suffix = k == 3 ? _XYZ[j] : string(j)
                push!(names, Symbol(name, "_", suffix))
                push!(cols, col[:, j])
            end
        end
    end
    return NamedTuple{Tuple(names)}(Tuple(cols))
end

Tables.istable(::Type{<:MinfluxTable}) = true
Tables.columnaccess(::Type{<:MinfluxTable}) = true
Tables.columns(t::MinfluxTable) = _flat_columns(t)
