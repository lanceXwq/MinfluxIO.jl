"""
    MinfluxIO

Read Abberior Imspector files in Julia:

- `.msr` / `.obf` files: image stacks ([`read_obf`](@ref), [`read_stack`](@ref)) and
  embedded MINFLUX data ([`load_minflux`](@ref))
- Imspector MINFLUX exports as `.npy` or `.mat` ([`load_minflux`](@ref); `.mat` needs `using MAT`)
"""
module MinfluxIO

using Base64: base64decode
using Dates: DateTime, unix2datetime
using Blosc: Blosc
using CodecZlib: ZlibDecompressor, ZlibDecompressorStream, GzipDecompressor
using JSON: JSON
using Tables: Tables
using libdeflate_jll: libdeflate

export read_obf, read_stack, read_stack_bytes, pixelsize, ObfFile, ObfStack, SIUnit
export load_minflux, minflux_measurements, final_localizations, beads
export MinfluxData, MinfluxTable, MinfluxGrid
export MfxContainer, parse_mfxdta, read_zarr_array, read_zarr_attrs, read_npy

include("inflate.jl")
include("obf.jl")
include("mfxdta.jl")
include("zarr.jl")
include("npy.jl")
include("minflux.jl")
include("tables.jl")
include("precompile.jl")

end
