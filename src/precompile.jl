# Precompilation workload: run the MINFLUX loading code on a tiny in-memory data
# block with Imspector's field layout, so the first `load_minflux` call in a
# session does not have to compile it.

using PrecompileTools: @setup_workload, @compile_workload
using CodecZlib: ZlibCompressor

@setup_workload begin
    # `zarr/mfx` dtype as written by Imspector 2026.7 (MINFLUX build), 99 bytes/record
    mfx_dtype = """[["vld","|b1"],["fnl","|b1"],["bot","|b1"],["eot","|b1"],["sta","|u1"],
        ["tim","<f8"],["tid","<u4"],["gri","<u4"],["thi","|u1"],["sqi","|u1"],["itr","<i4"],
        ["loc","<f8",[3]],["lnc","<f8",[3]],["eco","<u4"],["ecc","<u4"],["efo","<f4"],
        ["efc","<f4"],["fbg","<f4"],["cfr","<f2"],["dcr","<f2",[1]]]"""
    grid_dtype = """[["gri","<u4"],["xyz","<f8",[3]],["tim","<f4"],["str","<f4"]]"""
    zarray(dtype, n, compressor) = Vector{UInt8}("""{"chunks":[$n],"compressor":$compressor,
        "dtype":$dtype,"fill_value":null,"filters":null,"order":"C","shape":[$n],"zarr_format":2}""")
    n = 8
    files = Dict{String,Vector{UInt8}}(
        "zarr/.zattrs" => Vector{UInt8}("""{"rois":[],"version":"2.1"}"""),
        "zarr/mfx/.zarray" => zarray(mfx_dtype, n, """{"id":"zlib","level":1}"""),
        "zarr/mfx/.zattrs" => Vector{UInt8}("""{"label":"precompile","acquisition_date":"2026-01-01T00:00:00+01:00",
            "measurement":{"grids":{"mbm_0":{"type":"mbm"}}}}"""),
        "zarr/mfx/0" => transcode(ZlibCompressor, zeros(UInt8, n * 99)),
        "zarr/grd/mbm_0/.zattrs" => Vector{UInt8}("""{"controlling_thi":2}"""),
        "zarr/grd/mbm_0/points/.zarray" => zarray(grid_dtype, 2,
            """{"blocksize":0,"clevel":5,"cname":"lz4","id":"blosc","shuffle":1}"""),
        "zarr/grd/mbm_0/points/0" => Blosc.compress(zeros(UInt8, 2 * 36)),
        "zarr/mbm/mbm_0/.zattrs" => Vector{UInt8}("""{"used":[],"version":"2.2"}"""),
    )
    dirs = ["zarr", "zarr/mfx", "zarr/grd", "zarr/grd/mbm_0", "zarr/grd/mbm_0/points",
        "zarr/mbm", "zarr/mbm/mbm_0"]
    @compile_workload begin
        c = MfxContainer(3, DateTime(2026), files, dirs)
        d = _minflux_from_container(c, "precompile.msr")
        t = final_localizations(d)
        d.table[1]
        beads(d)
        Tables.columns(t)
        sprint(show, MIME"text/plain"(), d)
    end
end

# File-level entry points (their file-reading parts need a real file, so these
# are only compiled as far as type inference reaches).
precompile(read_obf, (String,))
precompile(read_stack, (ObfFile, Int))
precompile(read_stack, (ObfFile, String))
precompile(read_stack_bytes, (ObfFile, ObfStack))
precompile(parse_mfxdta, (Vector{UInt8},))
precompile(load_minflux, (String,))
precompile(read_npy, (String,))
