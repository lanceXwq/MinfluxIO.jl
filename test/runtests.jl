using MinfluxIO
using Test
using Base64
using Tables
using TOML

include("synthetic.jl")

# Optional tests on real data: set MINFLUXIO_TESTDATA to a directory holding the files listed in `REAL_FILES` (they are not part of the repository).
const TESTDATA = get(ENV, "MINFLUXIO_TESTDATA", "")

@testset "MinfluxIO.jl" begin
    tmp = mktempdir()

    img16 = Int16.(reshape(0:(6*5*3-1), 6, 5, 3))
    img32 = rand(Float32, 4, 7)
    blob_a, cols_a = minflux_store("meas_A"; drop_chunk = true)
    blob_b, _ = minflux_store("meas_B"; n = 7)
    blobs = [
        stack_blob("Img16", img16; labels = ["ExpControl X", "ExpControl Y", "ExpControl Z"]),
        stack_blob("Img32 raw", img32; compress = false),
        stack_blob("", zeros(UInt8, length(blob_a)); raw_bytes = blob_a, tags = Dict("minflux" => "{}")),
    ]
    one_path = joinpath(tmp, "one.msr")
    write(one_path, obf_bytes(blobs))
    two_path = joinpath(tmp, "two.msr")
    write(two_path, obf_bytes([blobs; [stack_blob("", zeros(UInt8, length(blob_b)); raw_bytes = blob_b)]]))

    @testset "OBF container" begin
        f = read_obf(one_path)
        @test length(f) == 3
        @test f.format_version == 2
        @test f.metadata == Dict("ome_xml" => "<OME/>")
        s = f[1]
        @test s.name == "Img16" && size(s) == (6, 5, 3) && s.eltype == Int16
        @test s.labels == ["ExpControl X", "ExpControl Y", "ExpControl Z"]
        @test sprint(show, s.units[1]) == "m" && sprint(show, s.units[3]) == "1"
        @test pixelsize(s) ≈ [1e-7, 1e-7, 1e-7]
        @test s.kind === :image && f[3].kind === :minflux
        @test f[3].tags == Dict("minflux" => "{}")
        @test read_stack(f, 1) == img16
        @test read_stack(f, "Img32 raw") == img32
        @test read_stack(one_path, 1) == img16
        @test f["Img16"] === f[1]
        @test_throws KeyError f["nope"]
        @test occursin("MINFLUX data block", sprint(show, MIME"text/plain"(), f))
        bad = joinpath(tmp, "bad.msr")
        write(bad, b"not an obf file at all")
        @test_throws ArgumentError read_obf(bad)
    end

    @testset "MFXDTA + Zarr" begin
        c = parse_mfxdta(blob_a)
        @test c.version == 3
        @test string(c.timestamp) == "2026-11-13T15:56:28"
        raw = read_zarr_array(c, "zarr/mfx")
        @test eltype(raw.cfr) == Float16 && eltype(raw.tid) == UInt32   # native types
        @test raw.tim[1:10] == cols_a.tim[1:10]
        @test all(iszero, raw.tim[11:20])                               # dropped chunk → fill value
        @test raw.loc[21:25, :] == cols_a.loc[21:25, :]
        @test read_zarr_attrs(c, "zarr")["version"] == "2.1"
        @test read_zarr_attrs(c, "zarr/nothing") === nothing
        @test_throws ArgumentError parse_mfxdta(b"nope, not this")
    end

    @testset "load_minflux (.msr)" begin
        d = load_minflux(one_path)
        t = d.table
        @test d.label == "meas_A"
        @test length(t) == 25
        @test t.vld isa Vector{Bool} && t.tid isa Vector{Int} && t.cfr isa Vector{Float32}
        @test t.dcr isa Vector{Float32}                                   # (n, 1) squeezed
        keep = [1:10; 21:25]
        @test t.tid[keep] == cols_a.tid[keep]
        @test t.thi[keep] == cols_a.thi[keep]
        @test t.cfr[keep] == Float32.(cols_a.cfr[keep])
        @test t.loc[keep, :] == cols_a.loc[keep, :]
        @test t.vld[keep] == cols_a.vld[keep]
        fl = final_localizations(d)
        @test length(fl) == count(t.vld .& t.fnl)
        @test t[3].tim == t.tim[3] && t[3].loc == t.loc[3, :]
        b = beads(d)
        @test collect(keys(b)) == ["mbm_0"]
        @test b["mbm_0"].gri == 61:64 && b["mbm_0"].xyz[2, :] == [2e-6, 4e-6, 0.0]
        @test b["mbm_0"].str == Float32.(1:4)
        @test d.mbm["mbm_0"]["used"] == []
        @test d.attributes["acquisition_date"] == "2026-01-01T00:00:00+01:00"
        cols = Tables.columns(t)
        @test Tables.istable(t)
        @test (:loc_x, :loc_y, :loc_z) ⊆ Tables.columnnames(cols)
        @test cols.loc_y == t.loc[:, 2]
        @test occursin("valid final", sprint(show, MIME"text/plain"(), d))

        @test minflux_measurements(two_path) == ["meas_A", "meas_B"]
        @test_throws ErrorException load_minflux(two_path)
        @test load_minflux(two_path; measurement = "meas_B").label == "meas_B"
        @test length(load_minflux(two_path; measurement = 2).table) == 7
        @test_throws ErrorException load_minflux(two_path; measurement = "meas_C")
        noimg = joinpath(tmp, "nominflux.msr")
        write(noimg, obf_bytes(blobs[1:2]))
        @test_throws ErrorException load_minflux(noimg)
        @test_throws ArgumentError load_minflux(joinpath(tmp, "x.tif"))
    end

    @testset "load_minflux (.npy)" begin
        n = 25
        buf, cols = mfx_records(n)
        check(t) = (t.tid == cols.tid && t.thi == cols.thi && t.vld == cols.vld && t.fnl == cols.fnl &&
                    t.tim == cols.tim && t.loc == cols.loc && t.cfr == Float32.(cols.cfr) &&
                    t.dcr == Float32.(cols.dcr) && t.itr == cols.itr)
        for version in (1, 2)
            p = joinpath(tmp, "export_v$version.npy")
            write(p, npy_bytes(MFX_DTYPE_PY, n, buf; version))
            d = load_minflux(p)
            @test d.label == "export_v$version" && length(d.table) == n
            @test d.table.tid isa Vector{Int} && d.table.cfr isa Vector{Float32}
            @test check(d.table)
        end
        raw = read_npy(joinpath(tmp, "export_v1.npy"))
        @test raw.cfr isa Vector{Float16} && raw.tid isa Vector{UInt32}         # native types
        # conversion during extraction: uint8 flags -> Bool, uint32 -> Int, sub-array kept
        pconv = joinpath(tmp, "convert.npy")
        recs = UInt8[]
        for (v, t) in ((0x00, 7), (0x01, 8), (0x02, 9))
            append!(recs, [v], le(UInt32(t)), le(1.0t), le(2.0t))
        end
        write(pconv, npy_bytes("[('vld', '|u1'), ('tim', '<u4'), ('xyz', '<f8', (2,))]", 3, recs))
        tc = load_minflux(pconv).table
        @test tc.vld == [false, true, true] && tc.vld isa Vector{Bool}
        @test tc.tim == [7.0, 8.0, 9.0] && tc.tim isa Vector{Float64}           # canonical: Float64
        @test tc.xyz == [7.0 14.0; 8.0 16.0; 9.0 18.0]
        @test read_npy(pconv).vld == UInt8[0, 1, 2]                             # native unchanged
        # files written by NumPy itself (see test/data/generate_npy_fixtures.py)
        data = joinpath(@__DIR__, "data")
        @test check(load_minflux(joinpath(data, "numpy_mfx_v1.npy")).table)
        @test check(load_minflux(joinpath(data, "numpy_mfx_aligned_v3.npy")).table)   # padding fields
        @test read_npy(joinpath(data, "numpy_plain.npy")) == 0:4
        @test_throws ErrorException load_minflux(joinpath(data, "numpy_plain.npy"))  # not a table
        p2d = joinpath(tmp, "2d.npy")
        write(p2d, npy_bytes("'<f8'", 4, zeros(UInt8, 32); shape = "(2, 2)"))
        @test_throws ErrorException read_npy(p2d)
        pnested = joinpath(tmp, "nested.npy")
        write(pnested, npy_bytes("[('itr', [('loc', '<f8')])]", 1, zeros(UInt8, 8)))
        @test_throws ErrorException read_npy(pnested)
        @test_throws ArgumentError read_npy(one_path)                                # not .npy
    end

    @testset "load_minflux (.mat)" begin
        mat_path = joinpath(tmp, "export.mat")
        write(mat_path, "placeholder")
        @test_throws ErrorException load_minflux(mat_path)               # MAT.jl not loaded yet
        @eval using MAT
        n = 6
        Base.invokelatest() do
            MAT.matwrite(mat_path, Dict(
                "vld" => UInt8[1, 1, 0, 1, 1, 0], "fnl" => UInt8[0, 1, 0, 0, 1, 0],
                "tim" => collect(0.1:0.1:0.6), "tid" => Int32[1, 1, 1, 2, 2, 2],
                "thi" => zeros(Int32, n), "loc" => rand(n, 3), "efo" => rand(n),
                "custom" => collect(1.0:n)))
        end
        d = Base.invokelatest(load_minflux, mat_path)
        t = d.table
        @test d.label == "export" && length(t) == n
        @test t.vld == Bool[1, 1, 0, 1, 1, 0] && t.tid isa Vector{Int} && t.efo isa Vector{Float32}
        @test size(t.loc) == (n, 3) && t.custom == 1.0:n
        @test propertynames(t)[1:3] == (:vld, :fnl, :tim)                 # standard fields first
        @test length(final_localizations(d)) == 2
        @test isempty(d.grids) && d.attributes === nothing
    end

    # Tests on real files. Set MINFLUXIO_TESTDATA to a directory holding the files and an `expected.toml` with one table per file, e.g.
    #
    #   ["measurement.msr"]
    #   rows = 725662      # records in the MINFLUX table
    #   valid = 51016      # records with vld
    #   final = 10440      # valid final localizations
    #   nstacks = 15       # OBF stacks (.msr only)
    #   label = "..."      # measurement label
    #
    # Every key is optional. Neither the data nor expected.toml is in the repository.
    @testset "real data" begin
        expected_path = joinpath(TESTDATA, "expected.toml")
        if isempty(TESTDATA)
            @info "MINFLUXIO_TESTDATA not set; skipping tests on real data"
        elseif !isfile(expected_path)
            @info "no expected.toml in MINFLUXIO_TESTDATA; skipping tests on real data"
        else
            for (name, exp) in TOML.parsefile(expected_path)
                path = joinpath(TESTDATA, name)
                if !isfile(path)
                    @info "real test file not found, skipping: $path"
                    continue
                end
                @testset "$name" begin
                    d = Base.invokelatest(load_minflux, path)
                    t = d.table
                    haskey(exp, "rows") && @test length(t) == exp["rows"]
                    haskey(exp, "valid") && @test count(t.vld) == exp["valid"]
                    haskey(exp, "final") && @test length(final_localizations(d)) == exp["final"]
                    haskey(exp, "label") && @test d.label == exp["label"]
                    if lowercase(splitext(name)[2]) in (".msr", ".obf")
                        f = read_obf(path)
                        haskey(exp, "nstacks") && @test length(f) == exp["nstacks"]
                        @test count(s -> s.kind === :minflux, f.stacks) >= 1
                        for s in f.stacks
                            s.kind === :image && @test size(read_stack(f, s)) == size(s)
                        end
                    end
                end
            end
        end
    end
end
