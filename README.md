# MinfluxIO.jl

[![CI](https://github.com/lanceXwq/MinfluxIO.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lanceXwq/MinfluxIO.jl/actions/workflows/CI.yml?query=branch%3Amain)

Read Abberior Imspector data in Julia: MINFLUX localizations embedded in `.msr`
files, Imspector's MINFLUX `.npy` and `.mat` exports, and the image stacks of
`.msr`/`.obf` files. No Imspector, Python or Java needed.

> [!WARNING]
> This package is experimental. Use it at your own risk. Its API may change
> without notice, and the MINFLUX reader relies on an undocumented file format
> that may change between Imspector versions. Check results against Imspector
> before relying on them.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/lanceXwq/MinfluxIO.jl")   # not registered yet
```

## Usage

### MINFLUX data

```julia
using MinfluxIO

d = load_minflux("13_TD_trk_638_561_.msr")
# MinfluxData "261113-162646_13_L75_640lp10_561lp08_pho010010_BG06en_dt900us_disConExp"
#   source: 13_TD_trk_638_561_.msr
#   acquired: 2026-11-13T16:26:46+01:00
#   records: 725662 (51016 valid, 10440 valid final)
#   final localizations per thread (thi): 0 => 7589, 1 => 2851
#   traces (tid) with final localizations: 1435
#   time: 32.51 – 1766.74 s
#   grids: MinfluxGrid("mbm_0", 133 points), MinfluxGrid("mbm_1", 133 points), MinfluxGrid("search_0", 563 points)

t = final_localizations(d)       # valid records of the final iteration (vld .& fnl)
t.loc                            # n × 3 matrix, metres
t.tim, t.tid, t.thi, t.eco       # vectors
t[t.thi .== 1]                   # select rows

using DataFrames
df = DataFrame(t)                # loc is split into loc_x, loc_y, loc_z

beads(d)                         # bead (MBM) grid points: gri, xyz, tim, str
d.attributes                     # measurement metadata (lasers, sequence, grids, ...)
```

Loading a 21 MB `.msr` with 725k records takes about 70 ms. Decompression runs
on several threads when Julia is started with them, e.g. `julia -t auto`.

If a file holds several MINFLUX measurements, list them with
`minflux_measurements(path)` and choose one with
`load_minflux(path; measurement = 2)` or `measurement = "label"`.

Imspector's MINFLUX exports load the same way and give the same
`MinfluxData`/`MinfluxTable` as `.msr` files. `.npy` needs nothing extra. `.mat`
needs [MAT.jl](https://github.com/JuliaIO/MAT.jl), which loads as an optional
package extension:

```julia
d = load_minflux("export.npy")

using MAT
d = load_minflux("export.mat")
```

An export holds only the localization table. Bead grids, grids and metadata are
only available from `.msr` files. `read_npy(path)` reads any 1-D `.npy` array
with its native element types.

### Images

```julia
f = read_obf("13_TD_trk_638_561_.msr")    # header info only, no data loaded
# ObfFile "13_TD_trk_638_561_.msr" (OBF format version 2, 15 stacks)
#   [ 1] Ch1 {38}  66×71×1×1 Int16
#   [ 3] GFP {19}  180×180×10×1 Int16
#   ...
#   [14] (unnamed)  MINFLUX data block

img = read_stack(f, 3)            # or read_stack(f, "GFP {19}")
s = f[3]
s.labels, s.units, pixelsize(s)   # ["ExpControl X", ...], [m, m, ...], [1e-7, ...]
s.tags["imspector"]               # Imspector XML metadata of that stack
f.metadata["ome_xml"]             # OME-XML of the whole file
```

The first array dimension is the first OBF dimension, usually x. Trailing
singleton dimensions are kept; remove them with `dropdims`.

## Data fields

Standard MINFLUX fields get the same Julia types whatever file they came from.
Integer fields become `Int`, so `diff(t.tid)` cannot wrap around. `Float16`
fields become `Float32`, because `Float16` sums overflow at 65504.

The meanings below are those commonly documented for Imspector MINFLUX
exports. Check Abberior's documentation for your software version.

| field | type | meaning |
|---|---|---|
| `vld` | Bool | valid localization |
| `fnl` | Bool | final iteration |
| `bot`, `eot` | Bool | beginning / end of trace |
| `sta` | Int | status code |
| `tim` | Float64 | time stamp (s) |
| `tid` | Int | trace ID |
| `gri` | Int | grid index |
| `thi` | Int | thread index (e.g. one per laser line in multi-colour sequences) |
| `sqi` | Int | sequence index |
| `itr` | Int | iteration |
| `loc` | Float64, n×3 | localization (m) |
| `lnc` | Float64, n×3 | localization without drift correction (m) |
| `eco`, `ecc` | Int | effective photon counts at offset / centre |
| `efo`, `efc` | Float32 | effective count frequency at offset / centre (Hz) |
| `fbg` | Float32 | background frequency estimate |
| `cfr` | Float32 | centre-frequency ratio |
| `dcr` | Float32 | detector channel ratio |

Fields beyond these are kept, converted by the same rules.

## How the file format is read

**Exports.** `.npy` files are read following NumPy's
[format specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html),
and `.mat` files through MAT.jl. No Imspector-written `.npy` file was available
during development. The reader is tested against files written by NumPy itself
with Imspector's MINFLUX field types, including a copy of a real measurement,
which loaded identically to the `.msr` it came from.

**`.msr` files.** These have two layers, and only one of them follows a
published spec.

**Documented: the OBF container.** `.msr` files are OBF files with extra
Imspector data appended. The file header, the stack chain, the stack footer
(units, dimension labels, tag dictionary), the data types and the zlib
compression all follow Abberior's
[file format documentation](https://imspectordocs.readthedocs.io/en/latest/fileformat.html).
The Imspector-specific data after the OBF metadata section (window layout,
hardware configuration) is not read.

**Not documented: the MINFLUX data block.** Imspector 16.3 and later store
MINFLUX data as a stack that is labelled as an unnamed `UInt8` image but holds
a binary block starting with `MFXDTA`. This format is not publicly documented.
Its layout was inferred from example files written by Imspector 2026.7
(MINFLUX build) and may change between Imspector versions:

```
"MFXDTA\0"       magic (7 bytes)
UInt32           version (3 so far)
UInt32           Unix timestamp; matches the end of the acquisition
zero padding     up to the first record
records          until a zero length:
                   UInt64 n, n bytes path ("zarr/mfx/.zarray", ...),
                   UInt8 is_directory, and for files: UInt64 m, m bytes content
```

The records form a [Zarr v2](https://zarr.readthedocs.io/en/stable/spec/v2.html)
store:

- `zarr/mfx` is the localization table, a structured array with zlib-compressed
  chunks.
- `zarr/grd/*` are the search and bead grids, Blosc/LZ4-compressed.
- `zarr/mbm/*` holds the drift-correction settings.
- `.zattrs` files hold the metadata.

The table fields match Imspector's documented `.npy`/`.mat` exports. Lower-level
access is available through `read_obf`, `read_stack_bytes`, `parse_mfxdta`,
`read_zarr_array` and `read_zarr_attrs`.

This reader is unofficial. Other Imspector versions may store the MINFLUX block
differently. If a file fails to load, please open an issue and include the
Imspector version (found in `f[i].tags["imspector"]` or in the error).

## Limitations

- Read-only.
- OBF stacks stored in chunks (format version 6 chunk positions) and RGB stacks
  are not supported yet. An informative error is raised for them.
- Only 1-D Zarr arrays without filters are supported inside MINFLUX blocks. This
  covers everything seen so far.
- `.npy`: only 1-D, little-endian arrays without nested structured or object
  fields.
- Older (pre-2025) Imspector MINFLUX export layouts are not supported.
- Tested on Imspector 2026.7 (MINFLUX build) `.msr` and `.mat` files, and on
  NumPy-written `.npy` files.

## Tests

```julia
using Pkg; Pkg.test("MinfluxIO")
```

The tests build small synthetic OBF/MSR, `.npy` and `.mat` files, so no real
data is needed. They also use a few tiny NumPy-written `.npy` files in
`test/data/`, generated by `test/data/generate_npy_fixtures.py`. To also test on real files, set `MINFLUXIO_TESTDATA` to
a directory holding the files listed in `REAL_FILES` in `test/runtests.jl`.
These files are not part of the repository.

## Benchmarks

[`benchmark/`](benchmark/) has scripts comparing MinfluxIO with a NumPy-based
Python reader, together with the results so far.

## Acknowledgments

MinfluxIO.jl was developed in the
[Ries Lab](https://maxperutzlabs.ac.at/research/research-groups/ries)
(Jonas Ries, Max Perutz Labs, University of Vienna), where the example MINFLUX
data used to develop and test the package were also recorded. These data are
not distributed with the package.

This project is not affiliated with or endorsed by Abberior Instruments.
Abberior and Imspector are names of Abberior Instruments GmbH.

## License

MIT; see [LICENSE](LICENSE).
