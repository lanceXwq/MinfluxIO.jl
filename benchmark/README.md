# Benchmarks

These scripts measure how fast MinfluxIO loads MINFLUX data from an `.msr` file and compare it with a NumPy-based Python reader that decodes the same data.

## Results

**Machine and software:** Apple M1 Pro (8 performance + 2 efficiency cores), 16 GiB RAM, macOS 27.2. Julia 1.13.4; Python 3.15.2 with NumPy 2.5.5.

**Test file:** a 21 MB `.msr` recorded in the Ries Lab and written by Imspector 2026.7 (MINFLUX build). It holds 725,662 MINFLUX records, 10,440 of them valid final localizations, plus 14 image stacks. The file is not part of this repository.

| | Julia, 1 thread | Julia, 4 threads | Python + NumPy |
|---|---|---|---|
| Decode the table (median / min) | 73.9 / 69.9 ms | 59.4 / 52.9 ms | 65.5 / 64.8 ms |
| Full `load_minflux` (median / min) | 75.0 / 71.2 ms | 55.8 / 54.4 ms | — |
| Memory allocated per full load | 242 MiB | 242 MiB | — |
| `using MinfluxIO` | 171 ms | | |
| First `load_minflux` in a session | 275 ms | | |
| Fresh process: start, load once, exit | 0.57–0.67 s | | 0.15 s |

The two "decode" rows are not quite the same work:

- **Python** reads the file, unpacks the MINFLUX block and decompresses the table into a NumPy structured array. NumPy keeps the records as stored, so the decode ends there.
- **Julia** does the same and then splits the records into one column per field (about 21 ms). Julia has no equivalent of NumPy's structured arrays for Imspector's packed records, and separate columns make later analysis fast.
- **Full `load_minflux`** also converts columns to the package's standard types and reads the grids and metadata. Together these add only 1–2 ms.

Where the time goes, single-threaded:

| Stage | Julia | Python |
|---|---|---|
| Decompress the 9 MB MINFLUX block | 26 ms | 25 ms |
| Decompress the 73 table chunks | 28 ms | 35 ms |
| Split records into columns | 21 ms | not done |

## History

The first version of the package (commit `10bee5f`) took 203 ms to decode the table and allocated 638 MiB per load. Its first call in a session took 3.2 s. Four changes brought it to the numbers above:

| Commit | Change | Effect |
|---|---|---|
| `760e039` | Decompress with libdeflate into preallocated buffers | table chunks 115 → 28 ms, MINFLUX block 48 → 26 ms |
| `a078f84` | Decode Zarr chunks on several threads | full load 75 ms (1 thread) → 56 ms (4 threads) |
| `d4495d5` | Read fields directly from records, in cache-sized blocks | column split 53 → 21 ms |
| `bb6be7c` | Precompile workload | first call 3.2 s → 0.28 s |

## Running the benchmarks

Julia, from the repository root (run each in a fresh process, because the first measurement is the first call in the session):

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/bench.jl path/to/file.msr
julia -t 4 --project=benchmark benchmark/bench.jl path/to/file.msr
```

Python (needs NumPy):

```sh
python benchmark/bench_python.py path/to/file.msr 20
```
