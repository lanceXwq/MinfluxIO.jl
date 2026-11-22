# Benchmark MinfluxIO on an .msr file.
#
#   julia --project=benchmark benchmark/bench.jl path/to/file.msr
#   julia -t 4 --project=benchmark benchmark/bench.jl path/to/file.msr
#
# Run it in a fresh process: the first measurement is the first call in the session (including any compilation left after precompilation).

t_using = @elapsed using MinfluxIO
using BenchmarkTools

length(ARGS) == 1 || error("usage: julia --project=benchmark benchmark/bench.jl file.msr")
const P = ARGS[1]

# OBF -> MINFLUX block -> `zarr/mfx` table, split into columns with the stored types
function decode_table(path)
    f = read_obf(path)
    s = only(filter(s -> s.kind === :minflux, f.stacks))
    return read_zarr_array(parse_mfxdta(read_stack_bytes(f, s)), "zarr/mfx")
end

t_first = @elapsed d = load_minflux(P)
println("threads: ", Threads.nthreads())
println("using MinfluxIO: ", round(Int, t_using * 1e3), " ms")
println("first load_minflux (incl. compilation): ", round(Int, t_first * 1e3), " ms")
println("records: ", length(d.table), ", valid final: ", length(final_localizations(d)))

for (name, fn) in (("decode table + column split", decode_table), ("load_minflux (full)", load_minflux))
    fn(P)
    b = @benchmark $fn($P) samples = 20 evals = 1 seconds = 60
    println(rpad(name, 30), "median ", round(median(b).time / 1e6, digits = 1), " ms, min ",
        round(minimum(b).time / 1e6, digits = 1), " ms, allocated ",
        round(Int, median(b).memory / 2^20), " MiB")
end
