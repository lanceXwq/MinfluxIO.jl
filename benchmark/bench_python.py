"""Reference Python/NumPy reader for the benchmark (needs NumPy).

    python benchmark/bench_python.py path/to/file.msr [repetitions]

Does the same decoding as MinfluxIO's `decode table` step: walks the OBF stack chain, finds the MINFLUX ("MFXDTA") block, unpacks the embedded Zarr store and decodes the `zarr/mfx` record array into a NumPy structured array. It does not split the records into columns, convert types, or read grids and metadata.
"""
import json
import statistics
import struct
import sys
import time
import zlib

import numpy as np


def read_msr_minflux(path):
    with open(path, "rb") as fh:
        d = fh.read()
    _, pos, _ = struct.unpack_from("<IQI", d, 10)
    blob = None
    while pos:                                    # walk the OBF stack chain
        q = pos + 16 + 8 + 300
        dt, ct, _, nl, dl, _, dlen_disk, nxt = struct.unpack_from("<IIIIIQQQ", d, q)
        q += 44 + nl + dl
        if dt == 1:                               # uint8 stack: check for MFXDTA
            raw = zlib.decompress(d[q:q + dlen_disk]) if ct else d[q:q + dlen_disk]
            if raw[:7] == b"MFXDTA\0":
                blob = raw
        pos = nxt
    p = blob.find(b"\x04\x00\x00\x00\x00\x00\x00\x00zarr")
    files = {}
    while p < len(blob) - 8:                      # unpack the serialized Zarr store
        n, = struct.unpack_from("<Q", blob, p)
        if n == 0:
            break
        name = blob[p + 8:p + 8 + n].decode()
        p += 8 + n
        isdir = blob[p]
        p += 1
        if not isdir:
            size, = struct.unpack_from("<Q", blob, p)
            p += 8
            files[name] = blob[p:p + size]
            p += size
    meta = json.loads(files["zarr/mfx/.zarray"])
    dtype = np.dtype([tuple(f) if len(f) == 2 else (f[0], f[1], tuple(f[2])) for f in meta["dtype"]])
    n, chunk = meta["shape"][0], meta["chunks"][0]
    out = np.zeros(n, dtype)
    for i in range((n + chunk - 1) // chunk):     # decode the zlib chunks
        key = f"zarr/mfx/{i}"
        if key in files:
            a = np.frombuffer(zlib.decompress(files[key]), dtype)
            k = min(chunk, n - i * chunk)
            out[i * chunk:i * chunk + k] = a[:k]
    return out


if __name__ == "__main__":
    path = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    t0 = time.perf_counter()
    a = read_msr_minflux(path)
    first = time.perf_counter() - t0
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        read_msr_minflux(path)
        times.append(time.perf_counter() - t0)
    print(f"records: {len(a)}, valid final: {int(np.sum(a['vld'] & a['fnl']))}")
    print(f"first {first * 1e3:.1f} ms, median {statistics.median(times) * 1e3:.1f} ms, "
          f"min {min(times) * 1e3:.1f} ms ({reps} repetitions)")
