"""Generate the small NumPy-written .npy test fixtures used by test/runtests.jl.

Run with any Python that has NumPy:  python generate_npy_fixtures.py
Values follow the formulas in `mfx_records` (test/synthetic.jl), with i = 1..n.
"""
import os
import numpy as np

N = 25
FIELDS = [("vld", "?"), ("fnl", "?"), ("tim", "<f8"), ("tid", "<u4"), ("thi", "u1"),
          ("itr", "<i4"), ("loc", "<f8", (3,)), ("cfr", "<f2"), ("dcr", "<f2", (1,))]


def records(dtype):
    a = np.zeros(N, dtype=dtype)
    i = np.arange(1, N + 1)
    a["vld"] = i % 2 == 1
    a["fnl"] = i % 3 == 0
    a["tim"] = 0.5 * i
    a["tid"] = i // 4
    a["thi"] = i % 2
    a["itr"] = i % 5
    for j in (1, 2, 3):
        a["loc"][:, j - 1] = 1e-9 * (100 * i + j)
    a["cfr"] = (0.25 * (i % 4)).astype(np.float16)
    a["dcr"][:, 0] = (0.5 * (i % 2)).astype(np.float16)
    return a


here = os.path.dirname(os.path.abspath(__file__))
packed = records(np.dtype(FIELDS))
aligned = records(np.dtype(FIELDS, align=True))   # has padding bytes between fields
with open(os.path.join(here, "numpy_mfx_v1.npy"), "wb") as f:
    np.lib.format.write_array(f, packed, version=(1, 0))
with open(os.path.join(here, "numpy_mfx_aligned_v3.npy"), "wb") as f:
    np.lib.format.write_array(f, aligned, version=(3, 0))
np.save(os.path.join(here, "numpy_plain.npy"), np.arange(5, dtype="<i8"))
print("written; aligned itemsize", aligned.dtype.itemsize, "packed itemsize", packed.dtype.itemsize)
