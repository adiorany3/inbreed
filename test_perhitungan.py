import pandas as pd
import numpy as np

UNKNOWN = {"", " ", "-", "--", "0", "na", "n/a", "nan", "none", "null"}

def clean(v):
    if v is None or str(v).strip().lower() in UNKNOWN:
        return None
    return str(v).strip()

df = pd.read_csv("contoh_dari_gambar.csv", dtype=str, keep_default_na=False)
for c in ["Animal_ID", "Sire_ID", "Dam_ID"]:
    df[c] = df[c].apply(clean)

parents = {r.Animal_ID: (r.Sire_ID, r.Dam_ID) for r in df.itertuples(index=False)}
order, state = [], {}

def visit(a):
    if state.get(a, 0) == 2:
        return
    if state.get(a, 0) == 1:
        raise RuntimeError("Siklus pedigree")
    state[a] = 1
    s, d = parents[a]
    for p in [s, d]:
        if p is not None:
            visit(p)
    state[a] = 2
    order.append(a)

for a in list(parents):
    visit(a)

idx = {a: i for i, a in enumerate(order)}
A = np.zeros((len(order), len(order)))
hasil = {}

for i, a in enumerate(order):
    s, d = parents[a]
    si = idx.get(s) if s else None
    di = idx.get(d) if d else None
    for j in range(i):
        val = 0.0
        if si is not None:
            val += 0.5 * A[si, j]
        if di is not None:
            val += 0.5 * A[di, j]
        A[i, j] = A[j, i] = val
    F = 0.5 * A[si, di] if si is not None and di is not None else 0.0
    A[i, i] = 1 + F
    hasil[a] = F * 100

assert round(hasil["B"], 2) == 25.00, hasil
assert round(hasil["A"], 2) == 37.50, hasil

print("Test berhasil.")
print("B =", hasil["B"], "%")
print("A =", hasil["A"], "%")
