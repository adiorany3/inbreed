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

df = df.dropna(subset=["Animal_ID"])

parents = {r.Animal_ID: (r.Sire_ID, r.Dam_ID) for r in df.itertuples(index=False)}
order, state = [], {}

def visit(a):
    if a is None:
        return
    if state.get(a, 0) == 2:
        return
    if state.get(a, 0) == 1:
        raise RuntimeError(f"Siklus pedigree di {a}")
    state[a] = 1
    s, d = parents.get(a, (None, None))
    for p in [s, d]:
        visit(p)
    state[a] = 2
    order.append(a)

for a in list(parents):
    if a:
        visit(a)

valid_order = [a for a in order if a]
idx = {a: i for i, a in enumerate(valid_order)}
n = len(valid_order)
A = np.zeros((n, n))
hasil = {}

for i, a in enumerate(valid_order):
    s, d = parents.get(a, (None, None))
    si = idx.get(s) if s else None
    di = idx.get(d) if d else None
    
    for j in range(i):
        val = 0.0
        if si is not None:
            val += 0.5 * A[si, j]
        if di is not None:
            val += 0.5 * A[di, j]
        A[i, j] = A[j, i] = val
        
    if si is not None and di is not None:
        F = 0.5 * A[si, di]
    else:
        F = 0.0
        
    A[i, i] = 1.0 + F
    hasil[a] = F * 100

print("Hasil Perhitungan:")
for k, v in hasil.items():
    print(f"{k}: {v:.2f}%")

assert round(hasil['I'], 2) == 0.0
assert round(hasil['B'], 2) == 25.0
assert round(hasil['A'], 2) == 37.5

print("\nTest berhasil.")
