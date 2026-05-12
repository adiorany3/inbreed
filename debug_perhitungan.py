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
print("Parents Map:", parents)

order = []
state = {}
def visit(a):
    if a is None or a not in parents: return
    if state.get(a, 0) == 2: return
    state[a] = 1
    s, d = parents[a]
    visit(s); visit(d)
    state[a] = 2
    order.append(a)

for a in parents: visit(a)
print("Order:", order)

idx = {a: i for i, a in enumerate(order)}
n = len(order)
A = np.zeros((n, n))
for i, a in enumerate(order):
    s, d = parents[a]
    si, di = idx.get(s), idx.get(d)
    for j in range(i):
        val = 0.0
        if si is not None: val += 0.5 * A[si, j]
        if di is not None: val += 0.5 * A[di, j]
        A[i, j] = A[j, i] = val
    F = 0.5 * A[si, di] if si is not None and di is not None else 0.0
    A[i, i] = 1.0 + F
    print(f"Animal {a}: F={F*100}%, A[{i},{i}]={A[i,i]}")

