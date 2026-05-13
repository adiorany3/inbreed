import pandas as pd
import numpy as np

def clean_id(value):
    UNKNOWN_VALUES = {"-", "none", "nan", "", None}
    if value is None: return None
    try:
        if pd.isna(value): return None
    except: pass
    text = str(value).strip()
    if text.lower() in UNKNOWN_VALUES: return None
    return text

def calculate_debug(df_input):
    # Standardize
    df_input = df_input.copy()
    for col in ["Animal_ID", "Sire_ID", "Dam_ID"]:
        df_input[col] = df_input[col].apply(clean_id)
    
    # Add founders
    animal_ids = set(df_input["Animal_ID"].dropna())
    parent_ids = set(df_input["Sire_ID"].dropna()).union(set(df_input["Dam_ID"].dropna()))
    missing = sorted(parent_ids - animal_ids)
    founder_df = pd.DataFrame({"Animal_ID": missing, "Sire_ID": [None]*len(missing), "Dam_ID": [None]*len(missing)})
    df_full = pd.concat([founder_df, df_input], ignore_index=True)
    
    # Parents map - EXTREMELY CAREFUL HERE
    parents_map = {}
    for row in df_full.itertuples(index=False):
        if row.Animal_ID:
            s_val = None if (isinstance(row.Sire_ID, float) and np.isnan(row.Sire_ID)) else row.Sire_ID
            d_val = None if (isinstance(row.Dam_ID, float) and np.isnan(row.Dam_ID)) else row.Dam_ID
            parents_map[str(row.Animal_ID)] = (s_val, d_val)
    
    print("Parents Map Head:", list(parents_map.items())[:5])
    
    # Sort
    order = []
    state = {}
    def visit(a):
        if a is None: return
        status = state.get(a, 0)
        if status == 2: return
        state[a] = 1
        s, d = parents_map.get(a, (None, None))
        for p in [s, d]:
            if p is not None: visit(str(p))
        state[a] = 2
        order.append(a)
    for a in list(parents_map.keys()): visit(a)
    
    # Matrix
    idx = {a: i for i, a in enumerate(order)}
    n = len(order)
    A = np.zeros((n, n))
    for i, a in enumerate(order):
        s, d = parents_map.get(a, (None, None))
        # Ensure lookup keys are same type
        si = idx.get(str(s)) if s is not None else None
        di = idx.get(str(d)) if d is not None else None
        
        for j in range(i):
            val = 0.0
            if si is not None: val += 0.5 * A[si, j]
            if di is not None: val += 0.5 * A[di, j]
            A[i, j] = A[j, i] = val
            
        F = 0.5 * A[si, di] if (si is not None and di is not None) else 0.0
        A[i, i] = 1.0 + F
        print(f"[{a}] S={s} (idx={si}), D={d} (idx={di}), F={F}")
    
    return {a: round((A[idx[a], idx[a]]-1)*100, 2) for a in order if a in ["I","B"]}

df = pd.DataFrame({
    "Animal_ID": ["I", "P", "C", "D", "X", "B", "A", "E", "F"],
    "Sire_ID": ["-", "-", "I", "I", "I", "D", "B", "B", "D"],
    "Dam_ID": ["-", "-", "P", "P", "P", "C", "C", "-", "-"]
})
print("RESULT:", calculate_debug(df))
