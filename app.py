import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# FINAL FIX - KALKULATOR INBREEDING SAPI
# ============================================================
# Input:
#   Animal_ID, Sire_ID, Dam_ID
#
# Output:
#   Hubungan parent A(sire, dam)
#   Koefisien inbreeding F
#   Persentase inbreeding
#   Kondisi inbreeding
#
# Rumus:
#   F_anak = 0.5 x A(sire, dam)
#
# Catatan:
#   Seluruh output display memakai "-" untuk nilai kosong,
#   sehingga tidak ada NaN pada tabel hasil.
# ============================================================


EMPTY = "-"
UNKNOWN_VALUES = {
    "",
    " ",
    "-",
    "--",
    "0",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "tidak diketahui",
    "tidak ada",
    "kosong",
}


def is_unknown(value) -> bool:
    """Menganggap blank, '-', 0, NA, None, NaN sebagai parent tidak diketahui."""
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    text = str(value).strip()
    return text.lower() in UNKNOWN_VALUES


def clean_id(value) -> Optional[str]:
    """ID internal untuk perhitungan. Kosong menjadi None."""
    if is_unknown(value):
        return None

    text = str(value).strip()

    # Antisipasi Excel membaca ID angka sebagai 1.0
    if text.endswith(".0"):
        try:
            num = float(text)
            if num.is_integer():
                text = str(int(num))
        except Exception:
            pass

    return text


def show_value(value) -> str:
    """Nilai display. Kosong menjadi '-'."""
    if is_unknown(value):
        return EMPTY
    return str(value).strip()


def clean_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan seluruh tabel untuk tampilan dan download.
    Object/string kosong menjadi '-'.
    Numeric kosong menjadi 0.
    """
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        else:
            out[col] = out[col].apply(show_value).astype(str)

    return out


def kondisi_inbreeding(percent: float) -> str:
    if percent <= 0:
        return "Tidak inbred"
    if percent < 6.25:
        return "Inbreeding rendah"
    if percent < 12.5:
        return "Inbreeding sedang"
    if percent < 25:
        return "Inbreeding tinggi"
    return "Inbreeding sangat tinggi"


def rekomendasi(percent: float) -> str:
    if percent <= 0:
        return "Aman berdasarkan pedigree: tidak terdeteksi inbreeding."
    if percent < 6.25:
        return "Masih rendah, tetapi tetap perlu monitoring."
    if percent < 12.5:
        return "Perlu perhatian. Hindari pengulangan perkawinan kerabat."
    if percent < 25:
        return "Risiko tinggi. Gunakan pasangan dari garis keturunan berbeda."
    return "Risiko sangat tinggi. Perkawinan kerabat dekat sebaiknya dihindari."


def contoh_dari_gambar() -> pd.DataFrame:
    """
    Interpretasi gambar:
    - I dan P adalah founder.
    - C, D, X berasal dari I x P.
    - B berasal dari D x C.
    - A berasal dari B x C.
    - E berasal dari B x parent tidak diketahui.
    - F berasal dari D x parent tidak diketahui.
    """
    return pd.DataFrame(
        {
            "Animal_ID": ["I", "P", "C", "D", "X", "B", "A", "E", "F"],
            "Sire_ID": ["-", "-", "I", "I", "I", "D", "B", "B", "D"],
            "Dam_ID": ["-", "-", "P", "P", "P", "C", "C", "-", "-"],
        }
    )


def contoh_sapi_lengkap() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Animal_ID": [
                "PEJANTAN_01",
                "INDUK_01",
                "SAPI_C",
                "SAPI_D",
                "SAPI_X",
                "SAPI_B",
                "SAPI_A",
                "SAPI_E",
                "SAPI_F",
            ],
            "Sire_ID": [
                "-",
                "-",
                "PEJANTAN_01",
                "PEJANTAN_01",
                "PEJANTAN_01",
                "SAPI_D",
                "SAPI_B",
                "SAPI_B",
                "SAPI_D",
            ],
            "Dam_ID": [
                "-",
                "-",
                "INDUK_01",
                "INDUK_01",
                "INDUK_01",
                "SAPI_C",
                "SAPI_C",
                "-",
                "-",
            ],
        }
    )


def standardize_input(raw_df: pd.DataFrame, id_col: str, sire_col: str, dam_col: str) -> pd.DataFrame:
    """Membuat dataframe internal yang aman untuk perhitungan."""
    if len({id_col, sire_col, dam_col}) != 3:
        raise ValueError("Kolom Animal_ID, Sire_ID, dan Dam_ID harus berbeda.")

    df = raw_df[[id_col, sire_col, dam_col]].copy()
    df.columns = ["Animal_ID", "Sire_ID", "Dam_ID"]

    for col in ["Animal_ID", "Sire_ID", "Dam_ID"]:
        df[col] = df[col].apply(clean_id)

    df = df.dropna(subset=["Animal_ID"]).copy()

    if df.empty:
        raise ValueError("Tidak ada Animal_ID yang valid.")

    duplicates = df.loc[df["Animal_ID"].duplicated(), "Animal_ID"].tolist()
    if duplicates:
        raise ValueError(f"Ada Animal_ID duplikat: {duplicates[:10]}")

    # Cek self-parent. Ini pasti membuat pedigree tidak valid.
    self_parent = df[
        ((df["Sire_ID"].notna()) & (df["Animal_ID"] == df["Sire_ID"]))
        | ((df["Dam_ID"].notna()) & (df["Animal_ID"] == df["Dam_ID"]))
    ]
    if not self_parent.empty:
        ids = self_parent["Animal_ID"].tolist()
        raise ValueError(f"Ada sapi yang menjadi parent untuk dirinya sendiri: {ids[:10]}")

    return df.reset_index(drop=True)


def add_founders(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Parent yang tidak punya baris sendiri ditambahkan sebagai founder."""
    animal_ids = set(df["Animal_ID"].dropna())
    parent_ids = set(df["Sire_ID"].dropna()).union(set(df["Dam_ID"].dropna()))
    missing = sorted(parent_ids - animal_ids)

    if not missing:
        return df.copy(), []

    founder_df = pd.DataFrame(
        {
            "Animal_ID": missing,
            "Sire_ID": [None] * len(missing),
            "Dam_ID": [None] * len(missing),
        }
    )

    return pd.concat([founder_df, df], ignore_index=True), missing


def sort_parent_before_child(df: pd.DataFrame) -> List[str]:
    """Topological sort agar parent selalu dihitung sebelum anak."""
    parents: Dict[str, Tuple[Optional[str], Optional[str]]] = {
        row.Animal_ID: (row.Sire_ID, row.Dam_ID)
        for row in df.itertuples(index=False)
    }

    order: List[str] = []
    state: Dict[str, int] = {}

    def visit(animal_id: str, path: List[str]):
        status = state.get(animal_id, 0)

        if status == 1:
            path_text = " -> ".join(path + [animal_id])
            raise ValueError(f"Terdeteksi siklus pedigree: {path_text}")

        if status == 2:
            return

        state[animal_id] = 1
        sire, dam = parents[animal_id]

        for parent_id in [sire, dam]:
            if parent_id is not None:
                if parent_id not in parents:
                    parents[parent_id] = (None, None)
                visit(parent_id, path + [animal_id])

        state[animal_id] = 2
        order.append(animal_id)

    for animal_id in list(parents.keys()):
        visit(animal_id, [])

    return order


def calculate(df_input: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Menghitung inbreeding.
    Return:
      standardized_df, result_df, matrix_df
    """
    df, missing_founders = add_founders(df_input)
    order = sort_parent_before_child(df)

    parent_map = {
        row.Animal_ID: (row.Sire_ID, row.Dam_ID)
        for row in df.itertuples(index=False)
    }

    n = len(order)
    if n > 5000:
        raise ValueError("Data terlalu besar untuk matriks penuh. Gunakan data lebih kecil.")

    idx = {animal_id: i for i, animal_id in enumerate(order)}
    A = np.zeros((n, n), dtype=float)
    rows = []

    for i, animal_id in enumerate(order):
        sire, dam = parent_map[animal_id]
        sire_idx = idx.get(sire) if sire is not None else None
        dam_idx = idx.get(dam) if dam is not None else None

        # A[i,j] = 0.5(A[sire,j] + A[dam,j])
        for j in range(i):
            val = 0.0
            if sire_idx is not None:
                val += 0.5 * A[sire_idx, j]
            if dam_idx is not None:
                val += 0.5 * A[dam_idx, j]

            A[i, j] = val
            A[j, i] = val

        # F_i = 0.5 x A[sire,dam]
        if sire_idx is not None and dam_idx is not None:
            parent_relation = float(A[sire_idx, dam_idx])
            F = 0.5 * parent_relation
        else:
            parent_relation = 0.0
            F = 0.0

        A[i, i] = 1.0 + F
        percent = F * 100

        if sire is None and dam is None:
            proses = "Founder / parent tidak diketahui: F = 0%."
        elif sire is None or dam is None:
            proses = "Salah satu parent tidak diketahui: F = 0%."
        else:
            proses = (
                f"F {animal_id} = 0,5 × A({sire},{dam}) = "
                f"0,5 × {parent_relation:.6f} = {F:.6f} = {percent:.2f}%."
            )

        same_parent_note = ""
        if sire is not None and dam is not None and sire == dam:
            same_parent_note = "Sire dan dam sama, cek ulang data."

        rows.append(
            {
                "Animal_ID": animal_id,
                "Sire_ID": show_value(sire),
                "Dam_ID": show_value(dam),
                "Hubungan_Parent_A": round(parent_relation, 6),
                "Koefisien_Inbreeding_F": round(F, 6),
                "Inbreeding_%": round(percent, 4),
                "Kondisi_Inbreeding": kondisi_inbreeding(percent),
                "Rekomendasi": rekomendasi(percent),
                "Proses_Perhitungan": proses,
                "Tipe_Data": "Founder tambahan" if animal_id in missing_founders else "Data input",
                "Catatan": same_parent_note,
            }
        )

    result = pd.DataFrame(rows)
    matrix = pd.DataFrame(A, index=order, columns=order)

    standardized_display = clean_display(df_input)

    # Garansi final: tidak ada NaN di result display
    result = clean_display(result)

    return standardized_display, result, matrix


def read_file(uploaded_file) -> pd.DataFrame:
    """Baca CSV/Excel tanpa mengubah blank menjadi NaN."""
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    return pd.read_excel(uploaded_file, dtype=str, keep_default_na=False)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return clean_display(df).to_csv(index=False, na_rep=EMPTY).encode("utf-8-sig")


def to_excel_bytes(result: pd.DataFrame, matrix: Optional[pd.DataFrame] = None) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        clean_display(result).to_excel(writer, index=False, sheet_name="Hasil_Inbreeding")
        if matrix is not None:
            matrix.to_excel(writer, sheet_name="Matriks_A")
    buffer.seek(0)
    return buffer.getvalue()


def dot_escape(value) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def make_dot(result_df: pd.DataFrame, max_nodes: int = 150) -> str:
    df = clean_display(result_df).head(max_nodes)
    animal_set = set(df["Animal_ID"].astype(str))

    dot = [
        "digraph Pedigree {",
        "rankdir=LR;",
        'graph [bgcolor="transparent", splines=true, overlap=false];',
        'node [shape=box, style="rounded,filled", color="#333333", fontname="Arial"];',
        'edge [color="#555555", fontname="Arial", fontsize=9];',
    ]

    for _, row in df.iterrows():
        animal = dot_escape(row["Animal_ID"])
        f_pct = float(row["Inbreeding_%"])
        kondisi = dot_escape(row["Kondisi_Inbreeding"])

        if f_pct >= 25:
            fill = "#FFE4E1"
        elif f_pct > 0:
            fill = "#FFF4CC"
        else:
            fill = "#FFFFFF"

        label = f"{animal}\\nF={f_pct:.2f}%\\n{kondisi}"
        dot.append(f'"{animal}" [label="{label}", fillcolor="{fill}"];')

    for _, row in df.iterrows():
        animal = dot_escape(row["Animal_ID"])
        sire = row["Sire_ID"]
        dam = row["Dam_ID"]

        if sire in animal_set:
            dot.append(f'"{dot_escape(sire)}" -> "{animal}" [label="sire"];')
        if dam in animal_set:
            dot.append(f'"{dot_escape(dam)}" -> "{animal}" [label="dam"];')

    dot.append("}")
    return "\n".join(dot)


def explanation_image_example():
    st.markdown(
        """
        ### Contoh perhitungan dari gambar

        Data gambar dibaca sebagai:

        ```text
        I dan P = founder
        C = I × P
        D = I × P
        X = I × P
        B = D × C
        A = B × C
        E = B × unknown
        F = D × unknown
        ```

        Karena **C** dan **D** sama-sama anak dari **I × P**, maka **C dan D adalah saudara kandung penuh**.

        **B = D × C**

        Jadi:

        ```text
        A(D,C) = 0,50
        F_B = 0,5 × A(D,C)
        F_B = 0,5 × 0,50
        F_B = 0,25 = 25%
        ```

        Untuk **A**:

        ```text
        A = B × C
        A(B,C) = 0,75
        F_A = 0,5 × 0,75
        F_A = 0,375 = 37,5%
        ```
        """
    )


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Kalkulator Inbreeding Sapi",
    page_icon="🐄",
    layout="wide",
)

st.title("🐄 Kalkulator Inbreeding Sapi - Final Fix")
st.caption("Menghitung koefisien inbreeding, kondisi inbreeding, grafik, dan bagan pedigree.")

with st.expander("Format data", expanded=True):
    st.markdown(
        """
        Gunakan 3 kolom berikut:

        | Animal_ID | Sire_ID | Dam_ID |
        |---|---|---|
        | I | - | - |
        | P | - | - |
        | C | I | P |
        | D | I | P |
        | B | D | C |

        Gunakan tanda **`-`** untuk parent yang tidak diketahui.
        """
    )

mode = st.radio(
    "Pilih sumber data",
    ["Contoh dari gambar", "Contoh sapi lengkap", "Unggah file sendiri"],
    horizontal=True,
)

if mode == "Contoh dari gambar":
    raw_df = contoh_dari_gambar()
    explanation_image_example()
elif mode == "Contoh sapi lengkap":
    raw_df = contoh_sapi_lengkap()
else:
    uploaded = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])
    if uploaded is None:
        st.info("Unggah file terlebih dahulu, atau gunakan contoh dari gambar.")
        st.dataframe(clean_display(contoh_dari_gambar()), use_container_width=True)
        st.stop()
    raw_df = read_file(uploaded)

st.subheader("1. Preview data")
st.dataframe(clean_display(raw_df), use_container_width=True)

columns = list(raw_df.columns)

if len(columns) < 3:
    st.error("File harus memiliki minimal 3 kolom.")
    st.stop()

default_id = columns.index("Animal_ID") if "Animal_ID" in columns else 0
default_sire = columns.index("Sire_ID") if "Sire_ID" in columns else min(1, len(columns) - 1)
default_dam = columns.index("Dam_ID") if "Dam_ID" in columns else min(2, len(columns) - 1)

st.subheader("2. Pilih kolom")
col1, col2, col3 = st.columns(3)
with col1:
    id_col = st.selectbox("Kolom Animal_ID", columns, index=default_id)
with col2:
    sire_col = st.selectbox("Kolom Sire_ID / pejantan", columns, index=default_sire)
with col3:
    dam_col = st.selectbox("Kolom Dam_ID / induk", columns, index=default_dam)

try:
    internal_df = standardize_input(raw_df, id_col, sire_col, dam_col)
    standardized_df, result_df, matrix_df = calculate(internal_df)
except Exception as error:
    st.error(f"Perhitungan gagal: {error}")
    st.stop()

result_input_only = result_df[result_df["Tipe_Data"] == "Data input"].copy()
result_input_only["Inbreeding_%"] = pd.to_numeric(result_input_only["Inbreeding_%"], errors="coerce").fillna(0)

st.subheader("3. Ringkasan")
total = len(result_input_only)
inbred_count = int((result_input_only["Inbreeding_%"] > 0).sum())
avg_f = float(result_input_only["Inbreeding_%"].mean()) if total else 0
max_f = float(result_input_only["Inbreeding_%"].max()) if total else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total sapi", total)
m2.metric("Sapi inbred", inbred_count)
m3.metric("Rata-rata F", f"{avg_f:.2f}%")
m4.metric("F tertinggi", f"{max_f:.2f}%")

if inbred_count > 0:
    st.success("Perhitungan berhasil. Kondisi inbreeding terdeteksi pada data ini.")
else:
    st.info("Perhitungan berhasil. Tidak ada inbreeding terdeteksi dari pedigree yang tersedia.")

st.subheader("4. Hasil perhitungan inbreeding")
st.dataframe(clean_display(result_input_only), use_container_width=True)

st.subheader("5. Grafik inbreeding")
chart = result_input_only[["Animal_ID", "Inbreeding_%"]].copy()
st.bar_chart(chart.set_index("Animal_ID"))

st.subheader("6. Bagan pedigree")
max_nodes = st.slider("Maksimal node pada bagan", 20, 300, 150, 10)
st.graphviz_chart(make_dot(result_df, max_nodes), use_container_width=True)

st.subheader("7. Proses perhitungan")
process_cols = [
    "Animal_ID",
    "Sire_ID",
    "Dam_ID",
    "Hubungan_Parent_A",
    "Koefisien_Inbreeding_F",
    "Inbreeding_%",
    "Kondisi_Inbreeding",
    "Proses_Perhitungan",
]
st.dataframe(clean_display(result_df[process_cols]), use_container_width=True)

show_matrix = st.checkbox("Tampilkan matriks hubungan A", value=False)
if show_matrix:
    st.subheader("8. Matriks hubungan A")
    st.dataframe(matrix_df.round(6), use_container_width=True)

st.subheader("9. Download")
d1, d2, d3 = st.columns(3)

with d1:
    st.download_button(
        "Download data standar CSV",
        data=to_csv_bytes(standardized_df),
        file_name="pedigree_standar.csv",
        mime="text/csv",
    )

with d2:
    st.download_button(
        "Download hasil CSV",
        data=to_csv_bytes(result_df),
        file_name="hasil_inbreeding_sapi.csv",
        mime="text/csv",
    )

with d3:
    st.download_button(
        "Download hasil Excel",
        data=to_excel_bytes(result_df, matrix_df if show_matrix else None),
        file_name="hasil_inbreeding_sapi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
