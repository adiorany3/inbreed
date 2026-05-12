import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# SOFTWARE KALKULATOR INBREEDING SAPI
# Input  : Pedigree Animal_ID, Sire_ID, Dam_ID
# Output : Koefisien inbreeding F, persentase, kondisi, bagan
#
# Rumus utama:
# F_individu = 0.5 x A(Sire, Dam)
#
# A(Sire, Dam) = hubungan kekerabatan aditif antar parent.
# ============================================================


DISPLAY_EMPTY = "-"
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
    """Cek nilai parent kosong/tidak diketahui."""
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    text = str(value).strip()
    return text.lower() in UNKNOWN_VALUES


def normalize_id(value) -> Optional[str]:
    """
    Membersihkan ID.
    Nilai kosong, '-', 0, NA, None, dan sejenisnya dianggap parent tidak diketahui.
    """
    if is_unknown(value):
        return None

    text = str(value).strip()

    # Jika Excel membaca kode 1 sebagai 1.0, ubah kembali menjadi 1.
    if text.endswith(".0"):
        try:
            number = float(text)
            if number.is_integer():
                text = str(int(number))
        except Exception:
            pass

    return text


def display_value(value) -> str:
    """Ubah nilai kosong/internal None menjadi '-' agar tidak tampil sebagai nilai kosong."""
    if is_unknown(value):
        return DISPLAY_EMPTY
    return str(value).strip()


def clean_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Versi final untuk ditampilkan/diunduh.
    Semua kolom object dipaksa menjadi string bersih.
    Semua nilai kosong diganti '-'.
    """
    clean = df.copy()

    for col in clean.columns:
        if pd.api.types.is_numeric_dtype(clean[col]):
            clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0)
        else:
            clean[col] = clean[col].apply(display_value).astype(str)

    return clean


def kondisi_inbreeding(percent: float) -> str:
    """Kondisi inbreeding berdasarkan persentase F."""
    if percent <= 0:
        return "Tidak inbred"
    if percent < 6.25:
        return "Inbreeding rendah"
    if percent < 12.5:
        return "Inbreeding sedang"
    if percent < 25:
        return "Inbreeding tinggi"
    return "Inbreeding sangat tinggi"


def rekomendasi_kondisi(percent: float) -> str:
    if percent <= 0:
        return "Aman: tidak terdeteksi inbreeding dari pedigree."
    if percent < 6.25:
        return "Masih rendah, tetapi tetap perlu dikontrol."
    if percent < 12.5:
        return "Perlu perhatian, hindari perkawinan kerabat dekat berikutnya."
    if percent < 25:
        return "Risiko tinggi, sebaiknya gunakan pejantan/induk dari garis berbeda."
    return "Risiko sangat tinggi, perkawinan sebaiknya dihindari untuk program pemuliaan."


def make_input_dataframe(raw_df: pd.DataFrame, id_col: str, sire_col: str, dam_col: str) -> pd.DataFrame:
    """Standarisasi data input menjadi Animal_ID, Sire_ID, Dam_ID."""
    df = raw_df[[id_col, sire_col, dam_col]].copy()
    df.columns = ["Animal_ID", "Sire_ID", "Dam_ID"]

    for col in ["Animal_ID", "Sire_ID", "Dam_ID"]:
        df[col] = df[col].apply(normalize_id)

    df = df.dropna(subset=["Animal_ID"]).copy()

    if df.empty:
        raise ValueError("Tidak ada Animal_ID yang valid. Periksa kembali kolom ID sapi.")

    duplicated = df[df["Animal_ID"].duplicated()]["Animal_ID"].tolist()
    if duplicated:
        raise ValueError(
            "Ada Animal_ID yang duplikat. Setiap sapi harus memiliki ID unik. "
            f"Contoh duplikat: {duplicated[:10]}"
        )

    return df


def add_missing_parents_as_founders(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Jika parent muncul di Sire_ID/Dam_ID tetapi tidak ada sebagai Animal_ID,
    parent ditambahkan sebagai founder.
    """
    animal_ids = set(df["Animal_ID"].dropna())
    parent_ids = set(df["Sire_ID"].dropna()).union(set(df["Dam_ID"].dropna()))
    missing_parents = sorted(parent_ids - animal_ids)

    if missing_parents:
        founder_df = pd.DataFrame(
            {
                "Animal_ID": missing_parents,
                "Sire_ID": [None] * len(missing_parents),
                "Dam_ID": [None] * len(missing_parents),
            }
        )
        df = pd.concat([founder_df, df], ignore_index=True)

    return df, missing_parents


def topological_order(df: pd.DataFrame) -> List[str]:
    """
    Mengurutkan pedigree agar parent selalu dihitung sebelum anak.
    """
    parents: Dict[str, Tuple[Optional[str], Optional[str]]] = {
        row.Animal_ID: (row.Sire_ID, row.Dam_ID)
        for row in df.itertuples(index=False)
    }

    order: List[str] = []
    state: Dict[str, int] = {}

    def visit(animal_id: str, path: List[str]):
        current = state.get(animal_id, 0)

        if current == 1:
            raise ValueError("Terdeteksi siklus/loop pedigree: " + " -> ".join(path + [animal_id]))

        if current == 2:
            return

        state[animal_id] = 1

        sire, dam = parents.get(animal_id, (None, None))
        for parent in [sire, dam]:
            if parent is not None:
                if parent not in parents:
                    parents[parent] = (None, None)
                visit(parent, path + [animal_id])

        state[animal_id] = 2
        order.append(animal_id)

    for animal_id in list(parents.keys()):
        visit(animal_id, [])

    return order


def calculate_inbreeding(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menghitung inbreeding dengan numerator relationship matrix.
    """
    df, missing_parents = add_missing_parents_as_founders(df)
    order = topological_order(df)

    parents = {
        row.Animal_ID: (row.Sire_ID, row.Dam_ID)
        for row in df.itertuples(index=False)
    }

    n = len(order)
    if n > 5000:
        raise ValueError("Jumlah individu terlalu besar untuk matriks penuh. Gunakan dataset lebih kecil atau algoritma sparse.")

    index = {animal_id: idx for idx, animal_id in enumerate(order)}
    A = np.zeros((n, n), dtype=float)

    rows = []

    for i, animal_id in enumerate(order):
        sire, dam = parents[animal_id]

        sire_idx = index.get(sire) if sire is not None else None
        dam_idx = index.get(dam) if dam is not None else None

        # Hubungan individu i dengan individu sebelumnya
        for j in range(i):
            value = 0.0
            if sire_idx is not None:
                value += 0.5 * A[sire_idx, j]
            if dam_idx is not None:
                value += 0.5 * A[dam_idx, j]

            A[i, j] = value
            A[j, i] = value

        # Koefisien inbreeding
        if sire_idx is not None and dam_idx is not None:
            hubungan_parent = float(A[sire_idx, dam_idx])
            F = 0.5 * hubungan_parent
        else:
            hubungan_parent = 0.0
            F = 0.0

        A[i, i] = 1.0 + F
        percent = F * 100

        if sire is None and dam is None:
            proses = "Founder: parent tidak diketahui, F = 0%."
        elif sire is None or dam is None:
            proses = "Salah satu parent tidak diketahui, F = 0%."
        else:
            proses = (
                f"F {animal_id} = 0,5 × A({sire},{dam}) = "
                f"0,5 × {hubungan_parent:.6f} = {F:.6f} = {percent:.2f}%."
            )

        rows.append(
            {
                "Animal_ID": animal_id,
                "Sire_ID": sire,
                "Dam_ID": dam,
                "Hubungan_Parent_A": hubungan_parent,
                "Koefisien_Inbreeding_F": F,
                "Inbreeding_%": percent,
                "Kondisi_Inbreeding": kondisi_inbreeding(percent),
                "Rekomendasi": rekomendasi_kondisi(percent),
                "Proses_Perhitungan": proses,
                "Tipe_Data": "Founder tambahan" if animal_id in missing_parents else "Data input",
                "Catatan": "Periksa: Sire_ID dan Dam_ID sama"
                if sire is not None and dam is not None and sire == dam
                else "",
            }
        )

    result_df = pd.DataFrame(rows)
    result_df["_order"] = result_df["Animal_ID"].map(index)
    result_df = result_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    matrix_df = pd.DataFrame(A, index=order, columns=order)

    return result_df, matrix_df


def sample_from_image() -> pd.DataFrame:
    """
    Contoh berdasarkan gambar user.
    Interpretasi:
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


def sample_cow_named() -> pd.DataFrame:
    """Contoh sapi dengan nama lebih realistis."""
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


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    dtype=str dan keep_default_na=False mencegah sel kosong otomatis berubah menjadi nilai kosong internal.
    """
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    return pd.read_excel(uploaded_file, dtype=str, keep_default_na=False)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    export = clean_for_display(df)
    text = export.to_csv(index=False, na_rep=DISPLAY_EMPTY)
    return text.encode("utf-8-sig")


def to_excel_bytes(result_df: pd.DataFrame, matrix_df: Optional[pd.DataFrame] = None) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        clean_for_display(result_df).to_excel(writer, index=False, sheet_name="Hasil_Inbreeding")
        if matrix_df is not None:
            matrix_df.to_excel(writer, sheet_name="Matriks_A")
    output.seek(0)
    return output.getvalue()


def make_graphviz(result_df: pd.DataFrame, max_nodes: int = 150) -> str:
    display = clean_for_display(result_df.head(max_nodes))
    dot = [
        "digraph Pedigree {",
        "rankdir=LR;",
        'graph [bgcolor="transparent", splines=true, overlap=false];',
        'node [shape=box, style="rounded,filled", fillcolor="white", color="#333333", fontname="Arial"];',
        'edge [color="#555555", fontname="Arial", fontsize=9];',
    ]

    animals = set(display["Animal_ID"].astype(str).tolist())

    for _, row in display.iterrows():
        animal = row["Animal_ID"]
        f_value = float(row["Inbreeding_%"])
        kondisi = row["Kondisi_Inbreeding"]

        if f_value >= 25:
            fill = "#FFE4E1"
        elif f_value > 0:
            fill = "#FFF4D6"
        else:
            fill = "white"

        label = f"{animal}\\nF={f_value:.2f}%\\n{kondisi}"
        dot.append(f'"{animal}" [label="{label}", fillcolor="{fill}"];')

    for _, row in display.iterrows():
        animal = row["Animal_ID"]
        sire = row["Sire_ID"]
        dam = row["Dam_ID"]

        if sire in animals:
            dot.append(f'"{sire}" -> "{animal}" [label="sire"];')
        if dam in animals:
            dot.append(f'"{dam}" -> "{animal}" [label="dam"];')

    dot.append("}")
    return "\n".join(dot)


def explain_example_from_image():
    st.markdown(
        """
        ### Gambaran perhitungan dari gambar

        Interpretasi gambar yang digunakan:

        - **I** dan **P** dianggap sebagai founder.
        - **C**, **D**, dan **X** berasal dari pasangan **I × P**.
        - **B** berasal dari **D × C**.
        - **A** berasal dari **B × C**.
        - **E** berasal dari **B × parent tidak diketahui**.
        - **F** berasal dari **D × parent tidak diketahui**.

        Karena **C** dan **D** sama-sama anak dari **I × P**, maka C dan D adalah saudara kandung penuh.

        **B = D × C**

        Maka:

        **F B = 0,5 × A(D,C) = 0,5 × 0,50 = 0,25 = 25%**

        Setelah itu, **A = B × C**. Karena B adalah anak dari C dan D, maka hubungan B dengan C lebih tinggi.

        **F A = 0,5 × A(B,C) = 0,5 × 0,75 = 0,375 = 37,5%**
        """
    )


def validate_no_display_empty_issue(df: pd.DataFrame) -> bool:
    """Pastikan tabel tampilan tidak memiliki nilai kosong internal."""
    display = clean_for_display(df)
    return not display.isna().any().any()


def run_app(raw_df: pd.DataFrame, source_label: str):
    st.subheader("1. Data Pedigree")
    st.caption(source_label)
    st.dataframe(clean_for_display(raw_df), use_container_width=True)

    columns = list(raw_df.columns)

    if len(columns) < 3:
        st.error("Data harus memiliki minimal 3 kolom: Animal_ID, Sire_ID, Dam_ID.")
        st.stop()

    default_id = columns.index("Animal_ID") if "Animal_ID" in columns else 0
    default_sire = columns.index("Sire_ID") if "Sire_ID" in columns else min(1, len(columns) - 1)
    default_dam = columns.index("Dam_ID") if "Dam_ID" in columns else min(2, len(columns) - 1)

    st.subheader("2. Kolom Pedigree")
    c1, c2, c3 = st.columns(3)
    with c1:
        id_col = st.selectbox("Kolom ID sapi", columns, index=default_id)
    with c2:
        sire_col = st.selectbox("Kolom sire / pejantan", columns, index=default_sire)
    with c3:
        dam_col = st.selectbox("Kolom dam / induk", columns, index=default_dam)

    show_matrix = st.checkbox("Tampilkan matriks hubungan A", value=False)

    # Perhitungan dibuat otomatis agar user langsung melihat kondisi inbreeding.
    try:
        clean_input = make_input_dataframe(raw_df, id_col, sire_col, dam_col)
        result_df, matrix_df = calculate_inbreeding(clean_input)
    except Exception as err:
        st.error(f"Perhitungan gagal: {err}")
        st.stop()

    display_result = clean_for_display(result_df)
    input_only = result_df[result_df["Tipe_Data"] == "Data input"].copy()
    display_input_only = clean_for_display(input_only)

    st.subheader("3. Ringkasan Kondisi Inbreeding")

    total = len(input_only)
    n_inbred = int((input_only["Inbreeding_%"] > 0).sum()) if total else 0
    avg_f = float(input_only["Inbreeding_%"].mean()) if total else 0.0
    max_f = float(input_only["Inbreeding_%"].max()) if total else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total sapi", f"{total}")
    m2.metric("Sapi inbred", f"{n_inbred}")
    m3.metric("Rata-rata F", f"{avg_f:.2f}%")
    m4.metric("F tertinggi", f"{max_f:.2f}%")

    if validate_no_display_empty_issue(result_df):
        st.success("Data berhasil dihitung. Nilai parent kosong ditampilkan sebagai '-' dan kondisi inbreeding sudah dihitung.")
    else:
        st.warning("Ada nilai kosong yang perlu diperiksa.")

    st.subheader("4. Hasil Perhitungan")
    rounded = display_input_only.copy()
    for col in ["Hubungan_Parent_A", "Koefisien_Inbreeding_F", "Inbreeding_%"]:
        if col in rounded.columns:
            rounded[col] = pd.to_numeric(rounded[col], errors="coerce").fillna(0).round(6 if col != "Inbreeding_%" else 4)
    st.dataframe(rounded, use_container_width=True)

    st.subheader("5. Grafik Nilai Inbreeding")
    chart_df = input_only[["Animal_ID", "Inbreeding_%"]].copy()
    if not chart_df.empty:
        st.bar_chart(chart_df.set_index("Animal_ID"))
    else:
        st.info("Tidak ada data yang dapat dibuat grafik.")

    st.subheader("6. Bagan Pedigree")
    max_nodes = st.slider("Jumlah maksimal node pada bagan", 20, 300, 150, 10)
    st.graphviz_chart(make_graphviz(result_df, max_nodes=max_nodes), use_container_width=True)

    st.subheader("7. Proses Perhitungan")
    proses_cols = [
        "Animal_ID",
        "Sire_ID",
        "Dam_ID",
        "Hubungan_Parent_A",
        "Koefisien_Inbreeding_F",
        "Inbreeding_%",
        "Kondisi_Inbreeding",
        "Proses_Perhitungan",
    ]
    proses_display = clean_for_display(result_df[proses_cols])
    st.dataframe(proses_display, use_container_width=True)

    if show_matrix:
        st.subheader("8. Matriks Hubungan A")
        st.dataframe(matrix_df.round(6), use_container_width=True)

    st.subheader("9. Unduh Hasil")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Unduh input CSV bersih",
            data=to_csv_bytes(clean_input),
            file_name="pedigree_input_bersih.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "Unduh hasil CSV",
            data=to_csv_bytes(result_df),
            file_name="hasil_inbreeding_sapi.csv",
            mime="text/csv",
        )
    with d3:
        st.download_button(
            "Unduh hasil Excel",
            data=to_excel_bytes(result_df, matrix_df if show_matrix else None),
            file_name="hasil_inbreeding_sapi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Kalkulator Inbreeding Sapi",
    page_icon="🐄",
    layout="wide",
)

st.title("🐄 Kalkulator Inbreeding Sapi")
st.caption("Software untuk menghitung koefisien inbreeding, kondisi inbreeding, grafik, dan bagan pedigree.")

with st.expander("Format CSV / Excel", expanded=True):
    st.markdown(
        """
        Format minimal:

        | Animal_ID | Sire_ID | Dam_ID |
        |---|---|---|
        | I | - | - |
        | P | - | - |
        | C | I | P |
        | D | I | P |
        | B | D | C |

        Gunakan tanda **`-`** untuk parent yang tidak diketahui.
        Aplikasi tetap dapat membaca sel kosong, `0`, `NA`, `N/A`, atau `None` sebagai parent tidak diketahui.
        """
    )

mode = st.radio(
    "Pilih data",
    [
        "Contoh dari gambar",
        "Contoh sapi dengan nama lengkap",
        "Unggah file sendiri",
    ],
    horizontal=True,
)

if mode == "Contoh dari gambar":
    raw_data = sample_from_image()
    explain_example_from_image()
    run_app(raw_data, "Data contoh ini dibuat dari gambar pedigree yang diberikan.")
elif mode == "Contoh sapi dengan nama lengkap":
    raw_data = sample_cow_named()
    run_app(raw_data, "Data contoh sapi dengan nama lengkap.")
else:
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])

    if uploaded_file is None:
        st.info("Silakan unggah file, atau pilih contoh dari gambar.")
        st.dataframe(clean_for_display(sample_from_image()), use_container_width=True)
        st.stop()

    raw_data = read_uploaded_file(uploaded_file)
    run_app(raw_data, f"File diunggah: {uploaded_file.name}")
