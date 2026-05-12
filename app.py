import io
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# APLIKASI PERHITUNGAN KOEFISIEN INBREEDING TERNAK
# Metode: Tabular Method / Numerator Relationship Matrix
#
# Rumus utama:
# F_anak = 0.5 x A[sire, dam]
#
# Keterangan:
# F_anak      = koefisien inbreeding anak
# A[sire,dam] = hubungan aditif antara pejantan dan induk
#
# Jika salah satu/kedua parent tidak diketahui, F dianggap 0.
# =========================================================


UNKNOWN_VALUES = {
    "",
    "0",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "-",
    "--",
    "unknown",
    "tidak ada",
    "kosong",
}


def normalize_id(value) -> Optional[str]:
    """Membersihkan ID ternak/parent. Nilai kosong dikembalikan sebagai None."""
    if pd.isna(value):
        return None

    text = str(value).strip()

    # Jika Excel membaca 1 sebagai 1.0, ubah menjadi "1"
    if text.endswith(".0"):
        try:
            if float(text).is_integer():
                text = str(int(float(text)))
        except ValueError:
            pass

    if text.lower() in UNKNOWN_VALUES:
        return None

    return text


def clean_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Agar hasil tidak menampilkan NaN/None, tetapi '-'."""
    display_df = df.copy()

    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            continue
        display_df[col] = display_df[col].where(display_df[col].notna(), "-")
        display_df[col] = display_df[col].replace({None: "-", "None": "-", "nan": "-", "NaN": ""})

    return display_df


def classify_inbreeding(percent: float) -> str:
    """Klasifikasi sederhana untuk mempermudah interpretasi."""
    if percent <= 0:
        return "Tidak terdeteksi"
    if percent < 6.25:
        return "Rendah"
    if percent < 12.5:
        return "Sedang"
    if percent < 25:
        return "Tinggi"
    return "Sangat tinggi"


def make_input_dataframe(
    raw_df: pd.DataFrame,
    id_col: str,
    sire_col: str,
    dam_col: str,
) -> pd.DataFrame:
    """Mengubah data input menjadi format standar: Animal_ID, Sire_ID, Dam_ID."""
    df = raw_df[[id_col, sire_col, dam_col]].copy()
    df.columns = ["Animal_ID", "Sire_ID", "Dam_ID"]

    for col in ["Animal_ID", "Sire_ID", "Dam_ID"]:
        df[col] = df[col].apply(normalize_id)

    df = df.dropna(subset=["Animal_ID"]).copy()

    if df.empty:
        raise ValueError("Tidak ada Animal_ID yang valid. Mohon cek kolom ID ternak.")

    if df["Animal_ID"].duplicated().any():
        duplicated = df.loc[df["Animal_ID"].duplicated(), "Animal_ID"].tolist()
        raise ValueError(
            "Ada ID ternak yang duplikat. Setiap Animal_ID harus unik. "
            f"Contoh duplikat: {duplicated[:10]}"
        )

    return df


def add_missing_parents_as_founders(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Jika parent muncul di Sire_ID/Dam_ID tetapi tidak ada sebagai Animal_ID,
    parent tersebut ditambahkan sebagai founder dengan parent kosong.
    """
    animal_ids = set(df["Animal_ID"])
    parent_ids = set(df["Sire_ID"].dropna()).union(set(df["Dam_ID"].dropna()))
    missing_parents = sorted(parent_ids - animal_ids)

    if missing_parents:
        founders = pd.DataFrame(
            {
                "Animal_ID": missing_parents,
                "Sire_ID": [None] * len(missing_parents),
                "Dam_ID": [None] * len(missing_parents),
            }
        )
        df = pd.concat([founders, df], ignore_index=True)

    return df, missing_parents


def topological_order(df: pd.DataFrame) -> List[str]:
    """
    Mengurutkan ternak agar parent selalu dihitung sebelum anak.
    Jika ada siklus pedigree, fungsi akan memberi error.
    """
    parents: Dict[str, Tuple[Optional[str], Optional[str]]] = {
        row.Animal_ID: (row.Sire_ID, row.Dam_ID)
        for row in df.itertuples(index=False)
    }

    order: List[str] = []
    state: Dict[str, int] = {}

    def visit(animal_id: str, stack: List[str]):
        current_state = state.get(animal_id, 0)

        if current_state == 1:
            cycle_path = " -> ".join(stack + [animal_id])
            raise ValueError(f"Terdeteksi siklus/loop pedigree yang tidak valid: {cycle_path}")

        if current_state == 2:
            return

        state[animal_id] = 1
        sire, dam = parents.get(animal_id, (None, None))

        for parent in [sire, dam]:
            if parent is not None:
                if parent not in parents:
                    parents[parent] = (None, None)
                visit(parent, stack + [animal_id])

        state[animal_id] = 2
        order.append(animal_id)

    for animal_id in list(parents.keys()):
        visit(animal_id, [])

    return order


def calculate_inbreeding(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menghitung koefisien inbreeding memakai numerator relationship matrix.

    Output:
    - result_df: tabel hasil per ternak
    - relationship_matrix_df: matriks hubungan aditif A
    """
    df, missing_parents = add_missing_parents_as_founders(df)
    order = topological_order(df)

    parents = {
        row.Animal_ID: (row.Sire_ID, row.Dam_ID)
        for row in df.itertuples(index=False)
    }

    n = len(order)

    if n > 5000:
        raise ValueError(
            "Pedigree terlalu besar untuk versi matriks penuh ini (> 5000 individu setelah founder ditambahkan). "
            "Gunakan versi algoritma sparse/khusus untuk dataset sangat besar."
        )

    index = {animal_id: i for i, animal_id in enumerate(order)}
    A = np.zeros((n, n), dtype=float)

    rows = []

    for i, animal_id in enumerate(order):
        sire, dam = parents[animal_id]
        sire_idx = index.get(sire) if sire is not None else None
        dam_idx = index.get(dam) if dam is not None else None

        # Hubungan individu i dengan semua individu sebelumnya
        for j in range(i):
            val = 0.0
            if sire_idx is not None:
                val += 0.5 * A[sire_idx, j]
            if dam_idx is not None:
                val += 0.5 * A[dam_idx, j]

            A[i, j] = val
            A[j, i] = val

        # Inbreeding anak berasal dari hubungan antara sire dan dam
        if sire_idx is not None and dam_idx is not None:
            parent_relationship = float(A[sire_idx, dam_idx])
            F = 0.5 * parent_relationship
        else:
            parent_relationship = 0.0
            F = 0.0

        A[i, i] = 1.0 + F
        F_percent = F * 100

        if sire is None and dam is None:
            proses = "Founder / parent tidak diketahui, sehingga F = 0%"
        elif sire is None or dam is None:
            proses = "Salah satu parent tidak diketahui, sehingga F = 0%"
        else:
            proses = (
                f"F = 0.5 x A({sire},{dam}) = "
                f"0.5 x {parent_relationship:.6f} = {F:.6f} = {F_percent:.2f}%"
            )

        rows.append(
            {
                "Animal_ID": animal_id,
                "Sire_ID": sire,
                "Dam_ID": dam,
                "Hubungan_Parent_A": parent_relationship,
                "Koefisien_Inbreeding_F": F,
                "Inbreeding_%": F_percent,
                "Kategori": classify_inbreeding(F_percent),
                "Proses_Perhitungan": proses,
                "Tipe_Data": "Founder tambahan"
                if animal_id in missing_parents
                else "Data input",
                "Catatan": "Parent sama, cek ulang data"
                if sire is not None and dam is not None and sire == dam
                else "",
            }
        )

    result_df = pd.DataFrame(rows)

    # Urutkan sesuai urutan perhitungan pedigree.
    result_df["_urutan_pedigree"] = result_df["Animal_ID"].map(index)
    result_df = result_df.sort_values("_urutan_pedigree").drop(columns="_urutan_pedigree").reset_index(drop=True)

    relationship_matrix_df = pd.DataFrame(A, index=order, columns=order)

    return result_df, relationship_matrix_df


def safe_dot_text(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return "-"
    return str(text).replace('"', '\\"')


def make_pedigree_dot(result_df: pd.DataFrame, max_nodes: int = 120) -> str:
    """
    Membuat bagan pedigree dalam format DOT untuk st.graphviz_chart.
    Untuk menjaga performa, tampilkan maksimal max_nodes baris pertama.
    """
    df = result_df.head(max_nodes).copy()

    dot = [
        "digraph Pedigree {",
        "rankdir=LR;",
        'graph [splines=true, overlap=false, bgcolor="transparent"];',
        'node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#444444", fontname="Arial"];',
        'edge [color="#555555", fontname="Arial", fontsize=10];',
    ]

    for _, row in df.iterrows():
        animal = safe_dot_text(row["Animal_ID"])
        label = f"{animal}\\nF={float(row['Inbreeding_%']):.2f}%"
        dot.append(f'"{animal}" [label="{label}"];')

    animals_in_graph = set(df["Animal_ID"].astype(str))

    for _, row in df.iterrows():
        animal = safe_dot_text(row["Animal_ID"])
        sire = row["Sire_ID"]
        dam = row["Dam_ID"]

        if sire is not None and str(sire) in animals_in_graph:
            dot.append(f'"{safe_dot_text(sire)}" -> "{animal}" [label="sire"];')

        if dam is not None and str(dam) in animals_in_graph:
            dot.append(f'"{safe_dot_text(dam)}" -> "{animal}" [label="dam"];')

    dot.append("}")
    return "\n".join(dot)


def sample_pedigree() -> pd.DataFrame:
    """
    Contoh sederhana:
    - P1 dan P2 adalah founder.
    - A1 dan A2 adalah full sibling karena parent-nya sama, P1 x P2.
    - B1 lahir dari A1 x A2, sehingga terjadi perkawinan saudara kandung.
      Hubungan A1 dan A2 = 0.50, maka F_B1 = 0.5 x 0.50 = 0.25 = 25%.
    """
    return pd.DataFrame(
        {
            "Animal_ID": ["P1", "P2", "A1", "A2", "B1", "B2", "C1"],
            "Sire_ID": [None, None, "P1", "P1", "A1", "A1", "B1"],
            "Dam_ID": [None, None, "P2", "P2", "A2", "P2", "B2"],
        }
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    export_df = clean_for_display(df)
    return export_df.to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(
    result_df: pd.DataFrame,
    matrix_df: Optional[pd.DataFrame] = None,
) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        clean_for_display(result_df).to_excel(writer, index=False, sheet_name="Hasil_Inbreeding")
        if matrix_df is not None:
            matrix_df.to_excel(writer, sheet_name="Matriks_A")
    buffer.seek(0)
    return buffer.getvalue()


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def run_calculation_ui(raw_df: pd.DataFrame, source_label: str):
    st.subheader("1. Preview Data")
    st.caption(source_label)
    st.dataframe(clean_for_display(raw_df.head(100)), use_container_width=True)

    st.subheader("2. Pilih Kolom Pedigree")
    columns = list(raw_df.columns)

    if len(columns) < 3:
        st.error("Data minimal harus memiliki 3 kolom: Animal_ID, Sire_ID, Dam_ID.")
        st.stop()

    default_id = "Animal_ID" if "Animal_ID" in columns else columns[0]
    default_sire = "Sire_ID" if "Sire_ID" in columns else columns[min(1, len(columns) - 1)]
    default_dam = "Dam_ID" if "Dam_ID" in columns else columns[min(2, len(columns) - 1)]

    col1, col2, col3 = st.columns(3)
    with col1:
        id_col = st.selectbox("Kolom ID ternak", columns, index=columns.index(default_id))
    with col2:
        sire_col = st.selectbox("Kolom pejantan/bapak", columns, index=columns.index(default_sire))
    with col3:
        dam_col = st.selectbox("Kolom betina/induk", columns, index=columns.index(default_dam))

    show_matrix = st.checkbox(
        "Tampilkan dan ekspor matriks hubungan aditif A",
        value=False,
        help="Aktifkan untuk dataset kecil/sedang. Matriks penuh bisa besar untuk banyak ternak.",
    )

    max_nodes = st.slider(
        "Jumlah maksimal ternak yang ditampilkan pada bagan pedigree",
        min_value=20,
        max_value=300,
        value=120,
        step=20,
    )

    if st.button("🔢 Hitung Koefisien Inbreeding", type="primary"):
        try:
            clean_df = make_input_dataframe(raw_df, id_col, sire_col, dam_col)
            result_df, matrix_df = calculate_inbreeding(clean_df)

            input_only = result_df[result_df["Tipe_Data"] == "Data input"].copy()
            founder_added = result_df[result_df["Tipe_Data"] == "Founder tambahan"].copy()

            st.subheader("3. Ringkasan Hasil")

            avg_inbreeding = input_only["Inbreeding_%"].mean() if not input_only.empty else 0
            max_inbreeding = input_only["Inbreeding_%"].max() if not input_only.empty else 0
            n_inbred = int((input_only["Inbreeding_%"] > 0).sum())
            total_animals = len(input_only)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total ternak input", f"{total_animals:,}")
            m2.metric("Ternak inbred", f"{n_inbred:,}")
            m3.metric("Rata-rata F", f"{avg_inbreeding:.2f}%")
            m4.metric("F tertinggi", f"{max_inbreeding:.2f}%")

            if not founder_added.empty:
                st.warning(
                    f"Ada {len(founder_added)} parent yang tidak ditemukan sebagai Animal_ID, "
                    "sehingga ditambahkan sebagai founder dengan F = 0%."
                )

            same_parent = input_only[input_only["Catatan"].str.contains("Parent sama", na=False)]
            if not same_parent.empty:
                st.warning(
                    "Ada baris dengan Sire_ID dan Dam_ID yang sama. "
                    "Mohon cek ulang karena ini biasanya tidak valid pada data ternak biparental."
                )
                st.dataframe(
                    clean_for_display(same_parent[["Animal_ID", "Sire_ID", "Dam_ID", "Catatan"]]),
                    use_container_width=True,
                )

            st.subheader("4. Tabel Hasil Koefisien Inbreeding")
            display_df = input_only.copy()
            display_df["Hubungan_Parent_A"] = display_df["Hubungan_Parent_A"].round(6)
            display_df["Koefisien_Inbreeding_F"] = display_df["Koefisien_Inbreeding_F"].round(6)
            display_df["Inbreeding_%"] = display_df["Inbreeding_%"].round(4)

            st.dataframe(clean_for_display(display_df), use_container_width=True)

            st.subheader("5. Bagan Pedigree")
            st.caption(
                "Panah menunjukkan arah dari parent ke anak. Label node menampilkan ID ternak dan nilai F."
            )
            dot = make_pedigree_dot(result_df, max_nodes=max_nodes)
            st.graphviz_chart(dot, use_container_width=True)

            st.subheader("6. Grafik 20 Ternak dengan Inbreeding Tertinggi")
            top_df = input_only.sort_values("Inbreeding_%", ascending=False).head(20)
            if not top_df.empty and top_df["Inbreeding_%"].max() > 0:
                chart_df = top_df.set_index("Animal_ID")[["Inbreeding_%"]]
                st.bar_chart(chart_df)
            else:
                st.info("Tidak ada nilai inbreeding > 0 pada data input.")

            st.subheader("7. Contoh Proses Perhitungan")
            st.markdown(
                """
                Rumus dasar yang digunakan:

                **F anak = 0,5 × hubungan antara pejantan dan induk**

                Contoh dari data demo:

                - **P1** dan **P2** adalah founder, sehingga F = 0%.
                - **A1** dan **A2** sama-sama anak dari P1 × P2, jadi A1 dan A2 adalah saudara kandung penuh.
                - Hubungan aditif antara **A1** dan **A2** adalah **0,50**.
                - **B1** lahir dari A1 × A2.
                - Maka **F B1 = 0,5 × 0,50 = 0,25 = 25%**.
                """
            )

            proses_cols = [
                "Animal_ID",
                "Sire_ID",
                "Dam_ID",
                "Hubungan_Parent_A",
                "Koefisien_Inbreeding_F",
                "Inbreeding_%",
                "Proses_Perhitungan",
            ]
            proses_df = display_df[proses_cols].copy()
            st.dataframe(clean_for_display(proses_df), use_container_width=True)

            if show_matrix:
                st.subheader("8. Matriks Hubungan A")
                st.dataframe(matrix_df.round(6), use_container_width=True)

            st.subheader("9. Unduh Hasil")
            col_a, col_b = st.columns(2)

            with col_a:
                st.download_button(
                    "⬇️ Unduh hasil CSV",
                    data=to_csv_bytes(result_df),
                    file_name="hasil_inbreeding_pedigree.csv",
                    mime="text/csv",
                )

            with col_b:
                st.download_button(
                    "⬇️ Unduh hasil Excel",
                    data=to_excel_bytes(result_df, matrix_df if show_matrix else None),
                    file_name="hasil_inbreeding_pedigree.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.error(str(e))
            st.stop()


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="Kalkulator Inbreeding Pedigree Ternak",
    page_icon="🐄",
    layout="wide",
)

st.title("🐄 Kalkulator Koefisien Inbreeding Ternak")
st.caption(
    "Menghitung persentase koefisien inbreeding dari data pedigree ternak, "
    "menampilkan tabel hasil, proses perhitungan, grafik, dan bagan pedigree."
)

with st.expander("Format data yang dibutuhkan", expanded=True):
    st.markdown(
        """
        Siapkan file **CSV** atau **Excel (.xlsx)** dengan minimal 3 kolom:

        | Kolom | Keterangan |
        |---|---|
        | **Animal_ID** | ID ternak/individu |
        | **Sire_ID** | ID pejantan/bapak |
        | **Dam_ID** | ID betina/induk |

        Parent yang tidak diketahui boleh dikosongkan atau diisi: `0`, `NA`, `N/A`, `-`, `None`.

        Aplikasi sudah diperbaiki agar data kosong tidak tampil sebagai `NaN`, melainkan sebagai `-`.
        Dalam perhitungan, nilai kosong tetap dianggap sebagai parent tidak diketahui/founder.
        """
    )

template_df = sample_pedigree()

st.download_button(
    label="⬇️ Unduh template CSV + contoh inbreeding",
    data=to_csv_bytes(template_df),
    file_name="template_pedigree_inbreeding.csv",
    mime="text/csv",
)

st.markdown("---")

mode = st.radio(
    "Pilih sumber data",
    ["Gunakan contoh bawaan", "Unggah file sendiri"],
    horizontal=True,
)

if mode == "Gunakan contoh bawaan":
    raw_df = template_df.copy()
    run_calculation_ui(raw_df, "Data contoh bawaan: menggambarkan perkawinan saudara kandung dan parent-offspring.")
else:
    uploaded_file = st.file_uploader(
        "Unggah file pedigree CSV atau Excel",
        type=["csv", "xlsx"],
    )

    if uploaded_file is None:
        st.info("Unggah file pedigree terlebih dahulu, atau pilih mode 'Gunakan contoh bawaan'.")
        st.dataframe(clean_for_display(template_df), use_container_width=True)
        st.stop()

    try:
        raw_df = read_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"File tidak dapat dibaca: {e}")
        st.stop()

    if raw_df.empty:
        st.error("File kosong. Mohon unggah file dengan data pedigree.")
        st.stop()

    run_calculation_ui(raw_df, f"File yang digunakan: {uploaded_file.name}")
