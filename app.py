import io
import pathlib
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# ============================================================
# BREEDING & INBREEDING ANALYTICS - APP.PY
# Informative English Version
# ============================================================

EMPTY = "-"

UNKNOWN_VALUES = {
    "", " ", "-", "--", "0", "na", "n/a", "nan", "none", "null",
    "unknown", "empty",
}


def is_unknown(value) -> bool:
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
    if is_unknown(value):
        return None

    text = str(value).strip()

    # Fix numeric IDs from Excel, for example 101.0 becomes 101
    if text.endswith(".0"):
        try:
            num = float(text)
            if num.is_integer():
                text = str(int(num))
        except Exception:
            pass

    return text


def show_value(value) -> str:
    if is_unknown(value):
        return EMPTY
    return str(value).strip()


def clean_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        else:
            out[col] = out[col].astype(str).apply(
                lambda x: EMPTY if is_unknown(x) else x
            )

    return out


def kondisi_inbreeding(percent: float) -> str:
    if percent <= 0:
        return "Not inbred"
    if percent < 6.25:
        return "Low inbreeding"
    if percent < 12.5:
        return "Moderate inbreeding"
    if percent < 25:
        return "High inbreeding"
    return "Very high inbreeding"


def rekomendasi(percent: float) -> str:
    if percent <= 0:
        return "Safe based on pedigree: no inbreeding detected."
    if percent < 6.25:
        return "Low risk, but pedigree monitoring is still recommended."
    if percent < 12.5:
        return "Requires attention. Avoid repeated mating between related animals."
    if percent < 25:
        return "High risk. Use mates from unrelated lineages."
    return "Very high risk. Avoid close-relative mating and do not prioritize as breeding stock."


def dampak_inbreeding(percent: float) -> str:
    if percent <= 0:
        return "Impact: Genetic variation is maintained. No inbreeding depression risk was detected from the available pedigree."
    if percent < 6.25:
        return "Impact: Minimal short-term influence, but cumulative effects may appear if related mating continues across generations."
    if percent < 12.5:
        return "Impact: Possible reduction in production performance, fertility, growth, milk yield, and higher risk of recessive genetic problems."
    if percent < 25:
        return "Impact: Inbreeding depression may become visible through weaker immunity, lower fertility, slow growth, and possible birth defects."
    return "Critical impact: Very high risk of embryo loss, infertility, low vitality, congenital defects, and long-term decline in genetic quality."


def contoh_sapi_lengkap() -> pd.DataFrame:
    """
    Ideal full cattle example for demonstrating all breeding management modules.

    This sample includes:
    - pedigree
    - sex
    - age
    - line
    - generation
    - reproductive status
    - health/defect status
    - genomic information
    - economic information
    - pen/cage/batch/location
    - multiple performance traits
    """
    return pd.DataFrame({
        "Animal_ID": [
            # Founder / G0
            "BULL_A0", "COW_A0", "BULL_B0", "COW_B0",
            "BULL_C0", "COW_C0", "BULL_D0", "COW_D0",

            # GGPS / G1
            "BULL_A1", "COW_A1", "BULL_B1", "COW_B1",
            "BULL_C1", "COW_C1", "BULL_D1", "COW_D1",

            # GPS / G2
            "BULL_A2", "COW_A2", "BULL_B2", "COW_B2",
            "BULL_C2", "COW_C2", "BULL_D2", "COW_D2",

            # PS / G3 controlled line crosses
            "BULL_AB_PS", "COW_AB_PS", "BULL_CD_PS", "COW_CD_PS",

            # FS / G4 terminal commercial animals
            "FS_001", "FS_002", "FS_003", "FS_004",
        ],

        "Sire_ID": [
            "-", "-", "-", "-",
            "-", "-", "-", "-",

            "BULL_A0", "BULL_A0", "BULL_B0", "BULL_B0",
            "BULL_C0", "BULL_C0", "BULL_D0", "BULL_D0",

            "BULL_A1", "BULL_A1", "BULL_B1", "BULL_B1",
            "BULL_C1", "BULL_C1", "BULL_D1", "BULL_D1",

            "BULL_A2", "BULL_A2", "BULL_C2", "BULL_C2",

            "BULL_AB_PS", "BULL_AB_PS", "BULL_CD_PS", "BULL_CD_PS",
        ],

        "Dam_ID": [
            "-", "-", "-", "-",
            "-", "-", "-", "-",

            "COW_A0", "COW_A0", "COW_B0", "COW_B0",
            "COW_C0", "COW_C0", "COW_D0", "COW_D0",

            "COW_A1", "COW_A1", "COW_B1", "COW_B1",
            "COW_C1", "COW_C1", "COW_D1", "COW_D1",

            "COW_B2", "COW_B2", "COW_D2", "COW_D2",

            "COW_CD_PS", "COW_CD_PS", "COW_AB_PS", "COW_AB_PS",
        ],

        "Sex": [
            "Male", "Female", "Male", "Female",
            "Male", "Female", "Male", "Female",

            "Male", "Female", "Male", "Female",
            "Male", "Female", "Male", "Female",

            "Male", "Female", "Male", "Female",
            "Male", "Female", "Male", "Female",

            "Male", "Female", "Male", "Female",

            "Male", "Female", "Male", "Female",
        ],

        "Age": [
            84, 84, 82, 82, 80, 80, 78, 78,
            60, 60, 58, 58, 56, 56, 54, 54,
            36, 36, 35, 35, 34, 34, 33, 33,
            24, 24, 24, 24,
            12, 12, 12, 12,
        ],

        "Line": [
            "Line A", "Line A", "Line B", "Line B",
            "Line C", "Line C", "Line D", "Line D",

            "Line A", "Line A", "Line B", "Line B",
            "Line C", "Line C", "Line D", "Line D",

            "Line A", "Line A", "Line B", "Line B",
            "Line C", "Line C", "Line D", "Line D",

            "Line AB", "Line AB", "Line CD", "Line CD",

            "Line ABCD", "Line ABCD", "Line ABCD", "Line ABCD",
        ],

        "Generation": [
            "Founder", "Founder", "Founder", "Founder",
            "Founder", "Founder", "Founder", "Founder",

            "GGPS", "GGPS", "GGPS", "GGPS",
            "GGPS", "GGPS", "GGPS", "GGPS",

            "GPS", "GPS", "GPS", "GPS",
            "GPS", "GPS", "GPS", "GPS",

            "PS", "PS", "PS", "PS",

            "FS", "FS", "FS", "FS",
        ],

        # Main phenotype used by EBV calculation. Example: adjusted body weight index.
        "Phenotype": [
            560, 520, 555, 515, 550, 510, 548, 508,
            590, 545, 585, 540, 580, 538, 575, 535,
            610, 560, 605, 558, 600, 555, 598, 552,
            630, 590, 625, 585,
            650, 640, 645, 638,
        ],

        # Multi-trait columns
        "Body_Weight": [
            560, 520, 555, 515, 550, 510, 548, 508,
            590, 545, 585, 540, 580, 538, 575, 535,
            610, 560, 605, 558, 600, 555, 598, 552,
            630, 590, 625, 585,
            650, 640, 645, 638,
        ],
        "Growth_Rate": [
            1.18, 1.10, 1.16, 1.08, 1.15, 1.07, 1.14, 1.06,
            1.25, 1.16, 1.23, 1.15, 1.22, 1.14, 1.20, 1.13,
            1.30, 1.20, 1.28, 1.19, 1.27, 1.18, 1.26, 1.17,
            1.34, 1.25, 1.32, 1.24,
            1.38, 1.34, 1.36, 1.33,
        ],
        "Fertility": [
            88, 90, 87, 89, 86, 88, 85, 87,
            91, 92, 90, 91, 89, 90, 88, 89,
            93, 94, 92, 93, 91, 92, 90, 91,
            94, 95, 93, 94,
            0, 0, 0, 0,
        ],
        "Survival_Rate": [
            96, 97, 96, 97, 95, 96, 95, 96,
            97, 98, 97, 98, 96, 97, 96, 97,
            98, 98, 98, 98, 97, 98, 97, 98,
            98, 99, 98, 99,
            99, 99, 99, 99,
        ],
        "Feed_Efficiency": [
            1.70, 1.75, 1.72, 1.76, 1.74, 1.78, 1.75, 1.79,
            1.62, 1.68, 1.64, 1.69, 1.65, 1.70, 1.66, 1.71,
            1.58, 1.64, 1.59, 1.65, 1.60, 1.66, 1.61, 1.67,
            1.55, 1.60, 1.56, 1.61,
            1.52, 1.54, 1.53, 1.55,
        ],

        # Reproductive and health status
        "Reproductive_Status": [
            "Ready", "Ready", "Ready", "Ready", "Ready", "Ready", "Ready", "Ready",
            "Ready", "Ready", "Ready", "Ready", "Ready", "Ready", "Ready", "Ready",
            "Ready", "Ready", "Ready", "Ready", "Ready", "Ready", "Ready", "Ready",
            "Ready", "Ready", "Ready", "Ready",
            "Terminal", "Terminal", "Terminal", "Terminal",
        ],
        "Health_Status": [
            "Healthy", "Healthy", "Healthy", "Healthy", "Healthy", "Healthy", "Healthy", "Healthy",
            "Healthy", "Healthy", "Healthy", "Healthy", "Healthy", "Healthy", "Healthy", "Healthy",
            "Healthy", "Healthy", "Healthy", "Healthy", "Healthy", "Healthy", "Healthy", "Healthy",
            "Healthy", "Healthy", "Healthy", "Healthy",
            "Healthy", "Healthy", "Healthy", "Healthy",
        ],
        "Defect_Status": [
            "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear",
            "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear",
            "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear",
            "Clear", "Clear", "Clear", "Clear",
            "Clear", "Clear", "Clear", "Clear",
        ],
        "Culling_Reason": ["-"] * 32,

        # Genomic support
        "Genomic_EBV": [
            8.2, 6.5, 7.9, 6.2, 7.6, 6.0, 7.4, 5.8,
            10.5, 8.4, 10.1, 8.1, 9.8, 7.9, 9.5, 7.7,
            12.2, 9.6, 11.8, 9.4, 11.5, 9.1, 11.2, 8.9,
            13.6, 11.0, 13.1, 10.7,
            14.5, 13.9, 14.2, 13.7,
        ],
        "Genomic_Reliability": [
            72, 70, 71, 69, 70, 68, 69, 67,
            78, 76, 77, 75, 76, 74, 75, 73,
            84, 82, 83, 81, 82, 80, 81, 79,
            88, 86, 87, 85,
            80, 80, 80, 80,
        ],

        # Economic support
        "Feed_Cost": [
            420, 390, 415, 388, 410, 385, 408, 382,
            450, 420, 445, 418, 440, 415, 438, 412,
            480, 445, 475, 442, 470, 438, 468, 435,
            500, 465, 495, 460,
            520, 515, 518, 512,
        ],
        "Production_Value": [
            760, 710, 750, 705, 745, 700, 740, 695,
            830, 780, 820, 775, 815, 770, 810, 765,
            900, 840, 890, 835, 880, 830, 875, 825,
            960, 900, 950, 890,
            1020, 1005, 1015, 1000,
        ],
        "Replacement_Cost": [
            120, 115, 120, 115, 118, 114, 118, 114,
            130, 125, 130, 125, 128, 123, 128, 123,
            140, 135, 140, 135, 138, 133, 138, 133,
            150, 145, 150, 145,
            0, 0, 0, 0,
        ],
        "Culling_Value": [
            220, 200, 218, 198, 215, 195, 212, 192,
            240, 220, 238, 218, 235, 215, 232, 212,
            260, 240, 258, 238, 255, 235, 252, 232,
            280, 260, 278, 258,
            300, 295, 298, 292,
        ],

        # Location / batch management
        "Farm": ["Demo Breeding Farm"] * 32,
        "House": [
            "Nucleus", "Nucleus", "Nucleus", "Nucleus", "Nucleus", "Nucleus", "Nucleus", "Nucleus",
            "GGPS", "GGPS", "GGPS", "GGPS", "GGPS", "GGPS", "GGPS", "GGPS",
            "GPS", "GPS", "GPS", "GPS", "GPS", "GPS", "GPS", "GPS",
            "PS", "PS", "PS", "PS",
            "Commercial", "Commercial", "Commercial", "Commercial",
        ],
        "Pen": [
            "A01", "A02", "B01", "B02", "C01", "C02", "D01", "D02",
            "A11", "A12", "B11", "B12", "C11", "C12", "D11", "D12",
            "A21", "A22", "B21", "B22", "C21", "C22", "D21", "D22",
            "AB31", "AB32", "CD31", "CD32",
            "FS41", "FS42", "FS43", "FS44",
        ],
        "Cage": [
            "Cage-01", "Cage-02", "Cage-03", "Cage-04", "Cage-05", "Cage-06", "Cage-07", "Cage-08",
            "Cage-11", "Cage-12", "Cage-13", "Cage-14", "Cage-15", "Cage-16", "Cage-17", "Cage-18",
            "Cage-21", "Cage-22", "Cage-23", "Cage-24", "Cage-25", "Cage-26", "Cage-27", "Cage-28",
            "Cage-31", "Cage-32", "Cage-33", "Cage-34",
            "Cage-41", "Cage-42", "Cage-43", "Cage-44",
        ],
        "Batch": [
            "Batch-F0", "Batch-F0", "Batch-F0", "Batch-F0", "Batch-F0", "Batch-F0", "Batch-F0", "Batch-F0",
            "Batch-GGPS", "Batch-GGPS", "Batch-GGPS", "Batch-GGPS", "Batch-GGPS", "Batch-GGPS", "Batch-GGPS", "Batch-GGPS",
            "Batch-GPS", "Batch-GPS", "Batch-GPS", "Batch-GPS", "Batch-GPS", "Batch-GPS", "Batch-GPS", "Batch-GPS",
            "Batch-PS", "Batch-PS", "Batch-PS", "Batch-PS",
            "Batch-FS", "Batch-FS", "Batch-FS", "Batch-FS",
        ],
        "Responsible_Person": ["Breeding Manager"] * 32,
        "Mating_Date": [
            "-", "-", "-", "-", "-", "-", "-", "-",
            "2024-01-10", "2024-01-10", "2024-01-12", "2024-01-12",
            "2024-01-14", "2024-01-14", "2024-01-16", "2024-01-16",
            "2025-02-10", "2025-02-10", "2025-02-12", "2025-02-12",
            "2025-02-14", "2025-02-14", "2025-02-16", "2025-02-16",
            "2026-03-10", "2026-03-10", "2026-03-12", "2026-03-12",
            "2026-05-01", "2026-05-01", "2026-05-02", "2026-05-02",
        ],
        "Expected_Offspring_Date": [
            "-", "-", "-", "-", "-", "-", "-", "-",
            "2024-10-10", "2024-10-10", "2024-10-12", "2024-10-12",
            "2024-10-14", "2024-10-14", "2024-10-16", "2024-10-16",
            "2025-11-10", "2025-11-10", "2025-11-12", "2025-11-12",
            "2025-11-14", "2025-11-14", "2025-11-16", "2025-11-16",
            "2026-12-10", "2026-12-10", "2026-12-12", "2026-12-12",
            "2027-02-01", "2027-02-01", "2027-02-02", "2027-02-02",
        ],
    })



def calculate_selection_response(h2: float, sd_p: float, intensity: float) -> float:
    """
    Selection response formula:
    R = i * h2 * sigma_p
    """
    return intensity * h2 * sd_p


def calculate_inbreeding_depression(f_percent: float, depression_rate: float) -> float:
    """
    Estimated performance reduction:
    Depression = F (%) * depression rate per 1% F
    """
    return f_percent * depression_rate


def analyze_hardy_weinberg(result_df: pd.DataFrame) -> Dict:
    avg_f = result_df["Inbreeding_%"].mean() / 100.0
    is_deviating = avg_f > 0.05

    status = "Deviation Occurring - Not in Equilibrium" if is_deviating else "Near Equilibrium"

    insight = (
        f"The average population inbreeding coefficient is {avg_f:.4f}. "
        "In population genetics, this value can be used as an indicator of deviation "
        "from the ideal random mating condition."
    )

    if is_deviating:
        suggestions = [
            "Outcrossing: introduce unrelated sires or semen from outside the population.",
            "Increase effective population size by using more active sires.",
            "Apply sire rotation and avoid excessive use of one popular sire.",
            "Consider crossbreeding for commercial production to restore heterozygosity.",
        ]
    else:
        suggestions = [
            "The current mating pattern still appears able to maintain genetic variation.",
            "Continue monitoring pedigrees to prevent sudden increases in inbreeding.",
        ]

    return {
        "status": status,
        "insight": insight,
        "saran": suggestions,
        "is_deviating": is_deviating,
    }


def standardize_input(
    raw_df,
    id_col,
    sire_col,
    dam_col,
    phenotype_col=None,
    sex_col=None,
    age_col=None,
    line_col=None,
    generation_col=None,
    trait_cols=None,
    extra_metadata_cols=None,
):
    """
    Standardizes uploaded data into required pedigree columns and optional breeding metadata.

    Required:
    - Animal_ID
    - Sire_ID
    - Dam_ID

    Optional:
    - Phenotype
    - Sex
    - Age
    - Line
    - Generation
    - Additional numeric trait columns for Multi-Trait Selection Index
    - Advanced metadata columns for breeding management modules
    """
    trait_cols = trait_cols or []
    extra_metadata_cols = extra_metadata_cols or {}

    selected = []
    output_names = []

    def add_col(source_col, output_col):
        if source_col and source_col != "-" and source_col in raw_df.columns and output_col not in output_names:
            selected.append(source_col)
            output_names.append(output_col)

    add_col(id_col, "Animal_ID")
    add_col(sire_col, "Sire_ID")
    add_col(dam_col, "Dam_ID")
    add_col(phenotype_col, "Phenotype")
    add_col(sex_col, "Sex")
    add_col(age_col, "Age")
    add_col(line_col, "Line")
    add_col(generation_col, "Generation")

    for output_col, source_col in extra_metadata_cols.items():
        add_col(source_col, output_col)

    for trait_col in trait_cols:
        if trait_col and trait_col != "-" and trait_col in raw_df.columns:
            safe_name = f"Trait_{str(trait_col).strip().replace(' ', '_')}"
            if safe_name not in output_names:
                selected.append(trait_col)
                output_names.append(safe_name)

    df = raw_df[selected].copy()
    df.columns = output_names

    for col in ["Animal_ID", "Sire_ID", "Dam_ID"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_id)

    text_meta_cols = [
        "Sex", "Line", "Generation", "Birth_Date", "Reproductive_Status",
        "Health_Status", "Defect_Status", "Culling_Reason", "Farm",
        "House", "Pen", "Cage", "Group", "Batch", "Mating_Date",
        "Expected_Offspring_Date", "Responsible_Person", "Survival_Status"
    ]

    for col in text_meta_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: EMPTY if is_unknown(x) else x.strip())

    df = df.dropna(subset=["Animal_ID"]).copy()

    if df.empty:
        raise ValueError("No valid Animal_ID was found.")

    return df.reset_index(drop=True)




def klasifikasi_ternak(ebv: float, f_percent: float) -> str:
    if ebv > 0.5 and f_percent < 3.125:
        return "Elite Stock"
    if ebv > 0 and f_percent < 6.25:
        return "Breeding Stock"
    if ebv > 1.0:
        return "Line Breeding"
    if f_percent > 12.5:
        return "Final Stock - Slaughter Only"
    return "Commercial"


def interpretasi_pemuliaan(
    ebv: float,
    f_percent: float,
    animal_id: str,
    results_df: Optional[pd.DataFrame] = None,
) -> str:
    status_ebv = "Superior" if ebv > 0 else "Below average"
    classification = klasifikasi_ternak(ebv, f_percent)

    is_dam = False
    if results_df is not None and "Dam_ID" in results_df.columns:
        is_dam = animal_id in results_df["Dam_ID"].astype(str).values

    msg = f"Genetic Status: {status_ebv} with EBV {ebv:.2f}. "

    if classification == "Elite Stock":
        if is_dam:
            msg += (
                "Classification: Elite Stock - Core Female. "
                "Recommendation: highly suitable as a core dam or embryo donor candidate."
            )
        else:
            msg += (
                "Classification: Elite Stock - Core Sire. "
                "Recommendation: highly suitable as a main sire candidate or frozen semen source."
            )
    elif classification == "Breeding Stock":
        msg += (
            "Classification: Breeding Stock. "
            "Recommendation: suitable as replacement breeding stock for the next generation."
        )
    elif classification == "Line Breeding":
        msg += (
            "Classification: Line Breeding. "
            "Recommendation: strong genetic potential, but kinship must be controlled carefully."
        )
    elif classification == "Final Stock - Slaughter Only":
        msg += (
            "Classification: Final Stock - Slaughter Only. "
            "Recommendation: not recommended for breeding because of high inbreeding risk."
        )
    else:
        msg += (
            "Classification: Commercial. "
            "Recommendation: suitable for commercial production, but not prioritized for genetic improvement."
        )

    return msg


def calculate_stats(df: pd.DataFrame):
    try:
        if "Phenotype" not in df.columns:
            return None

        valid_df = df[df["Phenotype"].apply(lambda x: not is_unknown(x))].copy()

        if len(valid_df) < 3:
            return None

        valid_df["Phenotype"] = pd.to_numeric(valid_df["Phenotype"], errors="coerce")
        valid_df["F"] = pd.to_numeric(valid_df["Inbreeding_%"], errors="coerce")
        valid_df = valid_df.dropna(subset=["Phenotype", "F"])

        if len(valid_df) < 3:
            return None

        correlation = valid_df["F"].corr(valid_df["Phenotype"])

        x = valid_df["F"]
        y = valid_df["Phenotype"]
        n = len(valid_df)

        denominator = (n * (x ** 2).sum() - (x.sum()) ** 2)

        if denominator == 0:
            return None

        b = (n * (x * y).sum() - x.sum() * y.sum()) / denominator
        a = (y.sum() - b * x.sum()) / n
        r_squared = correlation ** 2 if correlation == correlation else 0

        return {
            "correlation": correlation,
            "b": b,
            "a": a,
            "r_squared": r_squared,
        }
    except Exception:
        return None



def validate_input_data(df: pd.DataFrame, phenotype_enabled: bool = False) -> Dict:
    """
    Validates standardized pedigree data before calculation.
    Returns a dictionary containing errors, warnings, and cleaned hints.
    This function prevents long Streamlit tracebacks for common user input mistakes.
    """
    errors = []
    warnings = []

    required_cols = ["Animal_ID", "Sire_ID", "Dam_ID"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        errors.append(
            f"Missing required column(s): {', '.join(missing_cols)}. "
            "Please map Animal_ID, Sire_ID, and Dam_ID correctly."
        )
        return {"valid": False, "errors": errors, "warnings": warnings}

    if df.empty:
        errors.append("The uploaded file does not contain valid Animal_ID records.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Empty Animal_ID
    empty_animal = df["Animal_ID"].apply(is_unknown).sum()
    if empty_animal > 0:
        errors.append(
            f"There are {empty_animal} row(s) with empty Animal_ID. "
            "Every animal must have a unique Animal_ID."
        )

    # Duplicate Animal_ID
    duplicated_ids = df["Animal_ID"][df["Animal_ID"].duplicated(keep=False)].dropna().astype(str).unique().tolist()
    if duplicated_ids:
        shown = ", ".join(duplicated_ids[:10])
        more = "..." if len(duplicated_ids) > 10 else ""
        errors.append(
            f"Duplicate Animal_ID detected: {shown}{more}. "
            "Animal_ID must be unique. Please rename or remove duplicate records."
        )

    # Animal cannot be its own parent
    self_parent_rows = []
    for _, row in df.iterrows():
        animal = str(row["Animal_ID"]) if not is_unknown(row["Animal_ID"]) else None
        sire = str(row["Sire_ID"]) if not is_unknown(row["Sire_ID"]) else None
        dam = str(row["Dam_ID"]) if not is_unknown(row["Dam_ID"]) else None

        if animal and (animal == sire or animal == dam):
            self_parent_rows.append(animal)

    if self_parent_rows:
        shown = ", ".join(self_parent_rows[:10])
        more = "..." if len(self_parent_rows) > 10 else ""
        errors.append(
            f"Animal cannot be its own parent. Problematic Animal_ID: {shown}{more}."
        )

    # Same sire and dam
    same_parent_rows = []
    for _, row in df.iterrows():
        sire = str(row["Sire_ID"]) if not is_unknown(row["Sire_ID"]) else None
        dam = str(row["Dam_ID"]) if not is_unknown(row["Dam_ID"]) else None
        animal = str(row["Animal_ID"]) if not is_unknown(row["Animal_ID"]) else "-"

        if sire and dam and sire == dam:
            same_parent_rows.append(animal)

    if same_parent_rows:
        shown = ", ".join(same_parent_rows[:10])
        more = "..." if len(same_parent_rows) > 10 else ""
        warnings.append(
            f"Sire_ID and Dam_ID are the same for: {shown}{more}. "
            "Please verify the record because sire and dam should usually be different animals."
        )

    # Parent IDs not found in Animal_ID list are allowed as additional founders,
    # but warn user so they understand how the app handles it.
    animal_ids = set(df["Animal_ID"].dropna().astype(str))
    parent_ids = set(df["Sire_ID"].dropna().astype(str)).union(set(df["Dam_ID"].dropna().astype(str)))
    parent_ids = {p for p in parent_ids if not is_unknown(p)}

    missing_parents = sorted(parent_ids - animal_ids)
    if missing_parents:
        shown = ", ".join(missing_parents[:10])
        more = "..." if len(missing_parents) > 10 else ""
        warnings.append(
            f"{len(missing_parents)} parent ID(s) are not listed as Animal_ID: {shown}{more}. "
            "The system will treat them as additional founders with unknown parents."
        )

    # Phenotype validation
    if phenotype_enabled and "Phenotype" in df.columns:
        pheno_raw = df["Phenotype"].copy()
        non_empty_pheno = pheno_raw[~pheno_raw.apply(is_unknown)]

        if non_empty_pheno.empty:
            warnings.append(
                "Phenotype column was selected but all phenotype values are empty. "
                "EBV and selection response may not be informative."
            )
        else:
            numeric_pheno = pd.to_numeric(non_empty_pheno, errors="coerce")
            invalid_count = int(numeric_pheno.isna().sum())

            if invalid_count > 0:
                warnings.append(
                    f"There are {invalid_count} non-numeric Phenotype value(s). "
                    "Use numbers only, for example 450, 520.5, or 1200."
                )

    # Cycle check before matrix calculation
    try:
        parents_map = {}
        for _, row in df.iterrows():
            animal = None if is_unknown(row["Animal_ID"]) else str(row["Animal_ID"])
            if not animal:
                continue
            sire = None if is_unknown(row["Sire_ID"]) else str(row["Sire_ID"])
            dam = None if is_unknown(row["Dam_ID"]) else str(row["Dam_ID"])
            parents_map[animal] = (sire, dam)

        # Add missing parents as founders for cycle checking.
        for parent in parent_ids:
            parents_map.setdefault(parent, (None, None))

        state = {}

        def visit(animal, path_stack):
            if animal is None:
                return
            status = state.get(animal, 0)
            if status == 1:
                cycle_path = " -> ".join(path_stack + [animal])
                raise ValueError(cycle_path)
            if status == 2:
                return

            state[animal] = 1
            sire, dam = parents_map.get(animal, (None, None))
            for parent in [sire, dam]:
                if parent is not None:
                    visit(parent, path_stack + [animal])
            state[animal] = 2

        for animal in list(parents_map.keys()):
            visit(animal, [])

    except ValueError as cycle:
        errors.append(
            f"Pedigree cycle detected: {cycle}. "
            "Please check parent records. An animal cannot be its own ancestor."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def show_input_validation_messages(validation_result: Dict):
    """
    Shows clean Streamlit notifications for invalid uploaded data.
    """
    if validation_result.get("errors"):
        st.error("The uploaded data cannot be processed yet. Please fix the following issue(s):")
        for err in validation_result["errors"]:
            st.write(f"- {err}")

    if validation_result.get("warnings"):
        st.warning("The uploaded data can still be reviewed, but please check the following warning(s):")
        for warn in validation_result["warnings"]:
            st.write(f"- {warn}")

    with st.expander("Correct Data Writing Rules", expanded=False):
        st.markdown("""
        **Main Data Writing Rules**
        - `Animal_ID` must be unique and must not be empty.
        - Use `-` for unknown `Sire_ID` or `Dam_ID`.
        - An animal cannot be listed as its own sire or dam.
        - If using `Phenotype` or other traits, use numeric values only.
        - Avoid pedigree cycles, for example A is a parent of B but B is also an ancestor of A.
        """)



def merge_optional_metadata(std_df: pd.DataFrame, res_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges optional metadata and trait columns from standardized input into result dataframe.
    """
    optional_cols = [
        col for col in std_df.columns
        if col not in ["Sire_ID", "Dam_ID", "Phenotype"]
    ]

    if not optional_cols or "Animal_ID" not in optional_cols:
        return res_df

    meta_df = std_df[optional_cols].copy()
    out = res_df.merge(meta_df, on="Animal_ID", how="left", suffixes=("", "_meta"))

    for col in optional_cols:
        if col != "Animal_ID" and col in out.columns and not col.startswith("Trait_"):
            out[col] = out[col].apply(lambda x: EMPTY if is_unknown(x) else str(x).strip())

    return out



def normalize_sex_value(value: str) -> Optional[str]:
    """
    Converts common sex labels into Male/Female role labels.
    """
    if is_unknown(value):
        return None

    text = str(value).strip().lower()

    male_values = {
        "m", "male", "male", "pemale", "sire", "bull", "rooster", "cock",
        "l", "laki", "laki-laki", "ayam male"
    }

    female_values = {
        "f", "female", "female", "dam", "dam", "cow", "hen",
        "p", "perempuan", "ayam female"
    }

    if text in male_values:
        return "Male / Sire Candidate"

    if text in female_values:
        return "Female / Dam Candidate"

    return None


def calculate_data_quality_score(df: pd.DataFrame) -> Dict:
    """
    Calculates a simple data quality score for breeder review.
    """
    total = len(df)
    if total == 0:
        return {
            "score": 0,
            "grade": "Invalid",
            "details": pd.DataFrame(),
            "recommendation": "No valid data is available."
        }

    animal_complete = 100.0 * (1 - df["Animal_ID"].apply(is_unknown).mean()) if "Animal_ID" in df.columns else 0
    sire_complete = 100.0 * (1 - df["Sire_ID"].apply(is_unknown).mean()) if "Sire_ID" in df.columns else 0
    dam_complete = 100.0 * (1 - df["Dam_ID"].apply(is_unknown).mean()) if "Dam_ID" in df.columns else 0

    if "Phenotype" in df.columns:
        phenotype_complete = 100.0 * (1 - df["Phenotype"].apply(is_unknown).mean())
    else:
        phenotype_complete = 0

    sex_complete = 100.0 * (1 - df["Sex"].apply(is_unknown).mean()) if "Sex" in df.columns else 0
    age_complete = 100.0 * (1 - df["Age"].apply(is_unknown).mean()) if "Age" in df.columns else 0
    line_complete = 100.0 * (1 - df["Line"].apply(is_unknown).mean()) if "Line" in df.columns else 0
    generation_complete = 100.0 * (1 - df["Generation"].apply(is_unknown).mean()) if "Generation" in df.columns else 0

    duplicate_rate = 100.0 * df["Animal_ID"].duplicated().mean() if "Animal_ID" in df.columns else 100
    warning_rate = 100.0 * (df["Reproduction_Warning"].astype(str) != "").mean() if "Reproduction_Warning" in df.columns else 0

    parent_ids = set()
    if "Sire_ID" in df.columns:
        parent_ids |= set(df["Sire_ID"].dropna().astype(str))
    if "Dam_ID" in df.columns:
        parent_ids |= set(df["Dam_ID"].dropna().astype(str))
    parent_ids = {p for p in parent_ids if not is_unknown(p)}

    animal_ids = set(df["Animal_ID"].dropna().astype(str)) if "Animal_ID" in df.columns else set()
    missing_parent_rate = 100.0 * (len(parent_ids - animal_ids) / max(len(parent_ids), 1))

    components = {
        "Animal ID completeness": animal_complete,
        "Sire completeness": sire_complete,
        "Dam completeness": dam_complete,
        "Phenotype completeness": phenotype_complete,
        "Sex completeness": sex_complete,
        "Age completeness": age_complete,
        "Line completeness": line_complete,
        "Generation completeness": generation_complete,
        "Duplicate penalty": max(0, 100 - duplicate_rate),
        "Missing parent penalty": max(0, 100 - missing_parent_rate),
        "Warning penalty": max(0, 100 - warning_rate),
    }

    # Weighted score: core pedigree has the highest influence.
    score = (
        animal_complete * 0.15 +
        sire_complete * 0.12 +
        dam_complete * 0.12 +
        phenotype_complete * 0.12 +
        sex_complete * 0.10 +
        age_complete * 0.07 +
        line_complete * 0.08 +
        generation_complete * 0.08 +
        max(0, 100 - duplicate_rate) * 0.06 +
        max(0, 100 - missing_parent_rate) * 0.05 +
        max(0, 100 - warning_rate) * 0.05
    )

    if score >= 85:
        grade = "Excellent"
        recommendation = "Data quality is strong enough for structured breeding decisions."
    elif score >= 70:
        grade = "Good"
        recommendation = "Data can be used, but completing missing metadata will improve accuracy."
    elif score >= 50:
        grade = "Fair"
        recommendation = "Use results cautiously. Improve pedigree, sex, line, generation, and phenotype records."
    else:
        grade = "Poor"
        recommendation = "Data quality is too limited for reliable breeding decisions. Improve recording first."

    details = pd.DataFrame({
        "Data Quality Component": list(components.keys()),
        "Completeness / Score": [round(v, 2) for v in components.values()],
    })

    return {
        "score": round(float(score), 2),
        "grade": grade,
        "details": details,
        "recommendation": recommendation,
    }


def calculate_multi_trait_selection_index(result_df: pd.DataFrame, trait_columns: list, trait_weights: Dict[str, float]) -> pd.DataFrame:
    """
    Calculates a simple standardized Multi-Trait Selection Index.

    Each selected trait is converted into z-score:
    z = (value - mean) / sd

    Selection_Index = sum(weight * z_trait)

    Higher values indicate better overall multi-trait performance.
    """
    df = result_df.copy()

    if not trait_columns:
        df["Selection_Index"] = df.get("EBV", 0)
        df["Selection_Index_Rank"] = df["Selection_Index"].rank(ascending=False, method="dense").astype(int)
        return df

    index_value = pd.Series(0.0, index=df.index)

    total_weight = sum(float(trait_weights.get(col, 0)) for col in trait_columns)
    if total_weight <= 0:
        total_weight = 1.0

    used_traits = []

    for col in trait_columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() < 2:
            continue

        sd = values.std()
        if sd == 0 or pd.isna(sd):
            continue

        z = (values - values.mean()) / sd
        weight = float(trait_weights.get(col, 0)) / total_weight
        index_value = index_value + (z.fillna(0) * weight)
        used_traits.append(col)

    if not used_traits:
        df["Selection_Index"] = df.get("EBV", 0)
    else:
        # Combine multi-trait phenotype index with EBV direction.
        ebv = pd.to_numeric(df.get("EBV", 0), errors="coerce").fillna(0)
        ebv_z = (ebv - ebv.mean()) / ebv.std() if ebv.std() not in [0, np.nan] and not pd.isna(ebv.std()) else ebv * 0
        df["Selection_Index"] = (0.7 * index_value) + (0.3 * ebv_z.fillna(0))

    df["Selection_Index"] = df["Selection_Index"].round(4)
    df["Selection_Index_Rank"] = df["Selection_Index"].rank(ascending=False, method="dense").astype(int)

    return df


def evaluate_blocked_mating_rules(
    sire_id: str,
    dam_id: str,
    result_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    max_safe_f: float = 6.25,
) -> Dict:
    """
    Evaluates blocked mating rules for a proposed sire-dam pair.
    """
    sire_id = str(sire_id)
    dam_id = str(dam_id)

    df = result_df.copy()
    df["Animal_ID"] = df["Animal_ID"].astype(str)

    lookup = df.set_index("Animal_ID").to_dict("index") if "Animal_ID" in df.columns else {}

    sire_row = lookup.get(sire_id, {})
    dam_row = lookup.get(dam_id, {})

    sire_sire = None if is_unknown(sire_row.get("Sire_ID")) else str(sire_row.get("Sire_ID"))
    sire_dam = None if is_unknown(sire_row.get("Dam_ID")) else str(sire_row.get("Dam_ID"))
    dam_sire = None if is_unknown(dam_row.get("Sire_ID")) else str(dam_row.get("Sire_ID"))
    dam_dam = None if is_unknown(dam_row.get("Dam_ID")) else str(dam_row.get("Dam_ID"))

    reasons = []
    status = "Allowed"

    if sire_id == dam_id:
        reasons.append("Same animal cannot be mated with itself.")

    # Parent-offspring
    if dam_sire == sire_id or dam_dam == sire_id:
        reasons.append("Blocked: sire is a parent of the dam.")
    if sire_sire == dam_id or sire_dam == dam_id:
        reasons.append("Blocked: dam is a parent of the sire.")

    # Full sibling
    if sire_sire and sire_dam and dam_sire and dam_dam and sire_sire == dam_sire and sire_dam == dam_dam:
        reasons.append("Blocked: full-sibling mating.")

    # Half sibling
    elif (sire_sire and sire_sire == dam_sire) or (sire_dam and sire_dam == dam_dam) or (sire_sire and sire_sire == dam_dam) or (sire_dam and sire_dam == dam_sire):
        reasons.append("Warning: half-sibling or close shared-parent mating.")

    # Same line rules
    sire_line = str(sire_row.get("Line", EMPTY))
    dam_line = str(dam_row.get("Line", EMPTY))
    if not is_unknown(sire_line) and not is_unknown(dam_line) and sire_line == dam_line:
        reasons.append("Warning: same-line mating. Use only if maintaining pure line; avoid for crossing.")

    # FS as breeding nucleus
    sire_gen = str(sire_row.get("Generation", EMPTY)).upper()
    dam_gen = str(dam_row.get("Generation", EMPTY)).upper()
    if sire_gen == "FS" or dam_gen == "FS":
        reasons.append("Blocked: FS should be treated as terminal commercial stock, not breeding nucleus.")

    # Relationship-based risk
    predicted_f = None
    if sire_id in matrix_df.index and dam_id in matrix_df.columns:
        relationship = float(matrix_df.loc[sire_id, dam_id])
        predicted_f = 0.5 * relationship * 100
        if predicted_f >= 25:
            reasons.append("Blocked: predicted offspring F is very high.")
        elif predicted_f >= 12.5:
            reasons.append("High risk: predicted offspring F is above 12.5%.")
        elif predicted_f > max_safe_f:
            reasons.append("Caution: predicted offspring F is above selected safe threshold.")

    if any(reason.startswith("Blocked") for reason in reasons):
        status = "Blocked"
    elif reasons:
        status = "Warning"
    else:
        status = "Allowed"

    return {
        "Mating_Rule_Status": status,
        "Blocked_Mating_Reasons": " | ".join(reasons) if reasons else "No blocking rule detected.",
        "Rule_Predicted_F_%": None if predicted_f is None else round(float(predicted_f), 4),
    }


def apply_blocked_mating_rules_to_pairs(pair_df: pd.DataFrame, result_df: pd.DataFrame, matrix_df: pd.DataFrame, max_safe_f: float = 6.25) -> pd.DataFrame:
    """
    Adds blocked mating rule columns to simulated mating pairs.
    """
    if pair_df.empty:
        return pair_df

    out = pair_df.copy()
    statuses = []
    reasons = []

    for _, row in out.iterrows():
        eval_res = evaluate_blocked_mating_rules(
            row["Suggested_Sire"],
            row["Suggested_Dam"],
            result_df,
            matrix_df,
            max_safe_f=max_safe_f,
        )
        statuses.append(eval_res["Mating_Rule_Status"])
        reasons.append(eval_res["Blocked_Mating_Reasons"])

    out["Mating_Rule_Status"] = statuses
    out["Blocked_Mating_Reasons"] = reasons

    # Allowed and Warning pairs first; Blocked pairs last.
    status_order = {"Allowed": 0, "Warning": 1, "Blocked": 2}
    out["_status_order"] = out["Mating_Rule_Status"].map(status_order).fillna(9)
    out = out.sort_values(
        ["_status_order", "Predicted_Offspring_F_%", "Expected_Offspring_EBV"],
        ascending=[True, True, False],
    ).drop(columns=["_status_order"]).reset_index(drop=True)

    return out


def calculate(df_input, h2=0.3, depression_rate=1.0):
    animal_ids = set(df_input["Animal_ID"])
    parent_ids = set(df_input["Sire_ID"].dropna()).union(set(df_input["Dam_ID"].dropna()))
    missing = sorted(parent_ids - animal_ids)

    founder_df = pd.DataFrame({
        "Animal_ID": missing,
        "Sire_ID": [None] * len(missing),
        "Dam_ID": [None] * len(missing),
    })

    if "Phenotype" in df_input.columns:
        founder_df["Phenotype"] = None

    df_full = pd.concat([founder_df, df_input], ignore_index=True)

    parents_map = {}
    pheno_map = {}

    for row in df_full.itertuples(index=False):
        if row.Animal_ID:
            s_val = None if is_unknown(row.Sire_ID) else str(row.Sire_ID)
            d_val = None if is_unknown(row.Dam_ID) else str(row.Dam_ID)

            parents_map[str(row.Animal_ID)] = (s_val, d_val)

            if hasattr(row, "Phenotype"):
                pheno_map[str(row.Animal_ID)] = row.Phenotype

    order = []
    state = {}

    def visit(animal):
        if animal is None:
            return

        status = state.get(animal, 0)

        if status == 1:
            raise ValueError(f"Pedigree cycle detected at {animal}. Please check the parent records.")

        if status == 2:
            return

        state[animal] = 1
        sire, dam = parents_map.get(animal, (None, None))

        for parent in [sire, dam]:
            if parent is not None:
                visit(parent)

        state[animal] = 2
        order.append(animal)

    for animal in list(parents_map.keys()):
        visit(animal)

    n = len(order)
    idx_map = {animal: i for i, animal in enumerate(order)}
    A = np.zeros((n, n), dtype=np.float32)

    phenotype_values = [
        pd.to_numeric(v, errors="coerce")
        for v in pheno_map.values()
        if v is not None and not is_unknown(v)
    ]
    phenotype_values = [v for v in phenotype_values if not pd.isna(v)]
    pop_avg = np.mean(phenotype_values) if phenotype_values else 0

    parent_indices = []
    for animal in order:
        sire, dam = parents_map.get(animal, (None, None))
        si = idx_map.get(sire) if sire is not None else None
        di = idx_map.get(dam) if dam is not None else None
        parent_indices.append((si, di))

    rows = []

    for i in range(n):
        si, di = parent_indices[i]

        if si is not None and di is not None:
            A[i, 0:i] = A[0:i, i] = 0.5 * (A[si, 0:i] + A[di, 0:i])
            F = 0.5 * A[si, di]
        elif si is not None:
            A[i, 0:i] = A[0:i, i] = 0.5 * A[si, 0:i]
            F = 0.0
        elif di is not None:
            A[i, 0:i] = A[0:i, i] = 0.5 * A[di, 0:i]
            F = 0.0
        else:
            F = 0.0

        A[i, i] = 1.0 + F

        animal_id = order[i]
        p_val = pheno_map.get(animal_id)

        ebv = 0.0
        if p_val is not None and not is_unknown(p_val):
            try:
                ebv = h2 * (float(p_val) - pop_avg)
            except Exception:
                ebv = 0.0

        heterosis = 0.0
        if si is not None and di is not None and p_val is not None and not is_unknown(p_val):
            p_sire = pheno_map.get(order[si])
            p_dam = pheno_map.get(order[di])

            if not is_unknown(p_sire) and not is_unknown(p_dam):
                try:
                    p_anak = float(p_val)
                    p_avg_parents = 0.5 * (float(p_sire) + float(p_dam))
                    heterosis = p_anak - p_avg_parents
                except Exception:
                    heterosis = 0.0

        depression = calculate_inbreeding_depression(F * 100, depression_rate)

        rows.append({
            "Animal_ID": animal_id,
            "Sire_ID": parents_map.get(animal_id)[0],
            "Dam_ID": parents_map.get(animal_id)[1],
            "Phenotype": show_value(p_val),
            "EBV": round(float(ebv), 4),
            "Heterosis": round(float(heterosis), 4),
            "Inbreeding_Depression": round(float(depression), 4),
            "Inbreeding_%": round(float(F * 100), 4),
            "Data_Type": "Additional founder" if animal_id in missing else "Input data",
        })

    res_df = pd.DataFrame(rows)

    res_df["Classification"] = res_df.apply(
        lambda r: klasifikasi_ternak(r["EBV"], r["Inbreeding_%"]),
        axis=1,
    )

    res_df["Breeding_Interpretation"] = res_df.apply(
        lambda r: interpretasi_pemuliaan(
            r["EBV"],
            r["Inbreeding_%"],
            r["Animal_ID"],
            res_df,
        ),
        axis=1,
    )

    res_df["Inbreeding_Condition"] = res_df["Inbreeding_%"].apply(kondisi_inbreeding)
    res_df["Biological_Impact"] = res_df["Inbreeding_%"].apply(dampak_inbreeding)
    res_df["Recommendation"] = res_df["Inbreeding_%"].apply(rekomendasi)

    res_df["Reproduction_Warning"] = ""

    for idx, row in res_df.iterrows():
        sire = row["Sire_ID"]
        dam = row["Dam_ID"]

        if sire and dam:
            grand_sire_of_dam, _ = parents_map.get(str(dam), (None, None))
            if str(sire) == str(grand_sire_of_dam):
                res_df.at[idx, "Reproduction_Warning"] = "SIRE-DAUGHTER MATING DETECTED"

    matrix_df = pd.DataFrame(A, index=order, columns=order)

    return clean_display(df_input), res_df, matrix_df


def read_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    return pd.read_excel(uploaded_file, dtype=str, keep_default_na=False)


def dot_escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def make_dot(result_df, max_nodes=50):
    df = result_df.head(max_nodes)
    animal_set = set(df["Animal_ID"].astype(str))

    dot = [
        "digraph Pedigree {",
        "rankdir=LR;",
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10];',
        "edge [arrowsize=0.6];",
    ]

    for _, row in df.iterrows():
        animal = dot_escape(row["Animal_ID"])
        f_value = float(row["Inbreeding_%"])
        classification = row.get("Classification", "")

        fill = "#FFFFFF"
        border_color = "black"
        penwidth = "1.0"

        if f_value >= 25:
            fill = "#FFE4E1"
        elif f_value > 0:
            fill = "#FFF4CC"

        if classification == "Elite Stock":
            border_color = "#FFD700"
            penwidth = "3.0"
        elif classification == "Breeding Stock":
            border_color = "#32CD32"
            penwidth = "2.0"
        elif "Final Stock" in classification:
            border_color = "#DC2626"
            penwidth = "2.0"

        label = f"{animal}\\nF={f_value:.2f}%\\n{classification}"

        dot.append(
            f'"{animal}" [label="{label}", fillcolor="{fill}", color="{border_color}", penwidth="{penwidth}"];'
        )

    for _, row in df.iterrows():
        animal = dot_escape(row["Animal_ID"])
        sire = row["Sire_ID"]
        dam = row["Dam_ID"]

        if sire and str(sire) in animal_set:
            dot.append(f'"{dot_escape(sire)}" -> "{animal}" [label="sire"];')

        if dam and str(dam) in animal_set:
            dot.append(f'"{dot_escape(dam)}" -> "{animal}" [label="dam"];')

    dot.append("}")
    return "\n".join(dot)


def dots_to_pedigree(result_df, settings=None):
    lines = [
        "LIVESTOCK BREEDING & INBREEDING SUMMARY",
        "=" * 60,
        f"Printed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
    ]

    avg_f = result_df["Inbreeding_%"].mean()
    avg_ebv = result_df["EBV"].mean()

    lines.append("1. POPULATION STATISTICS")
    lines.append(f"   Total Animals        : {len(result_df)}")
    lines.append(f"   Average Inbreeding F : {avg_f:.2f}%")
    lines.append(f"   Average EBV          : {avg_ebv:.4f}")

    if "Phenotype" in result_df.columns:
        valid_phenos = pd.to_numeric(result_df["Phenotype"], errors="coerce").dropna()

        if not valid_phenos.empty:
            avg_p = valid_phenos.mean()
            sd_p = valid_phenos.std()
            lines.append(f"   Average Phenotype    : {avg_p:.2f}")
            lines.append(f"   Phenotype SD         : {sd_p:.2f}")

            if settings and "h2" in settings and "intensity" in settings:
                response = calculate_selection_response(settings["h2"], sd_p, settings["intensity"])
                lines.append(f"   Selection Response R : {response:.4f}")

    hwe_res = analyze_hardy_weinberg(result_df)
    lines.append(f"   HWE Genetic Status   : {hwe_res['status']}")
    lines.append("-" * 60)
    lines.append("")

    threshold_ebv = result_df["EBV"].quantile(0.75)
    selection_df = result_df[
        (result_df["EBV"] >= threshold_ebv) &
        (result_df["Inbreeding_%"] < 6.25)
    ].sort_values("EBV", ascending=False).head(10)

    lines.append("2. TOP SELECTION CANDIDATES")
    if not selection_df.empty:
        lines.append(f"   {'Animal_ID':<15} {'EBV':<10} {'F (%)':<8} {'Classification':<25}")
        for _, row in selection_df.iterrows():
            lines.append(
                f"   {str(row['Animal_ID']):<15} {row['EBV']:<10.4f} "
                f"{row['Inbreeding_%']:<8.2f} {str(row['Classification']):<25}"
            )
    else:
        lines.append("   No candidates met the selection criteria.")

    lines.append("-" * 60)
    lines.append("")

    threshold_low_ebv = result_df["EBV"].quantile(0.10)
    culling_df = result_df[
        (result_df["Inbreeding_%"] >= 25) |
        (result_df["Reproduction_Warning"] != "") |
        (result_df["EBV"] <= threshold_low_ebv)
    ].sort_values("Inbreeding_%", ascending=False).head(10)

    lines.append("3. PRIORITY CULLING CANDIDATES")
    if not culling_df.empty:
        lines.append(f"   {'Animal_ID':<15} {'F (%)':<8} {'Warning':<35}")
        for _, row in culling_df.iterrows():
            warning = row["Reproduction_Warning"] if row["Reproduction_Warning"] else "Low genetic value or high F"
            lines.append(
                f"   {str(row['Animal_ID']):<15} {row['Inbreeding_%']:<8.2f} {warning:<35}"
            )
    else:
        lines.append("   No high-priority culling candidates found.")

    lines.append("-" * 60)
    lines.append("")
    lines.append("4. COMPLETE INDIVIDUAL LOG")

    for _, row in result_df.iterrows():
        lines.append(
            f"   [{row['Animal_ID']}] F: {row['Inbreeding_%']:.2f}% | "
            f"EBV: {row['EBV']:.4f} | {row['Classification']}"
        )
        if row["Reproduction_Warning"]:
            lines.append(f"   WARNING: {row['Reproduction_Warning']}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("END OF REPORT")

    return "\n".join(lines)


def generate_pdf(result_df, settings=None):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    title_style = styles["Heading1"]
    title_style.alignment = 1

    elements.append(Paragraph("LIVESTOCK BREEDING ANALYSIS REPORT", title_style))
    elements.append(Spacer(1, 20))

    avg_f = result_df["Inbreeding_%"].mean()
    avg_ebv = result_df["EBV"].mean()

    elements.append(Paragraph("<b>1. Population Summary</b>", styles["Heading2"]))

    summary_data = [
        ["Total Population", f"{len(result_df)} animals"],
        ["Average Inbreeding F", f"{avg_f:.2f}%"],
        ["Average EBV", f"{avg_ebv:.4f}"],
        ["Analysis Time", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")],
    ]

    if "Phenotype" in result_df.columns:
        valid_phenos = pd.to_numeric(result_df["Phenotype"], errors="coerce").dropna()
        if not valid_phenos.empty:
            avg_p = valid_phenos.mean()
            sd_p = valid_phenos.std()
            summary_data.append(["Average Phenotype", f"{avg_p:.2f}"])
            summary_data.append(["Phenotype SD", f"{sd_p:.2f}"])

            if settings and "h2" in settings and "intensity" in settings:
                response = calculate_selection_response(settings["h2"], sd_p, settings["intensity"])
                summary_data.append(["Selection Response R", f"{response:.4f}"])

    summary_table = Table(summary_data, colWidths=[180, 250])
    summary_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 15))

    hwe_res = analyze_hardy_weinberg(result_df)

    elements.append(Paragraph("<b>2. Hardy-Weinberg Equilibrium Analysis</b>", styles["Heading2"]))
    elements.append(Paragraph(f"<b>Status:</b> {hwe_res['status']}", styles["Normal"]))
    elements.append(Paragraph(hwe_res["insight"], styles["Normal"]))
    elements.append(Paragraph("<b>Recommendations:</b>", styles["Normal"]))

    for item in hwe_res["saran"]:
        elements.append(Paragraph(f"- {item}", styles["Normal"]))

    elements.append(Spacer(1, 15))

    elements.append(Paragraph("<b>3. Individual Details and Classification</b>", styles["Heading2"]))

    table_data = [["Animal_ID", "F (%)", "EBV", "Classification"]]
    limit_df = result_df.head(1000)

    for _, row in limit_df.iterrows():
        table_data.append([
            str(row["Animal_ID"]),
            f"{row['Inbreeding_%']:.2f}",
            f"{row['EBV']:.4f}",
            str(row["Classification"]),
        ])

    if len(result_df) > 1000:
        table_data.append(["...", "...", "...", "Data truncated"])

    detail_table = Table(table_data, repeatRows=1, colWidths=[110, 80, 80, 180])
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))

    elements.append(detail_table)
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("<b>4. Mating Strategy</b>", styles["Heading2"]))
    elements.append(Paragraph(
        "To maintain genetic progress while reducing inbreeding risk, use animals with high EBV and low kinship. "
        "Avoid full-sibling, half-sibling, and parent-offspring mating. Use sire rotation and unrelated semen sources "
        "when needed.",
        styles["Normal"],
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer



def detect_sex_role(animal_id: str, sire_role_ids: set, dam_role_ids: set, explicit_sex=None) -> str:
    """
    Detects animal sex/role.

    Priority:
    1. Explicit Sex column if available.
    2. Sire_ID / Dam_ID role in pedigree.
    3. Common ID patterns.
    4. Unidentified.
    """
    explicit = normalize_sex_value(explicit_sex)
    if explicit:
        return explicit

    animal_text = str(animal_id).strip()
    animal_upper = animal_text.upper()

    male_keywords = [
        "SIRE", "BULL", "MALE", "PEJANTAN", "JANTAN",
        "ROOSTER", "COCK", "M_", "M-", "M.", "L_", "L-", "L."
    ]

    female_keywords = [
        "DAM", "COW", "FEMALE", "INDUK", "BETINA",
        "HEN", "F_", "F-", "F.", "P_", "P-", "P."
    ]

    in_sire = animal_text in sire_role_ids
    in_dam = animal_text in dam_role_ids

    if in_sire and not in_dam:
        return "Male / Sire Candidate"

    if in_dam and not in_sire:
        return "Female / Dam Candidate"

    if in_sire and in_dam:
        return "Parent Role - Check Sex"

    if any(key in animal_upper for key in male_keywords):
        return "Male / Sire Candidate"

    if any(key in animal_upper for key in female_keywords):
        return "Female / Dam Candidate"

    return "Unidentified Sex"


def add_sex_role_column(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Sex_Role column to make male and female identification easier.
    Uses explicit Sex column first if available.
    """
    df = result_df.copy()

    sire_role_ids = set(
        df["Sire_ID"]
        .dropna()
        .astype(str)
        .loc[lambda s: ~s.str.lower().isin(UNKNOWN_VALUES)]
    )

    dam_role_ids = set(
        df["Dam_ID"]
        .dropna()
        .astype(str)
        .loc[lambda s: ~s.str.lower().isin(UNKNOWN_VALUES)]
    )

    if "Sex" in df.columns:
        df["Sex_Role"] = df.apply(
            lambda r: detect_sex_role(r["Animal_ID"], sire_role_ids, dam_role_ids, r.get("Sex")),
            axis=1,
        )
    else:
        df["Sex_Role"] = df["Animal_ID"].astype(str).apply(
            lambda x: detect_sex_role(x, sire_role_ids, dam_role_ids)
        )

    return df


def infer_mating_candidates(result_df: pd.DataFrame):
    """
    Separates male/sire and female/dam candidates directly.
    If a real Sex column is not available, the system infers sex/role from:
    - whether the animal appears in Sire_ID or Dam_ID,
    - ID patterns such as SIRE, BULL, DAM, COW, JANTAN, BETINA,
    - EBV and inbreeding eligibility.
    """
    df = add_sex_role_column(result_df)

    selectable = df[
        (df["Inbreeding_%"] < 12.5) &
        (~df["Classification"].astype(str).str.contains("Final Stock", case=False, na=False))
    ].copy()

    if selectable.empty:
        selectable = df.copy()

    # Prefer animals with positive EBV when available.
    positive_selectable = selectable[selectable["EBV"] > 0].copy()
    if not positive_selectable.empty:
        selectable = positive_selectable

    sires = selectable[
        selectable["Sex_Role"].astype(str).str.contains("Male|Sire", case=False, na=False)
    ].copy()

    dams = selectable[
        selectable["Sex_Role"].astype(str).str.contains("Female|Dam", case=False, na=False)
    ].copy()

    # Fallback if uploaded data has no clear role/sex information.
    unidentified = selectable[
        selectable["Sex_Role"].astype(str).str.contains("Unidentified|Check", case=False, na=False)
    ].copy()

    if sires.empty and not unidentified.empty:
        sires = unidentified.sort_values(["EBV", "Inbreeding_%"], ascending=[False, True]).head(5).copy()
        sires["Sex_Role"] = "Male / Sire Candidate - Assumed"

    if dams.empty and not unidentified.empty:
        dams = unidentified.sort_values(["EBV", "Inbreeding_%"], ascending=[False, True]).head(10).copy()
        dams["Sex_Role"] = "Female / Dam Candidate - Assumed"

    sires = sires.sort_values(["EBV", "Inbreeding_%"], ascending=[False, True])
    dams = dams.sort_values(["EBV", "Inbreeding_%"], ascending=[False, True])

    return sires, dams

def simulate_mating_pairs(
    result_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    h2_value: float,
    depression_rate_value: float,
    max_offspring_f: float = 6.25,
    max_pairs: int = 20,
) -> pd.DataFrame:
    """
    Simulates recommended mating pairs using the additive relationship matrix.

    Predicted offspring inbreeding:
    F_offspring = 0.5 * relationship(sire, dam) * 100

    Expected offspring EBV:
    EBV_offspring = 0.5 * (EBV_sire + EBV_dam)
    """
    sires, dams = infer_mating_candidates(result_df)

    valid_phenotypes = pd.to_numeric(result_df.get("Phenotype", pd.Series(dtype=float)), errors="coerce").dropna()
    population_avg = valid_phenotypes.mean() if not valid_phenotypes.empty else np.nan

    simulations = []

    for _, sire in sires.iterrows():
        sire_id = str(sire["Animal_ID"])

        for _, dam in dams.iterrows():
            dam_id = str(dam["Animal_ID"])

            if sire_id == dam_id:
                continue

            if sire_id not in matrix_df.index or dam_id not in matrix_df.columns:
                continue

            relationship = float(matrix_df.loc[sire_id, dam_id])
            predicted_f = 0.5 * relationship * 100
            expected_ebv = 0.5 * (float(sire["EBV"]) + float(dam["EBV"]))

            sire_pheno = pd.to_numeric(sire.get("Phenotype", np.nan), errors="coerce")
            dam_pheno = pd.to_numeric(dam.get("Phenotype", np.nan), errors="coerce")

            if not pd.isna(sire_pheno) and not pd.isna(dam_pheno):
                expected_phenotype_base = 0.5 * (float(sire_pheno) + float(dam_pheno))
            elif not pd.isna(population_avg) and h2_value > 0:
                expected_phenotype_base = population_avg + (expected_ebv / h2_value)
            elif not pd.isna(population_avg):
                expected_phenotype_base = population_avg
            else:
                expected_phenotype_base = np.nan

            estimated_depression = calculate_inbreeding_depression(predicted_f, depression_rate_value)

            if not pd.isna(expected_phenotype_base):
                predicted_phenotype_after_depression = expected_phenotype_base - estimated_depression
            else:
                predicted_phenotype_after_depression = np.nan

            if predicted_f <= 0:
                risk_level = "Very Safe"
            elif predicted_f < 6.25:
                risk_level = "Recommended"
            elif predicted_f < 12.5:
                risk_level = "Use with Caution"
            elif predicted_f < 25:
                risk_level = "High Risk"
            else:
                risk_level = "Avoid"

            if predicted_f <= max_offspring_f and expected_ebv > 0:
                decision = "Recommended mating"
            elif predicted_f <= max_offspring_f:
                decision = "Genetically safe, but EBV is not strong"
            elif predicted_f < 12.5 and expected_ebv > 0:
                decision = "Possible, but monitor inbreeding"
            else:
                decision = "Not recommended"

            simulations.append({
                "Suggested_Sire": sire_id,
                "Suggested_Dam": dam_id,
                "Sire_EBV": round(float(sire["EBV"]), 4),
                "Dam_EBV": round(float(dam["EBV"]), 4),
                "Relationship_A": round(relationship, 4),
                "Predicted_Offspring_F_%": round(predicted_f, 4),
                "Expected_Offspring_EBV": round(expected_ebv, 4),
                "Expected_Phenotype_Base": None if pd.isna(expected_phenotype_base) else round(float(expected_phenotype_base), 4),
                "Estimated_Inbreeding_Depression": round(float(estimated_depression), 4),
                "Predicted_Phenotype_After_Depression": None if pd.isna(predicted_phenotype_after_depression) else round(float(predicted_phenotype_after_depression), 4),
                "Risk_Level": risk_level,
                "Decision": decision,
            })

    if not simulations:
        return pd.DataFrame()

    sim_df = pd.DataFrame(simulations)

    sim_df = sim_df.sort_values(
        by=["Predicted_Offspring_F_%", "Expected_Offspring_EBV", "Relationship_A"],
        ascending=[True, False, True],
    )

    return sim_df.head(max_pairs).reset_index(drop=True)





def make_pedigree_html(dot_source: str, title: str = "Pedigree Visualization") -> str:
    """
    Creates a standalone HTML file that renders Graphviz DOT using Viz.js from CDN.
    This version embeds DOT safely using JSON serialization to prevent syntax errors.
    """
    import json

    safe_title = str(title).replace("<", "&lt;").replace(">", "&gt;")
    dot_json = json.dumps(dot_source)

    filename_base = (
        safe_title.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/viz.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/full.render.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f8fafc;
            color: #0f172a;
        }}
        header {{
            padding: 18px 24px;
            background: #1e293b;
            color: white;
        }}
        header h1 {{
            margin: 0 0 6px 0;
            font-size: 22px;
        }}
        header p {{
            margin: 0;
            color: #cbd5e1;
            font-size: 14px;
        }}
        main {{
            padding: 20px;
        }}
        .toolbar {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }}
        button {{
            border: 0;
            background: #2563eb;
            color: white;
            padding: 10px 14px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
        }}
        button:hover {{
            background: #1d4ed8;
        }}
        #graph {{
            background: white;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            padding: 16px;
            overflow: auto;
            min-height: 70vh;
        }}
        #graph svg {{
            max-width: none;
            height: auto;
        }}
        .note {{
            font-size: 13px;
            color: #475569;
            margin-bottom: 12px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>{safe_title}</h1>
        <p>Standalone pedigree visualization. Use browser zoom or scroll to inspect large graphs.</p>
    </header>
    <main>
        <div class="toolbar">
            <button onclick="downloadSvg()">Download SVG</button>
            <button onclick="downloadDot()">Download DOT Source</button>
        </div>
        <div class="note">
            If the graph is very large, rendering may take time. For thousands of nodes, SVG/DOT can also be opened in external graph tools.
        </div>
        <div id="graph">Rendering graph...</div>
    </main>

    <script>
        const dotSource = {dot_json};
        let renderedSvg = "";

        const viz = new Viz();

        viz.renderSVGElement(dotSource)
            .then(function(element) {{
                renderedSvg = new XMLSerializer().serializeToString(element);
                const graph = document.getElementById("graph");
                graph.innerHTML = "";
                graph.appendChild(element);
            }})
            .catch(function(error) {{
                document.getElementById("graph").innerText = "Failed to render graph: " + error + "\\n\\nDOT preview:\\n" + dotSource.slice(0, 1000);
            }});

        function downloadSvg() {{
            if (!renderedSvg) {{
                alert("SVG is not ready yet.");
                return;
            }}
            const blob = new Blob([renderedSvg], {{type: "image/svg+xml"}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "{filename_base}.svg";
            a.click();
            URL.revokeObjectURL(url);
        }}

        function downloadDot() {{
            const blob = new Blob([dotSource], {{type: "text/vnd.graphviz"}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "{filename_base}.dot";
            a.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>
"""

def make_pedigree_report_html(graph_df: pd.DataFrame, dot_source: str, title: str = "Pedigree Report") -> str:
    """
    Creates a readable HTML report that includes graph rendering and a data table.
    """
    table_cols = [
        "Animal_ID", "Sex_Role", "Sire_ID", "Dam_ID", "Phenotype",
        "EBV", "Heterosis", "Inbreeding_%", "Classification",
        "Inbreeding_Condition", "Reproduction_Warning"
    ]
    existing_cols = [col for col in table_cols if col in graph_df.columns]
    table_html = clean_display(graph_df[existing_cols]).to_html(index=False, escape=False)

    graph_html = make_pedigree_html(dot_source, title=title)

    # Inject table before closing main tag.
    table_section = f"""
        <h2>Pedigree Data Table</h2>
        <div style="overflow:auto; background:white; border:1px solid #cbd5e1; border-radius:12px; padding:12px;">
            {table_html}
        </div>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 13px;
            }}
            th, td {{
                border: 1px solid #cbd5e1;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background: #e2e8f0;
            }}
        </style>
    """
    return graph_html.replace("</main>", table_section + "\n</main>")


def get_family_subgraph(result_df: pd.DataFrame, selected_id: str, generations_up: int = 3, generations_down: int = 2) -> pd.DataFrame:
    """
    Builds a pedigree subgraph around one selected animal:
    - ancestors up to generations_up
    - descendants up to generations_down
    """
    if not selected_id:
        return result_df.head(0).copy()

    df = result_df.copy()
    df["Animal_ID"] = df["Animal_ID"].astype(str)
    df["Sire_ID"] = df["Sire_ID"].apply(lambda x: None if is_unknown(x) else str(x))
    df["Dam_ID"] = df["Dam_ID"].apply(lambda x: None if is_unknown(x) else str(x))

    parents = {
        row["Animal_ID"]: (row["Sire_ID"], row["Dam_ID"])
        for _, row in df.iterrows()
    }

    children = {}
    for _, row in df.iterrows():
        animal = row["Animal_ID"]
        for parent in [row["Sire_ID"], row["Dam_ID"]]:
            if parent:
                children.setdefault(parent, set()).add(animal)

    keep = set([str(selected_id)])

    current = {str(selected_id)}
    for _ in range(generations_up):
        next_level = set()
        for animal in current:
            sire, dam = parents.get(animal, (None, None))
            for parent in [sire, dam]:
                if parent:
                    next_level.add(parent)
        keep.update(next_level)
        current = next_level

    current = {str(selected_id)}
    for _ in range(generations_down):
        next_level = set()
        for animal in current:
            next_level.update(children.get(animal, set()))
        keep.update(next_level)
        current = next_level

    return df[df["Animal_ID"].isin(keep)].copy()


def filter_pedigree_for_visualization(
    result_df: pd.DataFrame,
    mode: str,
    selected_id: Optional[str] = None,
    classification_filter: Optional[list] = None,
    only_inbred: bool = False,
    only_warning: bool = False,
    max_nodes: Optional[int] = None,
    generations_up: int = 3,
    generations_down: int = 2,
) -> pd.DataFrame:
    """
    Filters pedigree data for visualization while keeping full-data access through downloads.
    """
    df = result_df.copy()

    if mode == "Family subgraph around selected animal" and selected_id:
        df = get_family_subgraph(
            df,
            selected_id=selected_id,
            generations_up=generations_up,
            generations_down=generations_down,
        )

    elif mode == "High-risk animals only":
        df = df[
            (df["Inbreeding_%"] > 0) |
            (df["Reproduction_Warning"].astype(str) != "") |
            (df["Classification"].astype(str).str.contains("Final Stock", case=False, na=False))
        ].copy()

    elif mode == "Selection candidates only":
        threshold_ebv = df["EBV"].quantile(0.75)
        df = df[
            (df["EBV"] >= threshold_ebv) &
            (df["Inbreeding_%"] < 6.25)
        ].copy()

    elif mode == "Custom filter":
        if classification_filter:
            df = df[df["Classification"].isin(classification_filter)].copy()

        if only_inbred:
            df = df[df["Inbreeding_%"] > 0].copy()

        if only_warning:
            df = df[df["Reproduction_Warning"].astype(str) != ""].copy()

    if max_nodes is not None and max_nodes > 0 and len(df) > max_nodes:
        df = df.head(max_nodes).copy()

    return df


def make_dot_unlimited(result_df: pd.DataFrame, include_legend: bool = True) -> str:
    """
    Creates a Graphviz DOT pedigree graph without a hard node limit.
    For very large datasets, use download buttons or filtered rendering.
    """
    df = result_df.copy()

    if df.empty:
        return 'digraph Pedigree { label="No data to display"; }'

    df["Animal_ID"] = df["Animal_ID"].astype(str)
    animal_set = set(df["Animal_ID"].astype(str))

    dot = [
        "digraph Pedigree {",
        "rankdir=LR;",
        "splines=true;",
        "overlap=false;",
        'graph [fontname="Arial", fontsize=12, labelloc="t", label="Pedigree Chart Visualization"];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=9];',
        'edge [arrowsize=0.5, fontname="Arial", fontsize=8];',
    ]

    if include_legend:
        dot.extend([
            'subgraph cluster_legend {',
            'label="Legend";',
            'style="rounded,dashed";',
            '"legend_safe" [label="Safe / Not inbred", fillcolor="#FFFFFF"];',
            '"legend_low" [label="Inbred / Low-Moderate Risk", fillcolor="#FFF4CC"];',
            '"legend_high" [label="Very High Inbreeding", fillcolor="#FFE4E1"];',
            '"legend_elite" [label="Elite Stock Border", fillcolor="#FFFFFF", color="#FFD700", penwidth="3.0"];',
            '"legend_breeding" [label="Breeding Stock Border", fillcolor="#FFFFFF", color="#32CD32", penwidth="2.0"];',
            "}",
        ])

    for _, row in df.iterrows():
        animal = dot_escape(row["Animal_ID"])
        f_value = float(row.get("Inbreeding_%", 0))
        classification = str(row.get("Classification", ""))
        sex_role = str(row.get("Sex_Role", ""))
        warning = str(row.get("Reproduction_Warning", ""))

        fill = "#FFFFFF"
        border_color = "black"
        penwidth = "1.0"

        if f_value >= 25:
            fill = "#FFE4E1"
        elif f_value > 0:
            fill = "#FFF4CC"

        if classification == "Elite Stock":
            border_color = "#FFD700"
            penwidth = "3.0"
        elif classification == "Breeding Stock":
            border_color = "#32CD32"
            penwidth = "2.0"
        elif "Final Stock" in classification:
            border_color = "#DC2626"
            penwidth = "2.0"

        warning_text = "\\nWARNING" if warning and warning != "-" else ""

        label = (
            f"{row['Animal_ID']}\\n"
            f"{sex_role}\\n"
            f"F={f_value:.2f}% | EBV={float(row.get('EBV', 0)):.2f}\\n"
            f"{classification}{warning_text}"
        )

        dot.append(
            f'"{animal}" [label="{dot_escape(label)}", fillcolor="{fill}", color="{border_color}", penwidth="{penwidth}"];'
        )

    for _, row in df.iterrows():
        animal = dot_escape(row["Animal_ID"])
        sire = row.get("Sire_ID")
        dam = row.get("Dam_ID")

        if not is_unknown(sire) and str(sire) in animal_set:
            dot.append(f'"{dot_escape(sire)}" -> "{animal}" [label="sire", color="#2563EB"];')

        if not is_unknown(dam) and str(dam) in animal_set:
            dot.append(f'"{dot_escape(dam)}" -> "{animal}" [label="dam", color="#DB2777"];')

    dot.append("}")
    return "\n".join(dot)


def pedigree_summary_table(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a compact summary for large pedigree datasets.
    """
    df = result_df.copy()

    return pd.DataFrame({
        "Metric": [
            "Total animals",
            "Animals with known sire",
            "Animals with known dam",
            "Inbred animals",
            "High-risk animals F >= 12.5%",
            "Very high-risk animals F >= 25%",
            "Backcross / reproduction warning",
            "Elite Stock",
            "Breeding Stock",
            "Commercial",
            "Final Stock",
        ],
        "Value": [
            len(df),
            int((~df["Sire_ID"].apply(is_unknown)).sum()),
            int((~df["Dam_ID"].apply(is_unknown)).sum()),
            int((df["Inbreeding_%"] > 0).sum()),
            int((df["Inbreeding_%"] >= 12.5).sum()),
            int((df["Inbreeding_%"] >= 25).sum()),
            int((df["Reproduction_Warning"].astype(str) != "").sum()),
            int((df["Classification"] == "Elite Stock").sum()),
            int((df["Classification"] == "Breeding Stock").sum()),
            int((df["Classification"] == "Commercial").sum()),
            int((df["Classification"].astype(str).str.contains("Final Stock", case=False, na=False)).sum()),
        ],
    })


def build_pair_pool_for_pure_lines(
    result_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    h2_value: float,
    depression_rate_value: float,
    max_offspring_f: float,
) -> pd.DataFrame:
    """
    Builds all possible safe sire-dam pairings for pure line foundation.
    The safest pairs have low relationship, low predicted offspring F, and positive expected EBV.
    """
    sires, dams = infer_mating_candidates(result_df)

    rows = []

    valid_phenotypes = pd.to_numeric(result_df.get("Phenotype", pd.Series(dtype=float)), errors="coerce").dropna()
    population_avg = valid_phenotypes.mean() if not valid_phenotypes.empty else np.nan

    for _, sire in sires.iterrows():
        sire_id = str(sire["Animal_ID"])

        for _, dam in dams.iterrows():
            dam_id = str(dam["Animal_ID"])

            if sire_id == dam_id:
                continue

            if sire_id not in matrix_df.index or dam_id not in matrix_df.columns:
                continue

            relationship = float(matrix_df.loc[sire_id, dam_id])
            predicted_f = 0.5 * relationship * 100
            expected_ebv = 0.5 * (float(sire["EBV"]) + float(dam["EBV"]))

            sire_pheno = pd.to_numeric(sire.get("Phenotype", np.nan), errors="coerce")
            dam_pheno = pd.to_numeric(dam.get("Phenotype", np.nan), errors="coerce")

            if not pd.isna(sire_pheno) and not pd.isna(dam_pheno):
                expected_phenotype_base = 0.5 * (float(sire_pheno) + float(dam_pheno))
            elif not pd.isna(population_avg) and h2_value > 0:
                expected_phenotype_base = population_avg + (expected_ebv / h2_value)
            elif not pd.isna(population_avg):
                expected_phenotype_base = population_avg
            else:
                expected_phenotype_base = np.nan

            estimated_depression = calculate_inbreeding_depression(predicted_f, depression_rate_value)

            if not pd.isna(expected_phenotype_base):
                predicted_after_depression = expected_phenotype_base - estimated_depression
            else:
                predicted_after_depression = np.nan

            if predicted_f <= 0:
                risk_level = "Very Safe"
            elif predicted_f < 3.125:
                risk_level = "Very Low Risk"
            elif predicted_f <= max_offspring_f:
                risk_level = "Safe"
            elif predicted_f < 12.5:
                risk_level = "Caution"
            else:
                risk_level = "Avoid"

            safety_score = (
                (max_offspring_f - predicted_f) * 2
                + expected_ebv
                - relationship
            )

            rows.append({
                "Sire_ID": sire_id,
                "Sire_Role": sire.get("Sex_Role", "Male / Sire Candidate"),
                "Dam_ID": dam_id,
                "Dam_Role": dam.get("Sex_Role", "Female / Dam Candidate"),
                "Relationship_A": round(relationship, 4),
                "Predicted_F_%": round(predicted_f, 4),
                "Expected_EBV": round(expected_ebv, 4),
                "Expected_Phenotype_Base": None if pd.isna(expected_phenotype_base) else round(float(expected_phenotype_base), 4),
                "Estimated_Depression": round(float(estimated_depression), 4),
                "Predicted_Phenotype_After_Depression": None if pd.isna(predicted_after_depression) else round(float(predicted_after_depression), 4),
                "Risk_Level": risk_level,
                "Safety_Score": round(float(safety_score), 4),
            })

    if not rows:
        return pd.DataFrame()

    pool = pd.DataFrame(rows)
    pool = pool.sort_values(
        ["Predicted_F_%", "Expected_EBV", "Safety_Score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return pool


def select_four_safe_pure_lines(
    pair_pool: pd.DataFrame,
    max_offspring_f: float,
    required_lines: int = 4,
) -> pd.DataFrame:
    """
    Selects four foundation lines while trying to avoid repeated sires and dams.
    If the dataset is limited, the system will still provide the best available lines.
    """
    if pair_pool.empty:
        return pd.DataFrame()

    safe_pool = pair_pool[pair_pool["Predicted_F_%"] <= max_offspring_f].copy()

    if safe_pool.empty:
        safe_pool = pair_pool.copy()

    selected = []
    used_sires = set()
    used_dams = set()

    for _, row in safe_pool.iterrows():
        sire = row["Sire_ID"]
        dam = row["Dam_ID"]

        if sire in used_sires or dam in used_dams:
            continue

        selected.append(row)
        used_sires.add(sire)
        used_dams.add(dam)

        if len(selected) >= required_lines:
            break

    if len(selected) < required_lines:
        for _, row in safe_pool.iterrows():
            key = (row["Sire_ID"], row["Dam_ID"])
            existing = {(r["Sire_ID"], r["Dam_ID"]) for r in selected}

            if key in existing:
                continue

            selected.append(row)

            if len(selected) >= required_lines:
                break

    if not selected:
        return pd.DataFrame()

    selected_df = pd.DataFrame(selected).reset_index(drop=True)
    selected_df.insert(0, "Line", [f"Line {chr(65 + i)}" for i in range(len(selected_df))])

    selected_df["GGPS_Male"] = selected_df["Sire_ID"]
    selected_df["GGPS_Female"] = selected_df["Dam_ID"]
    selected_df["GGPS_Expected_F_%"] = selected_df["Predicted_F_%"]
    selected_df["GGPS_Expected_EBV"] = selected_df["Expected_EBV"]

    return selected_df


def simulate_stock_pyramid_from_lines(
    selected_lines: pd.DataFrame,
    max_offspring_f: float,
) -> pd.DataFrame:
    """
    Simulates a 4-level breeding stock pyramid:
    GGPS -> GPS -> PS -> FS.

    Conservative assumption:
    - GPS is produced from selected pure-line GGPS parents.
    - PS is produced from controlled crossing among different lines.
    - FS is final stock from commercial crossing between PS lines.

    This simulation keeps FS as terminal stock and avoids using high-risk combinations.
    """
    if selected_lines.empty:
        return pd.DataFrame()

    lines = selected_lines.reset_index(drop=True).copy()
    rows = []

    # GGPS level: foundation pure line pair
    for _, row in lines.iterrows():
        rows.append({
            "Stage": "GGPS",
            "Line_System": row["Line"],
            "Male_Source": row["GGPS_Male"],
            "Female_Source": row["GGPS_Female"],
            "Breeding_Model": "Pure line foundation",
            "Predicted_Offspring_F_%": row["GGPS_Expected_F_%"],
            "Expected_Offspring_EBV": row["GGPS_Expected_EBV"],
            "Risk_Level": "Safe" if row["GGPS_Expected_F_%"] <= max_offspring_f else "Caution",
            "Output": f"{row['Line']} GGPS replacement candidate",
            "Use": "Maintain pure line nucleus",
        })

    # GPS level: multiplication within selected line, with conservative carry-over.
    for _, row in lines.iterrows():
        gps_f = min(float(row["GGPS_Expected_F_%"]) * 1.10, 100)
        gps_ebv = float(row["GGPS_Expected_EBV"]) * 0.98

        rows.append({
            "Stage": "GPS",
            "Line_System": row["Line"],
            "Male_Source": f"{row['Line']} GGPS male line",
            "Female_Source": f"{row['Line']} GGPS female line",
            "Breeding_Model": "Pure line multiplication",
            "Predicted_Offspring_F_%": round(gps_f, 4),
            "Expected_Offspring_EBV": round(gps_ebv, 4),
            "Risk_Level": "Safe" if gps_f <= max_offspring_f else "Caution",
            "Output": f"{row['Line']} GPS candidate",
            "Use": "Supply parent-stock line",
        })

    # PS level: cross different pure lines in a controlled way.
    # Use four-line structure: A x B and C x D.
    if len(lines) >= 2:
        ps_pairs = []
        if len(lines) >= 4:
            ps_pairs = [(0, 1, "PS Male Line"), (2, 3, "PS Female Line")]
        else:
            ps_pairs = [(0, 1, "PS Composite Line")]

        for i, j, label in ps_pairs:
            left = lines.iloc[i]
            right = lines.iloc[j]
            ps_f = 0.5 * (float(left["GGPS_Expected_F_%"]) + float(right["GGPS_Expected_F_%"])) * 0.50
            ps_ebv = 0.5 * (float(left["GGPS_Expected_EBV"]) + float(right["GGPS_Expected_EBV"]))

            rows.append({
                "Stage": "PS",
                "Line_System": f"{left['Line']} × {right['Line']}",
                "Male_Source": f"{left['Line']} GPS",
                "Female_Source": f"{right['Line']} GPS",
                "Breeding_Model": "Controlled inter-line cross",
                "Predicted_Offspring_F_%": round(ps_f, 4),
                "Expected_Offspring_EBV": round(ps_ebv, 4),
                "Risk_Level": "Safe" if ps_f <= max_offspring_f else "Caution",
                "Output": label,
                "Use": "Produce parent stock for commercial crossing",
            })

    # FS level: terminal four-line commercial cross.
    if len(lines) >= 4:
        line_a = lines.iloc[0]
        line_b = lines.iloc[1]
        line_c = lines.iloc[2]
        line_d = lines.iloc[3]

        ps_male_ebv = 0.5 * (float(line_a["GGPS_Expected_EBV"]) + float(line_b["GGPS_Expected_EBV"]))
        ps_female_ebv = 0.5 * (float(line_c["GGPS_Expected_EBV"]) + float(line_d["GGPS_Expected_EBV"]))
        fs_ebv = 0.5 * (ps_male_ebv + ps_female_ebv)

        ps_male_f = 0.5 * (float(line_a["GGPS_Expected_F_%"]) + float(line_b["GGPS_Expected_F_%"])) * 0.50
        ps_female_f = 0.5 * (float(line_c["GGPS_Expected_F_%"]) + float(line_d["GGPS_Expected_F_%"])) * 0.50
        fs_f = 0.5 * (ps_male_f + ps_female_f) * 0.50

        rows.append({
            "Stage": "FS",
            "Line_System": "(Line A × Line B) × (Line C × Line D)",
            "Male_Source": "PS Male Line from Line A × Line B",
            "Female_Source": "PS Female Line from Line C × Line D",
            "Breeding_Model": "Terminal four-line cross",
            "Predicted_Offspring_F_%": round(fs_f, 4),
            "Expected_Offspring_EBV": round(fs_ebv, 4),
            "Risk_Level": "Safe" if fs_f <= max_offspring_f else "Caution",
            "Output": "Final Stock / Commercial offspring",
            "Use": "Commercial production only, not for breeding nucleus",
        })

    pyramid = pd.DataFrame(rows)
    return pyramid


def make_pure_line_flow_dot(pyramid_df: pd.DataFrame) -> str:
    """
    Creates a simple Graphviz flow for the GGPS-GPS-PS-FS structure.
    """
    dot = [
        "digraph PureLinePyramid {",
        "rankdir=TB;",
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, fillcolor="#F8FAFC"];',
        'edge [arrowsize=0.7];',
        '"GGPS" [label="GGPS\\nGreat Grand Parent Stock\\nPure line nucleus", fillcolor="#DBEAFE"];',
        '"GPS" [label="GPS\\nGrand Parent Stock\\nPure line multiplication", fillcolor="#DCFCE7"];',
        '"PS" [label="PS\\nParent Stock\\nControlled inter-line cross", fillcolor="#FEF3C7"];',
        '"FS" [label="FS\\nFinal Stock\\nTerminal commercial offspring", fillcolor="#FEE2E2"];',
        '"GGPS" -> "GPS";',
        '"GPS" -> "PS";',
        '"PS" -> "FS";',
    ]

    if not pyramid_df.empty:
        line_rows = pyramid_df[pyramid_df["Stage"] == "GGPS"]
        for _, row in line_rows.iterrows():
            line_name = str(row["Line_System"])
            safe_line = dot_escape(line_name)
            label = (
                f"{line_name}\\nMale: {row['Male_Source']}\\nFemale: {row['Female_Source']}\\n"
                f"F: {float(row['Predicted_Offspring_F_%']):.2f}%"
            )
            dot.append(
                f'"{safe_line}" [label="{dot_escape(label)}", fillcolor="#FFFFFF"];'
            )
            dot.append(f'"{safe_line}" -> "GGPS";')

    dot.append("}")
    return "\n".join(dot)



def build_breeder_summary_data(
    result_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    h2_value: float,
    intensity_value: float,
    depression_rate_value: float,
    safe_f_threshold: float = 6.25,
) -> Dict:
    """
    Builds a breeder-friendly summary from analysis results.
    """
    df = result_df.copy()

    total = len(df)
    inbred = int((df["Inbreeding_%"] > 0).sum())
    low_risk = int(((df["Inbreeding_%"] > 0) & (df["Inbreeding_%"] < 6.25)).sum())
    moderate_risk = int(((df["Inbreeding_%"] >= 6.25) & (df["Inbreeding_%"] < 12.5)).sum())
    high_risk = int(((df["Inbreeding_%"] >= 12.5) & (df["Inbreeding_%"] < 25)).sum())
    very_high_risk = int((df["Inbreeding_%"] >= 25).sum())
    warning_count = int((df["Reproduction_Warning"].astype(str) != "").sum())

    avg_f = float(df["Inbreeding_%"].mean()) if total else 0
    max_f = float(df["Inbreeding_%"].max()) if total else 0
    avg_ebv = float(df["EBV"].mean()) if total else 0
    max_ebv = float(df["EBV"].max()) if total else 0
    min_ebv = float(df["EBV"].min()) if total else 0

    class_counts = df["Classification"].value_counts().to_dict() if "Classification" in df.columns else {}

    threshold_ebv = df["EBV"].quantile(0.75) if total else 0
    threshold_low_ebv = df["EBV"].quantile(0.10) if total else 0

    selection_df = df[
        (df["EBV"] >= threshold_ebv) &
        (df["Inbreeding_%"] < safe_f_threshold) &
        (~df["Classification"].astype(str).str.contains("Final Stock", case=False, na=False))
    ].sort_values(["EBV", "Inbreeding_%"], ascending=[False, True]).copy()

    culling_df = df[
        (df["Inbreeding_%"] >= 25) |
        (df["Reproduction_Warning"].astype(str) != "") |
        (df["EBV"] <= threshold_low_ebv)
    ].sort_values(["Inbreeding_%", "EBV"], ascending=[False, True]).copy()

    valid_phenos = pd.to_numeric(df.get("Phenotype", pd.Series(dtype=float)), errors="coerce").dropna()

    phenotype_summary = {
        "available": not valid_phenos.empty,
        "average": float(valid_phenos.mean()) if not valid_phenos.empty else None,
        "sd": float(valid_phenos.std()) if len(valid_phenos) > 1 else None,
        "selection_response": None,
        "next_generation_estimate": None,
    }

    if not valid_phenos.empty and len(valid_phenos) > 1:
        sd_p = float(valid_phenos.std())
        response = calculate_selection_response(h2_value, sd_p, intensity_value)
        phenotype_summary["selection_response"] = float(response)
        phenotype_summary["next_generation_estimate"] = float(valid_phenos.mean() + response)

    hwe_res = analyze_hardy_weinberg(df)

    # Mating simulation summary
    mating_pairs = simulate_mating_pairs(
        df,
        matrix_df,
        h2_value=h2_value,
        depression_rate_value=depression_rate_value,
        max_offspring_f=safe_f_threshold,
        max_pairs=20,
    )

    best_pair = mating_pairs.iloc[0].to_dict() if not mating_pairs.empty else None

    # Pure line summary
    pair_pool = build_pair_pool_for_pure_lines(
        df,
        matrix_df,
        h2_value=h2_value,
        depression_rate_value=depression_rate_value,
        max_offspring_f=safe_f_threshold,
    )
    selected_lines = select_four_safe_pure_lines(
        pair_pool,
        max_offspring_f=safe_f_threshold,
        required_lines=4,
    ) if not pair_pool.empty else pd.DataFrame()

    pyramid_df = simulate_stock_pyramid_from_lines(
        selected_lines,
        max_offspring_f=safe_f_threshold,
    ) if not selected_lines.empty else pd.DataFrame()

    if very_high_risk > 0 or warning_count > 0:
        overall_status = "High Attention Required"
        conclusion = (
            "The population contains very high inbreeding risk or close-relative mating warnings. "
            "Breeding decisions should be controlled carefully. Avoid using high-risk animals as parents and prioritize unrelated mating."
        )
    elif avg_f >= 6.25 or high_risk > 0:
        overall_status = "Moderate Genetic Risk"
        conclusion = (
            "The population shows moderate genetic risk. Selection can continue, but every mating decision should be checked using the relationship matrix."
        )
    elif len(selection_df) > 0:
        overall_status = "Good Breeding Potential"
        conclusion = (
            "The population has usable breeding candidates with acceptable inbreeding levels. "
            "Use high-EBV and low-F animals while maintaining sire rotation and unrelated mating."
        )
    else:
        overall_status = "Limited Breeding Potential"
        conclusion = (
            "The population does not show strong breeding candidates under the current criteria. "
            "Improve recording quality, introduce unrelated genetics, or expand the breeding population."
        )

    recommendations = []

    if len(selection_df) > 0:
        recommendations.append("Use the top selection candidates as priority parents because they combine higher EBV and lower inbreeding risk.")
    else:
        recommendations.append("No strong selection candidates were detected. Consider adding unrelated sires or improving phenotype records.")

    if len(culling_df) > 0:
        recommendations.append("Do not prioritize culling candidates as breeding stock. Use them for commercial production or remove them from the breeding nucleus.")

    if warning_count > 0:
        recommendations.append("Immediately avoid sire-daughter, parent-offspring, and close-relative mating shown by reproduction warnings.")

    if best_pair:
        recommendations.append(
            f"Best simulated mating pair: male/sire {best_pair['Suggested_Sire']} with female/dam {best_pair['Suggested_Dam']}."
        )

    if not selected_lines.empty and len(selected_lines) >= 4:
        recommendations.append("A four-line GGPS-GPS-PS-FS structure can be simulated from the available data, but it should still be validated with larger founder numbers.")
    else:
        recommendations.append("A complete four-line pure-line pyramid needs more unrelated male and female founders for safer implementation.")

    recommendations.append("Recalculate EBV, inbreeding, relationship matrix, and mating simulation after every new generation.")

    return {
        "total": total,
        "inbred": inbred,
        "low_risk": low_risk,
        "moderate_risk": moderate_risk,
        "high_risk": high_risk,
        "very_high_risk": very_high_risk,
        "warning_count": warning_count,
        "avg_f": avg_f,
        "max_f": max_f,
        "avg_ebv": avg_ebv,
        "max_ebv": max_ebv,
        "min_ebv": min_ebv,
        "class_counts": class_counts,
        "selection_df": selection_df,
        "culling_df": culling_df,
        "phenotype_summary": phenotype_summary,
        "hwe_res": hwe_res,
        "mating_pairs": mating_pairs,
        "best_pair": best_pair,
        "selected_lines": selected_lines,
        "pyramid_df": pyramid_df,
        "overall_status": overall_status,
        "conclusion": conclusion,
        "recommendations": recommendations,
    }


def generate_breeder_summary_pdf(
    result_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    h2_value: float,
    intensity_value: float,
    depression_rate_value: float,
    safe_f_threshold: float = 6.25,
    farm_name: str = "Farm / Breeding Unit",
    breeder_name: str = "Breeder / Manager",
    species: str = "Livestock",
    report_notes: str = "",
):
    """
    Creates a complete official breeder-friendly PDF report.

    Report sections:
    - cover page
    - farm name
    - breeder name
    - analysis date
    - species
    - population summary
    - charts
    - top candidates
    - mating plan
    - pure line plan
    - risk warning
    - final recommendation
    - signature section
    """
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=28,
        leftMargin=28,
        topMargin=28,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()
    elements = []

    summary = build_breeder_summary_data(
        result_df,
        matrix_df,
        h2_value=h2_value,
        intensity_value=intensity_value,
        depression_rate_value=depression_rate_value,
        safe_f_threshold=safe_f_threshold,
    )

    # Local imports keep compatibility if reportlab is not installed.
    from reportlab.platypus import PageBreak
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.lib.units import inch

    title_style = styles["Heading1"]
    title_style.alignment = 1

    subtitle_style = styles["Heading2"]
    subtitle_style.alignment = 1

    normal_center = styles["Normal"].clone("NormalCenter")
    normal_center.alignment = 1

    small_style = styles["Normal"].clone("SmallStyle")
    small_style.fontSize = 8
    small_style.leading = 10

    analysis_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # ============================================================
    # COVER PAGE
    # ============================================================
    elements.append(Spacer(1, 80))
    elements.append(Paragraph("OFFICIAL BREEDING ANALYSIS REPORT", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Pedigree, Inbreeding, EBV, Mating, and Pure-Line Evaluation", subtitle_style))
    elements.append(Spacer(1, 40))

    cover_data = [
        ["Farm / Breeding Unit", farm_name],
        ["Breeder / Manager", breeder_name],
        ["Species", species],
        ["Analysis Date", analysis_date],
        ["Safe F Threshold", f"{safe_f_threshold:.2f}%"],
        ["Heritability (h²)", f"{h2_value:.2f}"],
        ["Selection Intensity (i)", f"{intensity_value:.2f}"],
        ["Depression Rate", f"{depression_rate_value:.2f} per 1% F"],
    ]

    cover_table = Table(cover_data, colWidths=[170, 280])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
    ]))
    elements.append(cover_table)
    elements.append(Spacer(1, 45))

    elements.append(Paragraph(
        "This report is generated as a decision-support document for livestock breeding management. "
        "It summarizes pedigree quality, genetic risk, estimated breeding value, mating recommendations, "
        "and pure-line pyramid readiness based on the uploaded dataset.",
        normal_center,
    ))

    if report_notes:
        elements.append(Spacer(1, 18))
        elements.append(Paragraph("<b>Report Notes</b>", styles["Heading3"]))
        elements.append(Paragraph(str(report_notes), styles["Normal"]))

    elements.append(Spacer(1, 80))
    elements.append(Paragraph(
        "Generated by Breeding & Inbreeding Analytics System",
        normal_center,
    ))

    elements.append(PageBreak())

    # ============================================================
    # EXECUTIVE SUMMARY
    # ============================================================
    elements.append(Paragraph("1. Executive Conclusion", styles["Heading1"]))
    elements.append(Paragraph(f"<b>Overall Status:</b> {summary['overall_status']}", styles["Normal"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(summary["conclusion"], styles["Normal"]))
    elements.append(Spacer(1, 10))

    if summary["warning_count"] > 0 or summary["very_high_risk"] > 0:
        warning_text = (
            f"<b>Risk Warning:</b> The dataset contains {summary['warning_count']} reproduction warning(s) "
            f"and {summary['very_high_risk']} animal(s) with very high inbreeding risk. "
            "These animals should not be prioritized as breeding nucleus candidates."
        )
        elements.append(Paragraph(warning_text, styles["Normal"]))
        elements.append(Spacer(1, 10))

    # ============================================================
    # POPULATION SUMMARY
    # ============================================================
    elements.append(Paragraph("2. Population Summary", styles["Heading1"]))

    pop_data = [
        ["Parameter", "Value"],
        ["Farm / Breeding Unit", farm_name],
        ["Breeder / Manager", breeder_name],
        ["Species", species],
        ["Analysis Date", analysis_date],
        ["Total animals", str(summary["total"])],
        ["Inbred animals", str(summary["inbred"])],
        ["Average inbreeding F", f"{summary['avg_f']:.2f}%"],
        ["Highest inbreeding F", f"{summary['max_f']:.2f}%"],
        ["Average EBV", f"{summary['avg_ebv']:.4f}"],
        ["Highest EBV", f"{summary['max_ebv']:.4f}"],
        ["Lowest EBV", f"{summary['min_ebv']:.4f}"],
        ["Reproduction warnings", str(summary["warning_count"])],
    ]

    pop_table = Table(pop_data, colWidths=[220, 260], repeatRows=1)
    pop_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(pop_table)
    elements.append(Spacer(1, 12))

    # ============================================================
    # CHARTS
    # ============================================================
    elements.append(Paragraph("3. Charts", styles["Heading1"]))

    chart_wrap = Table([["Inbreeding Risk Distribution", "Classification Distribution"]], colWidths=[240, 240])
    chart_wrap.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    elements.append(chart_wrap)
    elements.append(Spacer(1, 4))

    # Bar chart: inbreeding risk
    risk_labels = ["Low", "Moderate", "High", "Very High"]
    risk_values = [
        summary["low_risk"],
        summary["moderate_risk"],
        summary["high_risk"],
        summary["very_high_risk"],
    ]

    risk_drawing = Drawing(230, 160)
    risk_chart = VerticalBarChart()
    risk_chart.x = 30
    risk_chart.y = 30
    risk_chart.height = 100
    risk_chart.width = 170
    risk_chart.data = [risk_values]
    risk_chart.categoryAxis.categoryNames = risk_labels
    risk_chart.valueAxis.valueMin = 0
    risk_chart.valueAxis.valueMax = max(risk_values + [1])
    risk_chart.valueAxis.valueStep = max(1, int(max(risk_values + [1]) / 5) or 1)
    risk_chart.bars[0].fillColor = colors.HexColor("#2563eb")
    risk_chart.categoryAxis.labels.fontSize = 7
    risk_chart.valueAxis.labels.fontSize = 7
    risk_drawing.add(risk_chart)

    # Pie chart: classification
    class_counts = summary["class_counts"]
    class_drawing = Drawing(230, 160)
    if class_counts:
        pie = Pie()
        pie.x = 55
        pie.y = 25
        pie.width = 110
        pie.height = 110
        pie.data = list(class_counts.values())
        pie.labels = [str(k)[:14] for k in class_counts.keys()]
        pie.slices.strokeWidth = 0.5
        class_drawing.add(pie)
    else:
        # Fallback empty chart text is avoided due to reportlab drawing complexity.
        pass

    charts_table = Table([[risk_drawing, class_drawing]], colWidths=[240, 240])
    charts_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))
    elements.append(charts_table)
    elements.append(Spacer(1, 12))

    # ============================================================
    # RISK SUMMARY
    # ============================================================
    elements.append(Paragraph("4. Risk Warning and Inbreeding Distribution", styles["Heading1"]))

    risk_data = [
        ["Risk Category", "Count", "Interpretation"],
        ["Low risk: 0 < F < 6.25%", str(summary["low_risk"]), "Generally safe, monitor across generations."],
        ["Moderate risk: 6.25% <= F < 12.5%", str(summary["moderate_risk"]), "Use caution and avoid repeated related mating."],
        ["High risk: 12.5% <= F < 25%", str(summary["high_risk"]), "Not preferred for breeding nucleus."],
        ["Very high risk: F >= 25%", str(summary["very_high_risk"]), "Avoid as breeding stock."],
        ["Reproduction warning", str(summary["warning_count"]), "Check sire-daughter or close-relative mating."],
    ]

    risk_table = Table(risk_data, colWidths=[170, 60, 250], repeatRows=1)
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#991b1b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 10))

    # ============================================================
    # CLASSIFICATION SUMMARY
    # ============================================================
    elements.append(Paragraph("5. Classification Summary", styles["Heading1"]))

    class_rows = [["Classification", "Count"]]
    for label, count in summary["class_counts"].items():
        class_rows.append([str(label), str(count)])

    if len(class_rows) == 1:
        class_rows.append(["No classification available", "0"])

    class_table = Table(class_rows, colWidths=[300, 150], repeatRows=1)
    class_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(class_table)
    elements.append(Spacer(1, 10))

    # ============================================================
    # PHENOTYPE AND SELECTION RESPONSE
    # ============================================================
    elements.append(Paragraph("6. Phenotype and Selection Response", styles["Heading1"]))

    phenotype = summary["phenotype_summary"]
    if phenotype["available"]:
        phenotype_rows = [
            ["Parameter", "Value"],
            ["Average phenotype", f"{phenotype['average']:.4f}" if phenotype["average"] is not None else "-"],
            ["Phenotype standard deviation", f"{phenotype['sd']:.4f}" if phenotype["sd"] is not None else "-"],
            ["Selection response R", f"{phenotype['selection_response']:.4f}" if phenotype["selection_response"] is not None else "-"],
            ["Next generation estimate", f"{phenotype['next_generation_estimate']:.4f}" if phenotype["next_generation_estimate"] is not None else "-"],
        ]
        phenotype_table = Table(phenotype_rows, colWidths=[250, 220], repeatRows=1)
        phenotype_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16a34a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        elements.append(phenotype_table)
    else:
        elements.append(Paragraph(
            "Phenotype data is not available or not valid, so selection response could not be estimated.",
            styles["Normal"],
        ))
    elements.append(Spacer(1, 10))

    # ============================================================
    # TOP CANDIDATES
    # ============================================================
    elements.append(Paragraph("7. Top Breeding Candidates", styles["Heading1"]))

    selection_df = summary["selection_df"].head(15)
    if not selection_df.empty:
        selection_rows = [["Animal ID", "Sex/Role", "EBV", "F (%)", "Classification"]]
        for _, row in selection_df.iterrows():
            selection_rows.append([
                str(row["Animal_ID"]),
                str(row.get("Sex_Role", "-"))[:24],
                f"{row['EBV']:.4f}",
                f"{row['Inbreeding_%']:.2f}",
                str(row["Classification"])[:22],
            ])
        selection_table = Table(selection_rows, colWidths=[85, 145, 70, 60, 120], repeatRows=1)
        selection_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16a34a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(selection_table)
    else:
        elements.append(Paragraph("No strong selection candidates were detected under the current criteria.", styles["Normal"]))
    elements.append(Spacer(1, 10))

    # ============================================================
    # PRIORITY CULLING / NON-BREEDING
    # ============================================================
    elements.append(Paragraph("8. Priority Non-Breeding / Culling Candidates", styles["Heading1"]))

    culling_df = summary["culling_df"].head(15)
    if not culling_df.empty:
        culling_rows = [["Animal ID", "EBV", "F (%)", "Warning", "Classification"]]
        for _, row in culling_df.iterrows():
            warning = str(row.get("Reproduction_Warning", "-"))
            if not warning:
                warning = "-"
            culling_rows.append([
                str(row["Animal_ID"]),
                f"{row['EBV']:.4f}",
                f"{row['Inbreeding_%']:.2f}",
                warning[:30],
                str(row["Classification"])[:20],
            ])
        culling_table = Table(culling_rows, colWidths=[85, 65, 60, 170, 100], repeatRows=1)
        culling_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dc2626")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(culling_table)
    else:
        elements.append(Paragraph("No priority culling candidates were detected.", styles["Normal"]))
    elements.append(Spacer(1, 10))

    # ============================================================
    # MATING PLAN
    # ============================================================
    elements.append(Paragraph("9. Mating Plan", styles["Heading1"]))

    best_pair = summary["best_pair"]
    if best_pair:
        mating_text = (
            f"Best simulated pair: male/sire <b>{best_pair['Suggested_Sire']}</b> with "
            f"female/dam <b>{best_pair['Suggested_Dam']}</b>. "
            f"Predicted offspring F: <b>{best_pair['Predicted_Offspring_F_%']:.2f}%</b>. "
            f"Expected offspring EBV: <b>{best_pair['Expected_Offspring_EBV']:.4f}</b>. "
            f"Risk level: <b>{best_pair['Risk_Level']}</b>. "
            f"Decision: <b>{best_pair['Decision']}</b>."
        )
        elements.append(Paragraph(mating_text, styles["Normal"]))
        elements.append(Spacer(1, 6))

        mating_pairs = summary["mating_pairs"].head(10)
        if not mating_pairs.empty:
            mating_rows = [["Male/Sire", "Female/Dam", "F (%)", "Expected EBV", "Decision"]]
            for _, row in mating_pairs.iterrows():
                mating_rows.append([
                    str(row["Suggested_Sire"]),
                    str(row["Suggested_Dam"]),
                    f"{row['Predicted_Offspring_F_%']:.2f}",
                    f"{row['Expected_Offspring_EBV']:.4f}",
                    str(row["Decision"])[:28],
                ])
            mating_table = Table(mating_rows, colWidths=[90, 90, 60, 90, 150], repeatRows=1)
            mating_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            elements.append(mating_table)
    else:
        elements.append(Paragraph("No mating simulation could be generated from the current dataset.", styles["Normal"]))
    elements.append(Spacer(1, 10))

    # ============================================================
    # PURE LINE PLAN
    # ============================================================
    elements.append(Paragraph("10. Pure Line Plan: GGPS, GPS, PS, and FS", styles["Heading1"]))

    selected_lines = summary["selected_lines"]
    if not selected_lines.empty:
        line_rows = [["Line", "GGPS Male/Sire", "GGPS Female/Dam", "F (%)", "Expected EBV"]]
        for _, row in selected_lines.head(4).iterrows():
            line_rows.append([
                str(row["Line"]),
                str(row["GGPS_Male"]),
                str(row["GGPS_Female"]),
                f"{row['GGPS_Expected_F_%']:.2f}",
                f"{row['GGPS_Expected_EBV']:.4f}",
            ])
        line_table = Table(line_rows, colWidths=[55, 120, 120, 60, 100], repeatRows=1)
        line_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        elements.append(line_table)
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(
            "Recommended structure: keep GGPS/GPS as pure-line nucleus and multiplication stock, "
            "use PS as controlled inter-line parent stock, and use FS as terminal commercial stock only.",
            styles["Normal"],
        ))
    else:
        elements.append(Paragraph(
            "A complete pure-line pyramid could not be generated. More unrelated male and female founders are recommended.",
            styles["Normal"],
        ))
    elements.append(Spacer(1, 10))

    # ============================================================
    # FINAL RECOMMENDATION
    # ============================================================
    elements.append(Paragraph("11. Final Recommendation", styles["Heading1"]))

    for idx, rec in enumerate(summary["recommendations"], start=1):
        elements.append(Paragraph(f"{idx}. {rec}", styles["Normal"]))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "Final note: this report is a decision-support output based on available pedigree and phenotype data. "
        "Field health, fertility, management, environment, and breeder objectives must still be considered before implementation.",
        styles["Normal"],
    ))

    # ============================================================
    # SIGNATURE SECTION
    # ============================================================
    elements.append(PageBreak())
    elements.append(Paragraph("12. Signature Section", styles["Heading1"]))
    elements.append(Paragraph(
        "This section can be completed after the report has been reviewed by the breeder, farm manager, or breeding consultant.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 40))

    signature_data = [
        ["Prepared by", "Reviewed by", "Approved by"],
        ["", "", ""],
        ["Name:", "Name:", "Name:"],
        ["Position:", "Position:", "Position:"],
        ["Date:", "Date:", "Date:"],
        ["Signature:", "Signature:", "Signature:"],
    ]

    signature_table = Table(signature_data, colWidths=[160, 160, 160], rowHeights=[24, 70, 24, 24, 24, 55])
    signature_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.7, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(signature_table)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        f"Report generated for {farm_name} | Species: {species} | Analysis date: {analysis_date}",
        small_style,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer



def create_sample_excel_download() -> bytes:
    """
    Creates an XLSX file from the built-in Full Cattle Example sample.
    This allows users to download the sample directly from the application.
    """
    buffer = io.BytesIO()

    sample_df = contoh_sapi_lengkap()

    rules_df = pd.DataFrame({
        "Rule Category": [
            "Required Column",
            "Required Column",
            "Required Column",
            "Sex",
            "Age",
            "Line",
            "Generation",
            "Phenotype",
            "Multi-Trait",
            "Health",
            "Reproductive Status",
            "FS Rule",
            "Missing Parent",
            "Avoid",
        ],
        "Rule / Explanation": [
            "Animal_ID must be unique and not empty.",
            "Sire_ID is the sire/father. Use '-' if unknown.",
            "Dam_ID is the dam/mother. Use '-' if unknown.",
            "Use Male/Female, Male/Female, M/F, or L/P.",
            "Use a numeric value. Unit depends on your species/project.",
            "Use consistent line names such as Line A, Line B, Line C, Line D.",
            "Use values such as Founder, GGPS, GPS, PS, FS, G0, G1, G2.",
            "Use numeric values only. This column is used for basic EBV calculation.",
            "Body_Weight, Growth_Rate, Fertility, Survival_Rate, and Feed_Efficiency can be selected for Multi-Trait Selection Index.",
            "Use Health_Status and Defect_Status to prevent unhealthy animals from being recommended.",
            "Use Ready, Pregnant, Terminal, Infertile, Hold, or other clear labels.",
            "FS is terminal stock and should not be used as breeding nucleus.",
            "Parent IDs not found as Animal_ID will be treated as additional founders.",
            "Do not duplicate Animal_ID. Do not make an animal its own parent.",
        ],
    })

    mapping_df = pd.DataFrame({
        "Application Mapping": [
            "Animal_ID Column",
            "Sire_ID Column",
            "Dam_ID Column",
            "Phenotype Column",
            "Sex Column",
            "Age Column",
            "Line Column",
            "Generation Column",
            "Additional Trait Columns",
            "Reproductive Status Column",
            "Health Status Column",
            "Defect Status Column",
            "Culling Reason Column",
            "Pen Column",
            "Cage Column",
            "Batch Column",
            "Mating Date Column",
            "Expected Offspring Date Column",
            "Responsible Person Column",
            "Genomic EBV Column",
            "Genomic Reliability Column",
            "Feed Cost Column",
            "Production Value Column",
            "Replacement Cost Column",
            "Culling Value Column",
        ],
        "Select This Column": [
            "Animal_ID",
            "Sire_ID",
            "Dam_ID",
            "Phenotype",
            "Sex",
            "Age",
            "Line",
            "Generation",
            "Body_Weight, Growth_Rate, Fertility, Survival_Rate, Feed_Efficiency",
            "Reproductive_Status",
            "Health_Status",
            "Defect_Status",
            "Culling_Reason",
            "Pen",
            "Cage",
            "Batch",
            "Mating_Date",
            "Expected_Offspring_Date",
            "Responsible_Person",
            "Genomic_EBV",
            "Genomic_Reliability",
            "Feed_Cost",
            "Production_Value",
            "Replacement_Cost",
            "Culling_Value",
        ],
    })

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        sample_df.to_excel(writer, index=False, sheet_name="Pedigree_Data")
        rules_df.to_excel(writer, index=False, sheet_name="Data_Rules")
        mapping_df.to_excel(writer, index=False, sheet_name="Column_Mapping")

    buffer.seek(0)
    return buffer.getvalue()


def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-main: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f1f5f9;
            --text-sub: #94a3b8;
            --accent-primary: #3b82f6;
            --border-color: #334155;
            --header-gradient: linear-gradient(90deg, #1e293b 0%, #334155 100%);
        }

        @media (prefers-color-scheme: light) {
            :root {
                --bg-main: #f1f5f9;
                --card-bg: #ffffff;
                --text-main: #1e293b;
                --text-sub: #64748b;
                --border-color: #e2e8f0;
            }
        }

        .main {
            background-color: var(--bg-main) !important;
            font-family: 'Inter', sans-serif;
            color: var(--text-main) !important;
        }

        h1, h2, h3, h4, h5, h6, p, span, label {
            color: var(--text-main) !important;
        }

        .custom-header {
            background: var(--header-gradient);
            padding: 2rem 2rem;
            border-radius: 0 0 2rem 2rem;
            margin: -5rem -4rem 2rem -4rem;
            color: white !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            text-align: center;
        }

        .custom-header * {
            color: white !important;
        }

        .info-card {
            background: var(--card-bg) !important;
            color: var(--text-main) !important;
            padding: 1rem;
            border-radius: 1rem;
            border-left: 5px solid var(--accent-primary);
            border: 1px solid var(--border-color);
            margin: 1rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        div[data-testid="stMetric"] {
            background: var(--card-bg) !important;
            padding: 1.2rem;
            border-radius: 1rem;
            border: 1px solid var(--border-color) !important;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--border-color) !important;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
        <div class="custom-header">
            <h1 style="font-weight: 800; font-size: 3rem; margin-bottom: 0;">
                BREEDING & INBREEDING ANALYTICS
            </h1>
            <p style="font-size: 1.15rem; font-weight: 400; letter-spacing: 0.04em;">
                A decision support system for livestock genetic evaluation based on pedigree records,
                inbreeding coefficient, Estimated Breeding Value (EBV), heterosis, and genetic relationship analysis.
                <br>
                This application helps breeders, researchers, and farm managers identify superior breeding candidates,
                avoid close-relative mating, reduce inbreeding risk, and design sustainable breeding strategies.
            </p>
        </div>
    """, unsafe_allow_html=True)




def parse_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def evaluate_age_eligibility(result_df: pd.DataFrame, min_age: float, max_age: float) -> pd.DataFrame:
    """
    Uses optional Age column to classify breeding age eligibility.
    """
    df = result_df.copy()

    if "Age" not in df.columns:
        df["Age_Eligibility"] = "Age data not available"
        return df

    ages = pd.to_numeric(df["Age"], errors="coerce")

    def classify(age):
        if pd.isna(age):
            return "Age data not available"
        if age < min_age:
            return "Too Young"
        if age <= max_age:
            return "Ready / Productive"
        return "Old / Replace Soon"

    df["Age_Eligibility"] = ages.apply(classify)
    return df


def evaluate_reproductive_status(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses optional reproductive columns to classify reproductive readiness.
    """
    df = result_df.copy()

    if "Reproductive_Status" not in df.columns:
        df["Reproductive_Decision"] = "Reproductive data not available"
        return df

    def classify(value):
        if is_unknown(value):
            return "Reproductive data not available"
        text = str(value).lower()
        bad = ["infertile", "sterile", "culled", "not ready", "too young", "old", "abnormal", "failed"]
        hold = ["pregnant", "incubating", "recovering", "rest", "treatment"]
        good = ["ready", "fertile", "active", "productive", "laying", "breeding"]

        if any(k in text for k in bad):
            return "Do Not Breed"
        if any(k in text for k in hold):
            return "Hold / Monitor"
        if any(k in text for k in good):
            return "Ready for Breeding"
        return "Check Reproductive Status"

    df["Reproductive_Decision"] = df["Reproductive_Status"].apply(classify)
    return df


def evaluate_health_and_defect(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses optional health and defect records to flag animals that should not be used as breeding stock.
    """
    df = result_df.copy()

    def health_decision(row):
        health = str(row.get("Health_Status", "")).lower()
        defect = str(row.get("Defect_Status", "")).lower()
        culling = str(row.get("Culling_Reason", "")).lower()

        severe_words = [
            "defect", "deform", "lame", "chronic", "disease", "sick",
            "blind", "weak", "abnormal", "culled", "dead", "mortality",
            "reproductive disorder", "not for breeding"
        ]

        if any(k in health for k in severe_words) or any(k in defect for k in severe_words) or any(k in culling for k in severe_words):
            return "Do Not Breed - Health/Defect Risk"

        if health.strip() in ["", "-", "nan", "none"] and defect.strip() in ["", "-", "nan", "none"]:
            return "Health data not available"

        if "healthy" in health or "normal" in health or "clear" in defect:
            return "Health OK"

        return "Check Health Record"

    df["Health_Defect_Decision"] = df.apply(health_decision, axis=1)
    return df


def calculate_ebv_reliability(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple EBV reliability proxy based on data completeness and progeny count.
    This is not a formal BLUP reliability, but it helps breeders know whether EBV is supported by enough data.
    """
    df = result_df.copy()

    child_counts = {}
    for _, row in df.iterrows():
        for parent_col in ["Sire_ID", "Dam_ID"]:
            parent = row.get(parent_col)
            if not is_unknown(parent):
                child_counts[str(parent)] = child_counts.get(str(parent), 0) + 1

    def score(row):
        s = 0
        if not is_unknown(row.get("Sire_ID")):
            s += 20
        if not is_unknown(row.get("Dam_ID")):
            s += 20
        if not is_unknown(row.get("Phenotype")):
            s += 20
        if not is_unknown(row.get("Sex")):
            s += 10
        if not is_unknown(row.get("Generation")):
            s += 10
        progeny = child_counts.get(str(row.get("Animal_ID")), 0)
        s += min(20, progeny * 5)
        return min(100, s)

    df["EBV_Reliability_%"] = df.apply(score, axis=1)

    def grade(v):
        if v >= 80:
            return "High"
        if v >= 50:
            return "Medium"
        return "Low"

    df["EBV_Reliability_Level"] = df["EBV_Reliability_%"].apply(grade)
    return df


def apply_genomic_support(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds genomic support when optional genomic columns are available.
    """
    df = result_df.copy()

    if "Genomic_EBV" not in df.columns:
        df["Genomic_Status"] = "Genomic data not available"
        df["Combined_Genetic_Value"] = pd.to_numeric(df.get("EBV", 0), errors="coerce").fillna(0)
        return df

    ebv = pd.to_numeric(df.get("EBV", 0), errors="coerce").fillna(0)
    gebv = pd.to_numeric(df["Genomic_EBV"], errors="coerce")

    if "Genomic_Reliability" in df.columns:
        grel = pd.to_numeric(df["Genomic_Reliability"], errors="coerce").fillna(50) / 100
        grel = grel.clip(0, 1)
    else:
        grel = pd.Series([0.5] * len(df), index=df.index)

    df["Combined_Genetic_Value"] = ((1 - grel) * ebv + grel * gebv.fillna(ebv)).round(4)
    df["Genomic_Status"] = np.where(gebv.notna(), "Genomic data used", "Genomic data missing")
    return df


def calculate_generation_trend(result_df: pd.DataFrame) -> pd.DataFrame:
    if "Generation" not in result_df.columns:
        return pd.DataFrame()

    df = result_df.copy()
    df = df[~df["Generation"].apply(is_unknown)].copy()

    if df.empty:
        return pd.DataFrame()

    trend = df.groupby("Generation").agg(
        Total=("Animal_ID", "count"),
        Average_F=("Inbreeding_%", "mean"),
        Average_EBV=("EBV", "mean"),
        Average_Selection_Index=("Selection_Index", "mean") if "Selection_Index" in df.columns else ("EBV", "mean"),
        Elite_Count=("Classification", lambda x: int((x == "Elite Stock").sum())),
        Warning_Count=("Reproduction_Warning", lambda x: int((x.astype(str) != "").sum())),
    ).reset_index()

    return trend.round(4)


def calculate_line_performance(result_df: pd.DataFrame) -> pd.DataFrame:
    if "Line" not in result_df.columns:
        return pd.DataFrame()

    df = result_df.copy()
    df = df[~df["Line"].apply(is_unknown)].copy()

    if df.empty:
        return pd.DataFrame()

    line = df.groupby("Line").agg(
        Total=("Animal_ID", "count"),
        Average_F=("Inbreeding_%", "mean"),
        Average_EBV=("EBV", "mean"),
        Average_Selection_Index=("Selection_Index", "mean") if "Selection_Index" in df.columns else ("EBV", "mean"),
        Elite_Count=("Classification", lambda x: int((x == "Elite Stock").sum())),
        Final_Stock_Count=("Classification", lambda x: int((x.astype(str).str.contains("Final Stock", case=False, na=False)).sum())),
    ).reset_index()

    def line_recommendation(row):
        if row["Average_F"] >= 12.5:
            return "Reduce inbreeding / introduce unrelated genetics"
        if row["Average_Selection_Index"] >= line["Average_Selection_Index"].quantile(0.75):
            return "Priority line for selection"
        if row["Average_EBV"] >= line["Average_EBV"].mean():
            return "Good line, maintain and monitor"
        return "Use carefully or improve through selection"

    line["Line_Recommendation"] = line.apply(line_recommendation, axis=1)
    return line.round(4)


def calculate_economic_projection(result_df: pd.DataFrame) -> pd.DataFrame:
    df = result_df.copy()

    has_any = any(col in df.columns for col in ["Feed_Cost", "Production_Value", "Replacement_Cost", "Culling_Value"])
    if not has_any:
        df["Expected_Profit"] = np.nan
        df["Economic_Status"] = "Economic data not available"
        return df

    production = parse_numeric_series(df, "Production_Value").fillna(0)
    feed = parse_numeric_series(df, "Feed_Cost").fillna(0)
    replacement = parse_numeric_series(df, "Replacement_Cost").fillna(0)
    culling = parse_numeric_series(df, "Culling_Value").fillna(0)

    df["Expected_Profit"] = (production + culling - feed - replacement).round(4)

    def econ_status(v):
        if pd.isna(v):
            return "Economic data not available"
        if v > 0:
            return "Positive projected value"
        if v == 0:
            return "Neutral projected value"
        return "Negative projected value"

    df["Economic_Status"] = df["Expected_Profit"].apply(econ_status)
    return df


def assign_final_breeding_decision(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a practical breeder-friendly decision label.
    """
    df = result_df.copy()

    def decision(row):
        classification = str(row.get("Classification", ""))
        f = float(row.get("Inbreeding_%", 0))
        ebv = float(row.get("EBV", 0))
        age_decision = str(row.get("Age_Eligibility", ""))
        repro = str(row.get("Reproductive_Decision", ""))
        health = str(row.get("Health_Defect_Decision", ""))
        gen = str(row.get("Generation", "")).upper()
        warning = str(row.get("Reproduction_Warning", ""))

        if "Do Not Breed" in health:
            return "Do Not Breed - Health/Defect"
        if "Do Not Breed" in repro:
            return "Do Not Breed - Reproductive"
        if gen == "FS":
            return "Terminal FS Only"
        if warning:
            return "Do Not Breed - Relationship Warning"
        if f >= 25:
            return "Cull / Commercial Only"
        if "Too Young" in age_decision:
            return "Hold - Too Young"
        if "Old" in age_decision:
            return "Replace Soon"
        if classification == "Elite Stock" and ebv > 0 and f < 6.25:
            return "Use as Elite Breeder"
        if classification == "Breeding Stock" and f < 6.25:
            return "Use as Replacement Breeder"
        if f < 12.5 and ebv > 0:
            return "Use with Monitoring"
        if "Final Stock" in classification:
            return "Commercial / Slaughter Only"
        return "Need More Data"

    df["Final_Breeding_Decision"] = df.apply(decision, axis=1)
    return df


def generate_next_generation_ids(prefix: str, generation: str, count: int) -> pd.DataFrame:
    rows = []
    for i in range(1, int(count) + 1):
        rows.append({
            "Proposed_Offspring_ID": f"{prefix}_{generation}_{i:04d}",
            "Generation": generation,
            "Status": "Reserved ID"
        })
    return pd.DataFrame(rows)


def build_mating_calendar(pair_df: pd.DataFrame, start_date, interval_days: int = 7, expected_days: int = 21) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()

    start = pd.to_datetime(start_date)
    rows = []
    for i, row in pair_df.head(20).iterrows():
        mating_date = start + pd.Timedelta(days=int(i) * interval_days)
        expected_date = mating_date + pd.Timedelta(days=expected_days)
        rows.append({
            "Mating_ID": f"MATE_{i+1:04d}",
            "Male_Sire": row.get("Suggested_Sire"),
            "Female_Dam": row.get("Suggested_Dam"),
            "Mating_Date": mating_date.strftime("%Y-%m-%d"),
            "Expected_Offspring_Date": expected_date.strftime("%Y-%m-%d"),
            "Predicted_Offspring_F_%": row.get("Predicted_Offspring_F_%"),
            "Expected_Offspring_EBV": row.get("Expected_Offspring_EBV"),
            "Decision": row.get("Decision"),
            "Rule_Status": row.get("Mating_Rule_Status", "-"),
        })
    return pd.DataFrame(rows)


def summarize_missing_advanced_columns(result_df: pd.DataFrame) -> pd.DataFrame:
    recommended = {
        "Age": "Age eligibility and replacement planning",
        "Reproductive_Status": "Reproductive readiness and fertility screening",
        "Health_Status": "Health-based breeding exclusion",
        "Defect_Status": "Defect and abnormality filtering",
        "Line": "Line comparison and pure-line control",
        "Generation": "Generation trend and FS terminal-stock control",
        "Birth_Date": "Age tracking and generation management",
        "Mating_Date": "Breeding calendar",
        "Pen": "Pen/cage/location management",
        "Batch": "Batch management",
        "Genomic_EBV": "Genomic selection support",
        "Genomic_Reliability": "Genomic confidence weighting",
        "Feed_Cost": "Economic projection",
        "Production_Value": "Economic projection",
        "Culling_Value": "Economic projection",
    }

    rows = []
    for col, purpose in recommended.items():
        rows.append({
            "Recommended Column": col,
            "Status": "Available" if col in result_df.columns else "Not available in current sample",
            "Purpose": purpose,
        })
    return pd.DataFrame(rows)


def render_section_header(title: str, description: str = "", icon: str = "📌"):
    """
    Consistent section header for better readability.
    """
    st.markdown(f"""
    <div class="info-card" style="border-left: 5px solid #3b82f6;">
        <h3 style="margin-top:0;">{icon} {title}</h3>
        <p style="margin-bottom:0; color: var(--text-sub);">{description}</p>
    </div>
    """, unsafe_allow_html=True)


def render_small_note(text: str):
    """
    Compact note for breeder-friendly guidance.
    """
    st.caption(text)


def render_workflow_overview():
    """
    Shows a compact workflow of the application.
    """
    st.markdown("""
    <div class="info-card">
        <b>Brief Workflow</b>
        <ol>
            <li><b>Input:</b> choose a sample or upload a file, then map the required columns.</li>
            <li><b>Validation:</b> check data quality, pedigree completeness, and metadata.</li>
            <li><b>Analysis:</b> review the dashboard, pedigree, relationship matrix, mating, and pure-line modules.</li>
            <li><b>Output:</b> download the PDF report, visualization, mating calendar, or next-generation template.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)



def render_tab_help(title: str, points: list):
    """
    Collapsible explanation block for each tab to avoid visual redundancy.
    """
    with st.expander(f"Guide: {title}", expanded=False):
        for point in points:
            st.write(f"- {point}")




def render_footer():
    """
    Application footer.
    """
    year = pd.Timestamp.now().year
    st.markdown(f"""
    <div style="
        margin-top: 3rem;
        padding: 1.25rem 0;
        border-top: 1px solid var(--border-color);
        text-align: center;
        color: var(--text-sub);
        font-size: 0.9rem;
    ">
        Developed by <b>Galuh Adi Insani</b> © {year}
    </div>
    """, unsafe_allow_html=True)


def main():
    favicon_path = pathlib.Path(__file__).parent / "assets" / "favicon.svg"

    page_icon = str(favicon_path) if favicon_path.exists() else "🐄"

    st.set_page_config(
        page_title="Breeding & Inbreeding Analytics",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_css()
    render_header()

    with st.sidebar:
        st.markdown("## 🧭 Application Setup")

        with st.expander("Quick Guide", expanded=False):
            render_workflow_overview()
            st.markdown("""
            **Required:** `Animal_ID`, `Sire_ID`, `Dam_ID`  
            **Recommended:** `Sex`, `Age`, `Line`, `Generation`, `Phenotype`
            """)

        st.markdown("### 1. Data Source")

        mode = st.radio(
            "Select Data Source",
            ["Full cattle example", "Upload own file"],
            help="Use sample data to learn how the system works or upload your own CSV/Excel file.",
        )

        st.download_button(
            "Download Full Cattle Example XLSX",
            data=create_sample_excel_download(),
            file_name="full_cattle_example.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="Download the built-in ideal cattle pedigree sample in Excel format."
        )

        with st.expander("Required Data Format", expanded=False):
            st.markdown("""
            **Required columns:** `Animal_ID`, `Sire_ID`, `Dam_ID`  
            **Optional but recommended:** `Phenotype`, `Sex`, `Age`, `Line`, `Generation`

            Use `-` for unknown sire/dam. Keep every `Animal_ID` unique.
            """)

        if mode == "Full cattle example":
            raw_df = contoh_sapi_lengkap()
        else:
            uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
            if not uploaded:
                st.warning("Please upload a CSV or Excel file to begin.")
                st.stop()

            raw_df = read_file(uploaded)

        if len(raw_df) > 500:
            st.warning(
                f"Large data detected ({len(raw_df)} rows). "
                "The additive relationship matrix may take longer to calculate."
            )

        cols = list(raw_df.columns)

        st.markdown("### 2. Required Column Mapping")
        id_col = st.selectbox("Animal_ID Column", cols, index=0)
        sire_col = st.selectbox("Sire_ID Column", cols, index=1 if len(cols) > 1 else 0)
        dam_col = st.selectbox("Dam_ID Column", cols, index=2 if len(cols) > 2 else 0)

        phenotype_col = st.selectbox(
            "Phenotype Column - Optional",
            ["-"] + cols,
            index=cols.index("Phenotype") + 1 if "Phenotype" in cols else 0,
            help="Select a phenotype column to calculate Estimated Breeding Value.",
        )

        pheno_val = None if phenotype_col == "-" else phenotype_col

        st.markdown("### 3. Optional Breeding Metadata")
        sex_col = st.selectbox(
            "Sex Column - Optional",
            ["-"] + cols,
            index=cols.index("Sex") + 1 if "Sex" in cols else 0,
            help="Use values such as Male/Female, Male/Female, M/F, L/P."
        )
        age_col = st.selectbox(
            "Age Column - Optional",
            ["-"] + cols,
            index=cols.index("Age") + 1 if "Age" in cols else 0,
            help="Age can be written in weeks, months, or years depending on species."
        )
        line_col = st.selectbox(
            "Line Column - Optional",
            ["-"] + cols,
            index=cols.index("Line") + 1 if "Line" in cols else 0,
            help="Useful for pure-line systems such as Line A, Line B, Line C, Line D."
        )
        generation_col = st.selectbox(
            "Generation Column - Optional",
            ["-"] + cols,
            index=cols.index("Generation") + 1 if "Generation" in cols else 0,
            help="Examples: Founder, G0, G1, GGPS, GPS, PS, FS."
        )

        sex_val = None if sex_col == "-" else sex_col
        age_val = None if age_col == "-" else age_col
        line_val = None if line_col == "-" else line_col
        generation_val = None if generation_col == "-" else generation_col

        with st.expander("Advanced Optional Columns", expanded=False):
            st.caption("Map these columns only if they exist. If not available, the advanced modules will show guidance instead of errors.")

            birth_date_col = st.selectbox("Birth Date Column", ["-"] + cols, index=cols.index("Birth_Date") + 1 if "Birth_Date" in cols else 0)
            reproductive_col = st.selectbox("Reproductive Status Column", ["-"] + cols, index=cols.index("Reproductive_Status") + 1 if "Reproductive_Status" in cols else 0)
            health_col = st.selectbox("Health Status Column", ["-"] + cols, index=cols.index("Health_Status") + 1 if "Health_Status" in cols else 0)
            defect_col = st.selectbox("Defect Status Column", ["-"] + cols, index=cols.index("Defect_Status") + 1 if "Defect_Status" in cols else 0)
            culling_reason_col = st.selectbox("Culling Reason Column", ["-"] + cols, index=cols.index("Culling_Reason") + 1 if "Culling_Reason" in cols else 0)
            pen_col = st.selectbox("Pen Column", ["-"] + cols, index=cols.index("Pen") + 1 if "Pen" in cols else 0)
            cage_col = st.selectbox("Cage Column", ["-"] + cols, index=cols.index("Cage") + 1 if "Cage" in cols else 0)
            batch_col = st.selectbox("Batch Column", ["-"] + cols, index=cols.index("Batch") + 1 if "Batch" in cols else 0)
            mating_date_col = st.selectbox("Mating Date Column", ["-"] + cols, index=cols.index("Mating_Date") + 1 if "Mating_Date" in cols else 0)
            expected_date_col = st.selectbox("Expected Offspring Date Column", ["-"] + cols, index=cols.index("Expected_Offspring_Date") + 1 if "Expected_Offspring_Date" in cols else 0)
            responsible_col = st.selectbox("Responsible Person Column", ["-"] + cols, index=cols.index("Responsible_Person") + 1 if "Responsible_Person" in cols else 0)
            genomic_ebv_col = st.selectbox("Genomic EBV Column", ["-"] + cols, index=cols.index("Genomic_EBV") + 1 if "Genomic_EBV" in cols else 0)
            genomic_rel_col = st.selectbox("Genomic Reliability Column", ["-"] + cols, index=cols.index("Genomic_Reliability") + 1 if "Genomic_Reliability" in cols else 0)
            feed_cost_col = st.selectbox("Feed Cost Column", ["-"] + cols, index=cols.index("Feed_Cost") + 1 if "Feed_Cost" in cols else 0)
            production_value_col = st.selectbox("Production Value Column", ["-"] + cols, index=cols.index("Production_Value") + 1 if "Production_Value" in cols else 0)
            replacement_cost_col = st.selectbox("Replacement Cost Column", ["-"] + cols, index=cols.index("Replacement_Cost") + 1 if "Replacement_Cost" in cols else 0)
            culling_value_col = st.selectbox("Culling Value Column", ["-"] + cols, index=cols.index("Culling_Value") + 1 if "Culling_Value" in cols else 0)

        extra_metadata_cols = {
            "Birth_Date": None if birth_date_col == "-" else birth_date_col,
            "Reproductive_Status": None if reproductive_col == "-" else reproductive_col,
            "Health_Status": None if health_col == "-" else health_col,
            "Defect_Status": None if defect_col == "-" else defect_col,
            "Culling_Reason": None if culling_reason_col == "-" else culling_reason_col,
            "Pen": None if pen_col == "-" else pen_col,
            "Cage": None if cage_col == "-" else cage_col,
            "Batch": None if batch_col == "-" else batch_col,
            "Mating_Date": None if mating_date_col == "-" else mating_date_col,
            "Expected_Offspring_Date": None if expected_date_col == "-" else expected_date_col,
            "Responsible_Person": None if responsible_col == "-" else responsible_col,
            "Genomic_EBV": None if genomic_ebv_col == "-" else genomic_ebv_col,
            "Genomic_Reliability": None if genomic_rel_col == "-" else genomic_rel_col,
            "Feed_Cost": None if feed_cost_col == "-" else feed_cost_col,
            "Production_Value": None if production_value_col == "-" else production_value_col,
            "Replacement_Cost": None if replacement_cost_col == "-" else replacement_cost_col,
            "Culling_Value": None if culling_value_col == "-" else culling_value_col,
        }

        st.markdown("### 4. Multi-Trait Selection Index")
        excluded_trait_cols = {id_col, sire_col, dam_col, sex_val, age_val, line_val, generation_val}
        excluded_trait_cols.update([v for v in extra_metadata_cols.values() if v])
        candidate_trait_cols = [
            c for c in cols
            if c not in excluded_trait_cols
        ]

        default_traits = []
        for candidate in ["Phenotype", "Body_Weight", "Growth_Rate", "Fertility", "Survival_Rate", "Feed_Efficiency", "Egg_Production", "Egg_Weight", "Hatchability"]:
            if candidate in candidate_trait_cols:
                default_traits.append(candidate)

        multi_trait_cols = st.multiselect(
            "Additional Trait Columns - Optional",
            candidate_trait_cols,
            default=default_traits[:4],
            help="Select numeric traits to build a simple multi-trait selection index."
        )

        trait_weights = {}
        if multi_trait_cols:
            with st.expander("Trait Weights", expanded=False):
                st.caption("Weights are normalized automatically. Higher weight means the trait is more important.")
                for trait in multi_trait_cols:
                    trait_weights[trait] = st.slider(
                        f"Weight: {trait}",
                        0.0,
                        1.0,
                        1.0 / max(len(multi_trait_cols), 1),
                        0.05,
                    )

        st.markdown("### 5. Genetic Parameters")
        h2 = st.slider("Heritability (h²)", 0.0, 1.0, 0.3, 0.05)
        depression_rate = st.slider("Inbreeding Depression Rate per 1% F", 0.0, 5.0, 1.0, 0.1)

        st.markdown("### 6. Selection Parameters")
        intensity = st.slider("Selection Intensity (i)", 0.0, 3.0, 1.5, 0.1)

        st.markdown("### 7. Advanced Rules")
        with st.expander("Age and Calendar Rules", expanded=False):
            min_breeding_age = st.number_input("Minimum breeding age", min_value=0.0, max_value=1000.0, value=18.0, step=1.0)
            max_breeding_age = st.number_input("Maximum productive breeding age", min_value=0.0, max_value=1000.0, value=156.0, step=1.0)
            calendar_start_date = st.date_input("Default mating calendar start date", value=pd.Timestamp.now().date())
            mating_interval_days = st.number_input("Mating interval between pairs - days", min_value=1, max_value=365, value=7, step=1)
            expected_offspring_days = st.number_input("Expected offspring/hatch/birth days after mating", min_value=1, max_value=1000, value=21, step=1)

    try:
        try:
            internal = standardize_input(
                raw_df,
                id_col,
                sire_col,
                dam_col,
                phenotype_col=pheno_val,
                sex_col=sex_val,
                age_col=age_val,
                line_col=line_val,
                generation_col=generation_val,
                trait_cols=multi_trait_cols,
                extra_metadata_cols=extra_metadata_cols,
            )
        except Exception as input_error:
            st.error("The uploaded data format is not valid yet.")
            st.write("Please check your column mapping and required data format.")
            with st.expander("Error detail", expanded=False):
                st.write(str(input_error))
            show_input_validation_messages({
                "errors": [
                    "Column mapping failed. Make sure Animal_ID, Sire_ID, and Dam_ID are selected correctly."
                ],
                "warnings": []
            })
            st.stop()

        validation_result = validate_input_data(
            internal,
            phenotype_enabled=pheno_val is not None,
        )

        show_input_validation_messages(validation_result)

        if not validation_result["valid"]:
            st.stop()

        try:
            std_df, res_df, matrix_df = calculate(
                internal,
                h2=h2,
                depression_rate=depression_rate,
            )
            res_df = merge_optional_metadata(internal, res_df)
        except ValueError as calc_error:
            st.error("The pedigree structure could not be calculated.")
            st.write("Please fix the parent-child relationship records according to the rules below.")
            show_input_validation_messages({
                "errors": [str(calc_error)],
                "warnings": []
            })
            st.stop()
        except Exception as calc_error:
            st.error("The data could not be processed because the pedigree format is not valid.")
            st.write("Please check the uploaded file and follow the data writing rules below.")
            show_input_validation_messages({
                "errors": [
                    "Calculation failed. Check duplicated IDs, parent IDs, empty values, phenotype format, or pedigree cycles."
                ],
                "warnings": []
            })
            with st.expander("Technical detail for debugging", expanded=False):
                st.write(str(calc_error))
            st.stop()

        res_display_data = res_df[res_df["Data_Type"] == "Input data"].copy()
        res_display_data = add_sex_role_column(res_display_data)

        trait_column_map = {
            original: f"Trait_{str(original).strip().replace(' ', '_')}"
            for original in multi_trait_cols
        }
        selected_trait_result_cols = [
            mapped for mapped in trait_column_map.values()
            if mapped in res_display_data.columns
        ]

        res_display_data = calculate_multi_trait_selection_index(
            res_display_data,
            selected_trait_result_cols,
            {f"Trait_{str(k).strip().replace(' ', '_')}": v for k, v in trait_weights.items()},
        )

        res_display_data = evaluate_age_eligibility(res_display_data, min_breeding_age, max_breeding_age)
        res_display_data = evaluate_reproductive_status(res_display_data)
        res_display_data = evaluate_health_and_defect(res_display_data)
        res_display_data = calculate_ebv_reliability(res_display_data)
        res_display_data = apply_genomic_support(res_display_data)
        res_display_data = calculate_economic_projection(res_display_data)
        res_display_data = assign_final_breeding_decision(res_display_data)

        res_df = res_df.merge(
            res_display_data[[
                "Animal_ID", "Sex_Role", "Selection_Index", "Selection_Index_Rank",
                "Age_Eligibility", "Reproductive_Decision", "Health_Defect_Decision",
                "EBV_Reliability_%", "EBV_Reliability_Level", "Genomic_Status",
                "Combined_Genetic_Value", "Expected_Profit", "Economic_Status",
                "Final_Breeding_Decision"
            ]],
            on="Animal_ID",
            how="left",
            suffixes=("", "_display")
        )

        data_quality = calculate_data_quality_score(res_display_data)

        tabs = st.tabs([
            "1. Dashboard",
            "2. Genetic Charts",
            "3. Pedigree Explorer",
            "4. Relationship Matrix",
            "5. Heterosis",
            "6. Pure Line Pyramid",
            "7. Breeder Report",
            "8. Field Guide",
            "9. Advanced Breeding Management",
        ])

        with tabs[0]:
            render_section_header(
                "Dashboard",
                "Main summary of population status, data quality, inbreeding, EBV, selection index, and key breeder decisions.",
                "📊"
            )
            render_tab_help(
                "Dashboard",
                [
                    "Check the Data Quality Score first.",
                    "Use F, EBV, Selection Index, and Final Decision to understand population status.",
                    "Review selection and culling candidates before deciding mating pairs."
                ]
            )
            st.subheader("Data Summary")

            total_animals = len(res_display_data)
            inbred_animals = len(res_display_data[res_display_data["Inbreeding_%"] > 0])
            avg_f = float(res_display_data["Inbreeding_%"].mean()) if total_animals else 0
            max_f = float(res_display_data["Inbreeding_%"].max()) if total_animals else 0
            avg_ebv = float(res_display_data["EBV"].mean()) if total_animals else 0

            m0, m1, m2, m3, m4 = st.columns(5)
            m0.metric("Heritability h²", f"{h2:.2f}")
            m1.metric("Total Population", f"{total_animals}")
            m2.metric("Inbred Animals", f"{inbred_animals}")
            m3.metric("Average F", f"{avg_f:.2f}%")
            m4.metric("Average EBV", f"{avg_ebv:.4f}")

            st.markdown("### Data Quality Score")
            dq1, dq2, dq3 = st.columns(3)
            dq1.metric("Data Quality Score", f"{data_quality['score']:.2f}/100")
            dq2.metric("Data Quality Grade", data_quality["grade"])
            dq3.metric("Selected Traits", f"{len(selected_trait_result_cols)}")
            st.info(data_quality["recommendation"])

            with st.expander("Show Data Quality Details", expanded=False):
                st.dataframe(data_quality["details"], hide_index=True, use_container_width=True)

            if selected_trait_result_cols:
                st.markdown("### Multi-Trait Selection Index")
                st.caption("Selection_Index combines standardized selected traits and EBV. Higher values indicate better multi-trait breeding potential.")
                top_index = res_display_data.sort_values("Selection_Index", ascending=False).head(10)
                st.dataframe(
                    top_index[["Animal_ID", "Sex_Role", "Selection_Index", "Selection_Index_Rank", "EBV", "Inbreeding_%", "Classification"]],
                    hide_index=True,
                    use_container_width=True,
                )

            st.markdown("### Livestock Classification Distribution")

            if not res_display_data.empty:
                dist = res_display_data["Classification"].value_counts()
                dist_cols = st.columns(max(len(dist), 1))

                for i, (label, count) in enumerate(dist.items()):
                    dist_cols[i].metric(label, f"{count}")

                has_elite = any(res_display_data["Classification"] == "Elite Stock")

                if not has_elite:
                    st.warning("Warning: No Elite Stock was found in this population.")
                    st.markdown("""
                    <div class="info-card">
                    <b>Genetic Improvement Advice:</b><br>
                    The current population does not contain animals classified as <b>Elite Stock</b>, meaning no individual currently meets the ideal combination of high EBV and low inbreeding level.
                    To improve the genetic quality of the next generation, the following actions are recommended:
                    <ol>
                        <li><b>Outcrossing:</b> Introduce unrelated sires or frozen semen from outside the current population.</li>
                        <li><b>Strict Selection:</b> Prioritize animals classified as <b>Breeding Stock</b> as replacement parents.</li>
                        <li><b>Phenotype Re-evaluation:</b> Ensure performance data are recorded accurately because EBV depends strongly on phenotype quality.</li>
                        <li><b>Inbreeding Control:</b> Keep the inbreeding coefficient below 3% whenever possible.</li>
                        <li><b>Sire Rotation:</b> Avoid using one sire too frequently across the population.</li>
                    </ol>
                    </div>
                    """, unsafe_allow_html=True)

            if pheno_val:
                st.markdown("---")
                st.subheader("Selection Response Estimation")

                valid_phenos = pd.to_numeric(res_display_data["Phenotype"], errors="coerce").dropna()

                if not valid_phenos.empty:
                    avg_p = valid_phenos.mean()
                    sd_p = valid_phenos.std()
                    response = calculate_selection_response(h2, sd_p, intensity)

                    valid_heterosis = pd.to_numeric(
                        res_display_data["Heterosis"],
                        errors="coerce",
                    )
                    valid_heterosis = valid_heterosis[valid_heterosis != 0]
                    avg_h = valid_heterosis.mean() if not valid_heterosis.empty else 0.0

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Average Phenotype", f"{avg_p:.2f}")
                    r2.metric("Phenotype SD", f"{sd_p:.2f}")
                    r3.metric("Selection Response R", f"{response:.4f}")
                    r4.metric("Average Heterosis", f"{avg_h:.4f}")

                    st.info(f"""
                    **Result Interpretation**

                    - **Current Population Average:** The average phenotype value is **{avg_p:.2f}** units.
                    - **Predicted Genetic Progress:** With heritability **{h2}** and selection intensity **{intensity}**, the expected improvement in the next generation is **{response:.4f}** units.
                    - **Next Generation Estimate:** The predicted average performance of offspring is approximately **{avg_p + response:.2f}** units.
                    - **Average Heterosis:** The average heterosis value is **{avg_h:.4f}** units.

                    **Scientific Meaning**

                    Selection response estimates how much improvement can be expected after choosing the best animals as parents. Higher selection intensity may increase genetic progress, but it can also increase inbreeding risk if only a few related animals are repeatedly used.

                    **Breeding Recommendation**

                    Use animals with high EBV and low inbreeding as parent candidates. Avoid selecting animals with high EBV if they also have high inbreeding coefficients.
                    """)

                else:
                    st.warning("Phenotype data is invalid or empty, so selection response cannot be calculated.")

            backcross_cases = res_display_data[res_display_data["Reproduction_Warning"] != ""]

            if not backcross_cases.empty:
                st.markdown("---")
                st.error(f"Detected {len(backcross_cases)} sire-daughter mating case(s) or backcross risk.")

                c1, c2 = st.columns(2)

                with c1:
                    st.write("**List of Exposed Individuals:**")
                    st.dataframe(
                        backcross_cases[["Animal_ID", "Sire_ID", "Dam_ID", "Inbreeding_%"]],
                        hide_index=True,
                    )

                with c2:
                    st.markdown("""
                    <div class="info-card" style="border-left: 4px solid #ef4444;">
                    <b>Reproductive Risk Explanation:</b><br>
                    Mating a sire with its own daughter is a high-risk form of close inbreeding and may produce an inbreeding coefficient of approximately <b>25%</b> or higher.
                    <ul>
                        <li><b>Main Risks:</b> reduced fertility, weak immunity, slow growth, birth defects, and low offspring survival.</li>
                        <li><b>AI Strategy:</b> always check livestock records before selecting semen.</li>
                        <li><b>Semen Rotation:</b> use semen from unrelated sires or different genetic lines.</li>
                        <li><b>Management Decision:</b> offspring from close-relative mating should not be prioritized as replacement breeding stock.</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Inbreeding vs Phenotype Relationship Analysis")

            stats = calculate_stats(res_display_data)

            if stats:
                s1, s2, s3 = st.columns(3)

                s1.metric("Correlation r", f"{stats['correlation']:.4f}")
                s2.metric("Regression b", f"{stats['b']:.4f}")
                s3.metric("R-squared", f"{stats['r_squared']:.4f}")

                st.info(
                    f"**Regression Interpretation:** The estimated regression equation is "
                    f"Y = {stats['a']:.2f} + ({stats['b']:.4f})X. "
                    f"This means every 1% increase in inbreeding is predicted to change the phenotype value by "
                    f"{stats['b']:.4f} units. A negative value indicates possible performance decline as inbreeding increases."
                )
            else:
                st.info("Regression analysis requires at least three valid phenotype records.")

            st.markdown("---")
            st.subheader("Hardy-Weinberg Equilibrium Analysis")

            hwe_res = analyze_hardy_weinberg(res_display_data)

            hw1, hw2 = st.columns([0.4, 0.6])

            with hw1:
                st.write(f"#### Status: {hwe_res['status']}")
                st.write(hwe_res["insight"])
                st.caption(
                    "HWE analysis indicates whether the population is close to random mating or already shows signs of accumulated inbreeding."
                )

            with hw2:
                st.write("**Management Strategy Recommendation:**")
                for item in hwe_res["saran"]:
                    st.write(f"- {item}")

            if hwe_res["is_deviating"]:
                st.warning(
                    "The population shows signs of deviation from random mating. This may indicate accumulated inbreeding or repeated use of related animals."
                )
            else:
                st.success(
                    "The population is relatively close to a healthy genetic distribution. Continue pedigree monitoring."
                )

            st.markdown("---")
            st.markdown("### Complete Calculation Results")
            st.caption(
                "This table summarizes sex/role identification, pedigree information, EBV, heterosis, inbreeding coefficient, biological impact, classification, and recommendation."
            )
            st.info(
                "Sex/Role is automatically identified from pedigree usage. Animals appearing as `Sire_ID` are labeled as male/sire candidates, while animals appearing as `Dam_ID` are labeled as female/dam candidates. If the system cannot detect it, the animal will be labeled as unidentified."
            )

            st.dataframe(clean_display(res_display_data), use_container_width=True, height=500)

            st.markdown("---")
            csel1, csel2 = st.columns([0.4, 0.6])

            with csel1:
                st.subheader("Individual Genetic Interpretation")
                selected_animal = st.selectbox("Select Animal:", res_display_data["Animal_ID"])

            if selected_animal:
                row = res_display_data[res_display_data["Animal_ID"] == selected_animal].iloc[0]

                with csel2:
                    st.info(f"**Selected Animal:** {row['Animal_ID']}\n\n{row['Breeding_Interpretation']}")
                    st.markdown(f"""
                    - **Inbreeding Coefficient:** {row['Inbreeding_%']}% ({row['Inbreeding_Condition']})
                    - **Estimated Performance Impact:** -{row['Inbreeding_Depression']} units
                    - **Biological Impact:** {row['Biological_Impact']}
                    - **Recommendation:** {row['Recommendation']}
                    """)

            st.markdown("### Interpretation Guide")

            with st.expander("Click to understand the main breeding analysis terms"):
                st.markdown("""
                **1. Inbreeding Coefficient (F)**  
                Measures the probability that an animal inherits identical genes from both parents due to common ancestry.

                **2. Estimated Breeding Value (EBV)**  
                Estimates the genetic potential of an animal that can be passed to its offspring.

                **3. Inbreeding Depression**  
                The potential reduction in performance caused by increased homozygosity.

                **4. Heterosis**  
                The performance advantage of offspring compared with the average of both parents.

                **5. Selection Response (R)**  
                The expected improvement in the next generation after selecting superior parents.

                **6. Relationship Matrix (A)**  
                A matrix showing genetic relationship values between individuals. Lower values are preferred for mating pairs.
                """)

            st.markdown("### Population Impact Analysis")
            st.info(dampak_inbreeding(max_f))

            st.markdown("### Download Report")
            dl1, dl2, dl3, dl4 = st.columns(4)

            csv_data = clean_display(res_df).to_csv(index=False).encode("utf-8")
            dl1.download_button(
                "Download CSV",
                csv_data,
                "breeding_analytics_data.csv",
                "text/csv",
                use_container_width=True,
            )

            matrix_csv = matrix_df.to_csv().encode("utf-8")
            dl2.download_button(
                "Download Matrix CSV",
                matrix_csv,
                "relationship_matrix.csv",
                "text/csv",
                use_container_width=True,
            )

            current_settings = {"h2": h2, "intensity": intensity}
            txt_report = dots_to_pedigree(res_display_data, settings=current_settings)
            dl3.download_button(
                "Download TXT Summary",
                txt_report.encode("utf-8"),
                "breeding_summary.txt",
                "text/plain",
                use_container_width=True,
            )

            pdf_data = generate_pdf(res_display_data, settings=current_settings)

            if pdf_data:
                dl4.download_button(
                    "Download PDF Report",
                    pdf_data,
                    "breeding_report.pdf",
                    "application/pdf",
                    use_container_width=True,
                )
            else:
                dl4.info("Install reportlab to enable PDF download.")

            st.markdown("---")
            st.subheader("Selection & Culling Recommendations")

            st.markdown("""
            This section identifies animals that should be prioritized as parent candidates and animals that should not be used for future breeding.

            **Selection Candidates** are animals with relatively high EBV and low inbreeding risk.  
            **Culling Candidates** are animals with very high inbreeding, close-relative mating warnings, or low genetic potential.
            """)

            threshold_ebv = res_display_data["EBV"].quantile(0.75)
            selection_df = res_display_data[
                (res_display_data["EBV"] >= threshold_ebv) &
                (res_display_data["Inbreeding_%"] < 6.25)
            ].sort_values("EBV", ascending=False)

            threshold_low_ebv = res_display_data["EBV"].quantile(0.10)
            culling_df = res_display_data[
                (res_display_data["Inbreeding_%"] >= 25) |
                (res_display_data["Reproduction_Warning"] != "") |
                (res_display_data["EBV"] <= threshold_low_ebv)
            ].sort_values("Inbreeding_%", ascending=False)

            rc1, rc2 = st.columns(2)

            with rc1:
                st.success(f"Selection Candidates: {len(selection_df)} animal(s)")
                st.write("Recommended based on high EBV and low inbreeding risk.")
                st.dataframe(
                    selection_df[["Animal_ID", "EBV", "Inbreeding_%", "Classification"]],
                    hide_index=True,
                    use_container_width=True,
                )

            with rc2:
                st.error(f"Culling Candidates: {len(culling_df)} animal(s)")
                st.write("Identified based on high inbreeding, backcross warning, or low genetic value.")
                st.dataframe(
                    culling_df[["Animal_ID", "EBV", "Inbreeding_%", "Reproduction_Warning"]],
                    hide_index=True,
                    use_container_width=True,
                )

            st.markdown("---")
            st.subheader("Future Mating Strategy for Inbreeding Prevention")

            st.markdown("""
            This section does not only provide general advice. It also identifies which animals are suitable to be mated and simulates the expected offspring result using the additive relationship matrix.

            The main formula used in the simulation is:

            **Predicted Offspring Inbreeding (F) = 0.5 × Relationship between Sire and Dam × 100**

            Therefore, animals with a low relationship value are preferred because they are expected to produce offspring with lower inbreeding risk.
            """)

            selected_sires, selected_dams = infer_mating_candidates(res_display_data)

            ms1, ms2 = st.columns(2)

            with ms1:
                st.markdown("#### Potential Sire Candidates")
                st.caption("Male animals are identified from Sire_ID roles or male ID patterns such as SIRE, BULL, MALE, PEJANTAN, or JANTAN.")
                st.dataframe(
                    selected_sires[["Animal_ID", "Sex_Role", "EBV", "Inbreeding_%", "Classification"]],
                    hide_index=True,
                    use_container_width=True,
                )

            with ms2:
                st.markdown("#### Potential Dam Candidates")
                st.caption("Female animals are identified from Dam_ID roles or female ID patterns such as DAM, COW, FEMALE, INDUK, or BETINA.")
                st.dataframe(
                    selected_dams[["Animal_ID", "Sex_Role", "EBV", "Inbreeding_%", "Classification"]],
                    hide_index=True,
                    use_container_width=True,
                )

            st.markdown("#### Mating Simulation Settings")

            sim_col1, sim_col2 = st.columns(2)

            with sim_col1:
                max_offspring_f = st.slider(
                    "Maximum acceptable predicted offspring inbreeding F (%)",
                    min_value=0.0,
                    max_value=25.0,
                    value=6.25,
                    step=0.25,
                    help="Pairs with predicted offspring F below this value are considered safer."
                )

            with sim_col2:
                max_sim_pairs = st.slider(
                    "Number of mating pairs to display",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    help="The table will show the best simulated pairs based on low offspring F and high expected EBV."
                )

            simulated_pairs = simulate_mating_pairs(
                res_display_data,
                matrix_df,
                h2_value=h2,
                depression_rate_value=depression_rate,
                max_offspring_f=max_offspring_f,
                max_pairs=max_sim_pairs,
            )

            simulated_pairs = apply_blocked_mating_rules_to_pairs(
                simulated_pairs,
                res_display_data,
                matrix_df,
                max_safe_f=max_offspring_f,
            )

            st.markdown("#### Recommended Mating Pairs and Offspring Simulation")

            if simulated_pairs.empty:
                st.warning(
                    "No suitable mating simulation could be generated. Please check whether the dataset contains enough potential sires and dams, or add a Sex/Role indicator in the animal ID."
                )
            else:
                recommended_only = simulated_pairs[
                    simulated_pairs["Decision"].astype(str).str.contains("Recommended|safe|monitor", case=False, na=False)
                ]

                sim_metric_1, sim_metric_2, sim_metric_3, sim_metric_4 = st.columns(4)

                sim_metric_1.metric("Simulated Pairs", f"{len(simulated_pairs)}")
                sim_metric_2.metric(
                    "Best Offspring F",
                    f"{simulated_pairs['Predicted_Offspring_F_%'].min():.2f}%"
                )
                sim_metric_3.metric(
                    "Highest Expected EBV",
                    f"{simulated_pairs['Expected_Offspring_EBV'].max():.4f}"
                )
                sim_metric_4.metric(
                    "Recommended / Safe Pairs",
                    f"{len(recommended_only)}"
                )

                st.dataframe(
                    simulated_pairs,
                    hide_index=True,
                    use_container_width=True,
                )

                best_pair = simulated_pairs.iloc[0]

                best_sire_id = best_pair["Suggested_Sire"]
                best_dam_id = best_pair["Suggested_Dam"]

                best_sire_role = "Male / Sire Candidate"
                best_dam_role = "Female / Dam Candidate"

                if "Sex_Role" in res_display_data.columns:
                    sire_match = res_display_data[res_display_data["Animal_ID"].astype(str) == str(best_sire_id)]
                    dam_match = res_display_data[res_display_data["Animal_ID"].astype(str) == str(best_dam_id)]

                    if not sire_match.empty:
                        best_sire_role = sire_match.iloc[0]["Sex_Role"]

                    if not dam_match.empty:
                        best_dam_role = dam_match.iloc[0]["Sex_Role"]

                st.success(f"""
                **Best Suggested Pair**

                The most suitable pair based on the current simulation is:

                **Male / Sire:** {best_sire_id}  
                **Female / Dam:** {best_dam_id}

                This means the recommended mating pair is **{best_sire_id} as the male/sire** and **{best_dam_id} as the female/dam**.

                - **Sire role identification:** {best_sire_role}
                - **Dam role identification:** {best_dam_role}
                - **Predicted offspring inbreeding:** {best_pair['Predicted_Offspring_F_%']:.2f}%
                - **Expected offspring EBV:** {best_pair['Expected_Offspring_EBV']:.4f}
                - **Relationship value:** {best_pair['Relationship_A']:.4f}
                - **Estimated inbreeding depression:** {best_pair['Estimated_Inbreeding_Depression']:.4f} units
                - **Risk level:** {best_pair['Risk_Level']}
                - **Decision:** {best_pair['Decision']}
                """)

                st.markdown("#### Simulation Interpretation")

                st.info("""
                **How to read the simulation table:**

                - `Relationship_A` shows the genetic relationship between the proposed sire and dam.
                - `Predicted_Offspring_F_%` estimates the inbreeding coefficient of the future offspring.
                - `Expected_Offspring_EBV` estimates the average genetic merit inherited from both parents.
                - `Expected_Phenotype_Base` estimates offspring performance before inbreeding depression adjustment.
                - `Predicted_Phenotype_After_Depression` estimates performance after subtracting the inbreeding depression effect.
                - `Risk_Level` and `Decision` help determine whether the pair should be used, monitored, or avoided.

                The best mating pair is not always the pair with the highest EBV. A pair with slightly lower EBV but much lower predicted offspring inbreeding is often safer for long-term population improvement.
                """)

                st.download_button(
                    "Download Mating Simulation CSV",
                    simulated_pairs.to_csv(index=False).encode("utf-8"),
                    "mating_simulation.csv",
                    "text/csv",
                    use_container_width=True,
                )

            st.markdown("""
            <div class="info-card" style="border-left: 4px solid #3b82f6;">
            <b>Mating Strategy to Prevent Future Inbreeding:</b>
            <ul>
                <li><b>Avoid close-relative mating:</b> do not mate animals that share the same sire, dam, or close ancestors.</li>
                <li><b>Use unrelated sires:</b> select sires from different genetic lines with low relationship values to the dams.</li>
                <li><b>Apply sire rotation:</b> avoid using one sire too frequently.</li>
                <li><b>Use artificial insemination wisely:</b> choose semen from tested, unrelated sires.</li>
                <li><b>Check the Relationship Matrix:</b> prioritize mating pairs with relationship values close to zero.</li>
                <li><b>Balance EBV and inbreeding:</b> do not select an animal only because it has high EBV.</li>
                <li><b>Evaluate every generation:</b> recalculate EBV, inbreeding, and relationship values whenever new offspring are added.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Important Livestock Breeding Parameters")

            with st.expander("See detailed breeding concept explanation"):
                p1, p2 = st.columns(2)

                with p1:
                    st.markdown("""
                    **1. Heritability (h²)**  
                    Heritability measures how much of the observed phenotype variation is caused by additive genetic factors.

                    - Low heritability (< 0.20): strongly influenced by environment.
                    - Moderate heritability (0.20 - 0.40): common in growth traits.
                    - High heritability (> 0.40): common in carcass or some production traits.

                    **2. Estimated Breeding Value (EBV)**  
                    EBV predicts the genetic merit of an animal as a parent.
                    """)

                with p2:
                    st.markdown("""
                    **3. Selection Intensity (i)**  
                    Selection intensity describes how strict the selection process is.

                    **4. Tandem Selection**  
                    Tandem selection improves one trait first before focusing on another trait.

                    **5. Correlation and Regression**  
                    Correlation shows association strength, while regression estimates phenotype change when inbreeding increases.
                    """)

            st.markdown("### Insights & Management Strategy")

            mi1, mi2 = st.columns(2)

            with mi1:
                st.info("""
                **Why is inbreeding dangerous?**

                Inbreeding increases the chance that harmful recessive genes will appear in offspring. This may lead to:

                - Lower fertility
                - Reduced growth rate
                - Weaker immune system
                - Higher disease susceptibility
                - Lower milk or meat production
                - Higher risk of birth defects
                - Reduced offspring survival
                """)

            with mi2:
                st.success("""
                **Recommended prevention strategy**

                1. Rotate sires regularly.
                2. Use unrelated sires or external semen through artificial insemination.
                3. Avoid full-sibling, half-sibling, and parent-offspring mating.
                4. Record pedigree and phenotype data consistently.
                5. Use this application before making mating decisions.
                6. Recalculate inbreeding and EBV after every new generation.
                """)

        with tabs[1]:
            render_section_header(
                "Genetic Charts",
                "Visual summary of inbreeding distribution, EBV ranking, and phenotype relationship.",
                "📈"
            )
            render_tab_help(
                "Genetic Charts",
                [
                    "Summarizes inbreeding distribution, EBV, and phenotype relationships.",
                    "Use the charts to detect risk patterns and superior candidates."
                ]
            )
            st.subheader("Genetic Charts")

            v1, v2 = st.columns(2)

            with v1:
                st.markdown("### Inbreeding F Distribution")
                f_data = res_display_data["Inbreeding_%"]

                if not f_data.empty:
                    counts, bin_edges = np.histogram(f_data, bins=10)
                    hist_df = pd.DataFrame({
                        "F range (%)": [
                            f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}"
                            for i in range(len(counts))
                        ],
                        "Animal Count": counts,
                    })
                    st.bar_chart(hist_df.set_index("F range (%)"))

                st.caption("This graph shows the distribution of inbreeding levels in the population.")

            with v2:
                st.markdown("### Top 10 Genetic Potential Based on EBV")
                top_ebv = res_display_data.sort_values("EBV", ascending=False).head(10)

                if not top_ebv.empty:
                    st.bar_chart(top_ebv.set_index("Animal_ID")["EBV"])

                st.caption("This chart shows the top animals with the highest Estimated Breeding Value.")

            st.markdown("---")
            st.markdown("### Inbreeding vs Phenotype Relationship")

            if "Phenotype" in res_display_data.columns:
                scatter_data = res_display_data[
                    res_display_data["Phenotype"].apply(lambda x: not is_unknown(x))
                ].copy()

                if not scatter_data.empty:
                    scatter_data["Phenotype_Val"] = pd.to_numeric(
                        scatter_data["Phenotype"],
                        errors="coerce",
                    )
                    scatter_data["EBV_Size"] = pd.to_numeric(
                        scatter_data["EBV"],
                        errors="coerce",
                    ).abs() + 1

                    st.scatter_chart(
                        scatter_data,
                        x="Inbreeding_%",
                        y="Phenotype_Val",
                        color="Classification",
                        size="EBV_Size",
                    )
                    st.caption(
                        "Dots show the relationship between inbreeding and phenotype. Color indicates classification."
                    )
                else:
                    st.info("No valid phenotype data is available for scatter visualization.")
            else:
                st.info("No phenotype column is available.")

        with tabs[2]:
            render_section_header(
                "Pedigree Explorer",
                "Interactive pedigree exploration for family subgraph, high-risk animals, selection candidates, and full graph export.",
                "🌳"
            )
            render_tab_help(
                "Pedigree Explorer",
                [
                    "Use family subgraph mode for large datasets.",
                    "Use the high-risk filter to find problematic relationships.",
                    "Download HTML for complete visualization."
                ]
            )
            st.subheader("Pedigree Explorer")

            st.markdown("""
            This section supports both **complete pedigree information** and **safe visualization for large datasets**. 
            Rendering thousands of nodes directly in the browser is possible, but it may be slow depending on your computer and browser. 
            For large datasets, use filtered modes or download the full DOT file for external visualization.
            """)

            total_nodes = len(res_df)
            total_edges = int((~res_df["Sire_ID"].apply(is_unknown)).sum() + (~res_df["Dam_ID"].apply(is_unknown)).sum())

            pg_m1, pg_m2, pg_m3, pg_m4 = st.columns(4)
            pg_m1.metric("Total Nodes", f"{total_nodes}")
            pg_m2.metric("Estimated Edges", f"{total_edges}")
            pg_m3.metric("Inbred Animals", f"{int((res_df['Inbreeding_%'] > 0).sum())}")
            pg_m4.metric("High Risk F >= 12.5%", f"{int((res_df['Inbreeding_%'] >= 12.5).sum())}")

            with st.expander("Pedigree Summary", expanded=False):
                st.dataframe(
                    pedigree_summary_table(res_df),
                    hide_index=True,
                    use_container_width=True,
                )

            st.markdown("### Visualization Mode")

            graph_mode = st.selectbox(
                "Choose graph display mode",
                [
                    "Family subgraph around selected animal",
                    "High-risk animals only",
                    "Selection candidates only",
                    "Custom filter",
                    "Full pedigree graph",
                ],
                help="For thousands of animals, filtered views are recommended for faster rendering."
            )

            selected_graph_animal = None
            classification_filter = None
            only_inbred_filter = False
            only_warning_filter = False
            generations_up = 3
            generations_down = 2
            render_limit = None

            if graph_mode == "Family subgraph around selected animal":
                selected_graph_animal = st.selectbox(
                    "Select animal to inspect",
                    res_df["Animal_ID"].astype(str).tolist(),
                    help="The graph will show selected animal, ancestors, and descendants."
                )

                fam_col1, fam_col2 = st.columns(2)
                with fam_col1:
                    generations_up = st.slider("Ancestor generations", 1, 10, 3)
                with fam_col2:
                    generations_down = st.slider("Descendant generations", 0, 10, 2)

            elif graph_mode == "Custom filter":
                available_classes = sorted(res_df["Classification"].dropna().astype(str).unique().tolist())

                classification_filter = st.multiselect(
                    "Filter by classification",
                    available_classes,
                    default=available_classes,
                )

                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    only_inbred_filter = st.checkbox("Show only inbred animals", value=False)
                with filter_col2:
                    only_warning_filter = st.checkbox("Show only animals with reproduction warning", value=False)

            elif graph_mode == "Full pedigree graph":
                st.warning(
                    "Full graph mode may be heavy for hundreds or thousands of animals. If rendering is slow, download the DOT file or switch to filtered mode."
                )

            if graph_mode != "Full pedigree graph":
                render_limit = st.number_input(
                    "Maximum nodes to render in browser, 0 means no limit",
                    min_value=0,
                    max_value=10000,
                    value=500,
                    step=50,
                    help="This only limits what is rendered on screen. Full data remains available in downloads."
                )
                render_limit = None if render_limit == 0 else int(render_limit)
            else:
                confirm_full_render = st.checkbox(
                    "I understand the browser may slow down. Render the full graph.",
                    value=False,
                )
                if not confirm_full_render:
                    render_limit = 500
                    st.info("Full rendering is not confirmed, so the preview is limited to the first 500 nodes.")
                else:
                    render_limit = None

            graph_df = filter_pedigree_for_visualization(
                res_df,
                mode=graph_mode,
                selected_id=selected_graph_animal,
                classification_filter=classification_filter,
                only_inbred=only_inbred_filter,
                only_warning=only_warning_filter,
                max_nodes=render_limit,
                generations_up=generations_up,
                generations_down=generations_down,
            )

            st.markdown("### Graph Preview")
            st.caption(
                f"Rendering {len(graph_df)} node(s) in the browser. Full pedigree still contains {len(res_df)} node(s)."
            )

            if graph_df.empty:
                st.warning("No animals match the selected visualization filter.")
            else:
                dot_preview = make_dot_unlimited(graph_df, include_legend=True)
                st.graphviz_chart(dot_preview)

                with st.expander("Rendered Graph Data", expanded=False):
                    display_cols = [
                        "Animal_ID", "Sex_Role", "Sire_ID", "Dam_ID", "Phenotype",
                        "EBV", "Heterosis", "Inbreeding_%", "Classification",
                        "Inbreeding_Condition", "Reproduction_Warning"
                    ]
                    existing_cols = [col for col in display_cols if col in graph_df.columns]
                    st.dataframe(
                        clean_display(graph_df[existing_cols]),
                        hide_index=True,
                        use_container_width=True,
                    )

            st.markdown("### Download Visualization and Pedigree Data")

            full_dot = make_dot_unlimited(res_df, include_legend=True)
            preview_dot = make_dot_unlimited(graph_df, include_legend=True) if not graph_df.empty else ""

            full_html = make_pedigree_report_html(
                res_df,
                full_dot,
                title="Full Pedigree Visualization Report"
            )

            preview_html = make_pedigree_report_html(
                graph_df,
                preview_dot,
                title="Preview Pedigree Visualization Report"
            ) if not graph_df.empty else ""

            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                st.download_button(
                    "Download Full Visualization HTML",
                    full_html.encode("utf-8"),
                    "full_pedigree_visualization.html",
                    "text/html",
                    use_container_width=True,
                    help="Open this file directly in a browser. It contains the full pedigree graph and data table."
                )

            with export_col2:
                st.download_button(
                    "Download Current View HTML",
                    preview_html.encode("utf-8"),
                    "current_pedigree_view.html",
                    "text/html",
                    use_container_width=True,
                    help="Open this file in a browser. It contains only the currently rendered/filtered view."
                )

            with export_col3:
                pedigree_csv_cols = [
                    "Animal_ID", "Sex_Role", "Sire_ID", "Dam_ID", "Phenotype",
                    "EBV", "Heterosis", "Inbreeding_%", "Classification",
                    "Inbreeding_Condition", "Biological_Impact", "Recommendation",
                    "Reproduction_Warning"
                ]
                existing_export_cols = [col for col in pedigree_csv_cols if col in res_df.columns]
                st.download_button(
                    "Download Full Pedigree CSV",
                    clean_display(res_df[existing_export_cols]).to_csv(index=False).encode("utf-8"),
                    "full_pedigree_information.csv",
                    "text/csv",
                    use_container_width=True,
                )

            with st.expander("Advanced export for graph software", expanded=False):
                st.caption(
                    "Use DOT only if you want to open the graph in Graphviz, Gephi, yEd, or another external graph visualization tool."
                )
                adv_col1, adv_col2 = st.columns(2)

                with adv_col1:
                    st.download_button(
                        "Download Full DOT Source",
                        full_dot.encode("utf-8"),
                        "full_pedigree_graph.dot",
                        "text/vnd.graphviz",
                        use_container_width=True,
                    )

                with adv_col2:
                    st.download_button(
                        "Download Current View DOT Source",
                        preview_dot.encode("utf-8"),
                        "current_pedigree_view.dot",
                        "text/vnd.graphviz",
                        use_container_width=True,
                    )

            st.info("""
            **Recommended workflow for very large samples:**

            1. Use **Family subgraph** to inspect one animal's ancestors and descendants.
            2. Use **High-risk animals only** to quickly identify problematic matings.
            3. Use **Selection candidates only** to inspect animals suitable for breeding.
            4. Use **Full Pedigree DOT** for complete external visualization when the dataset contains thousands of animals.
            """)
        with tabs[3]:
            render_section_header(
                "Relationship Matrix",
                "Matrix of genetic relationships between animals. Lower values are preferred for safe mating.",
                "🧬"
            )
            render_tab_help(
                "Relationship Matrix",
                [
                    "Values close to zero are safer for mating.",
                    "Download CSV if the dataset is too large."
                ]
            )
            st.subheader("Additive Relationship Matrix")

            st.write(
                "The relationship matrix shows genetic relationship values between individuals. "
                "For mating decisions, choose pairs with low values or close to zero."
            )

            if len(matrix_df) > 500:
                st.warning(
                    "The matrix is larger than 500x500 and may cause browser lag. Download the CSV if needed."
                )
                if st.button("Display Matrix Anyway"):
                    st.dataframe(matrix_df, use_container_width=True)
            else:
                st.dataframe(matrix_df, use_container_width=True)

        with tabs[4]:
            render_section_header(
                "Heterosis & Crossbreeding",
                "Evaluation of offspring performance compared with parental average and crossbreeding potential.",
                "🔀"
            )
            render_tab_help(
                "Heterosis",
                [
                    "Positive heterosis may indicate good crossbred performance.",
                    "Always check EBV and inbreeding before using animals for breeding."
                ]
            )
            st.subheader("Heterosis Analysis")

            if "Heterosis" in res_display_data.columns:
                heterosis_df = res_display_data.copy()
                heterosis_df["Heterosis"] = pd.to_numeric(
                    heterosis_df["Heterosis"],
                    errors="coerce",
                ).fillna(0)

                avg_heterosis = heterosis_df["Heterosis"].mean()
                max_heterosis = heterosis_df["Heterosis"].max()
                min_heterosis = heterosis_df["Heterosis"].min()

                h1, h2_col, h3 = st.columns(3)
                h1.metric("Average Heterosis", f"{avg_heterosis:.4f}")
                h2_col.metric("Highest Heterosis", f"{max_heterosis:.4f}")
                h3.metric("Lowest Heterosis", f"{min_heterosis:.4f}")

                st.info("""
                **Heterosis Interpretation**

                Heterosis measures offspring performance compared with the average performance of both parents.
                A positive value indicates that the offspring performs better than the parental average.
                A negative value indicates that the offspring performs below the parental average.

                In commercial livestock systems, heterosis is useful for improving growth, survival, fertility, and production traits through planned crossbreeding.
                """)

                top_h = heterosis_df.sort_values("Heterosis", ascending=False).head(10)

                st.markdown("### Top Heterosis Individuals")
                st.dataframe(
                    top_h[["Animal_ID", "Sire_ID", "Dam_ID", "Phenotype", "Heterosis", "EBV", "Inbreeding_%"]],
                    hide_index=True,
                    use_container_width=True,
                )

                if not top_h.empty:
                    st.bar_chart(top_h.set_index("Animal_ID")["Heterosis"])

                st.markdown("### Crossbreeding Management Notes")
                st.markdown("""
                - Positive heterosis may indicate useful genetic combinations between sire and dam.
                - Avoid using heterosis alone as a selection criterion; always check EBV and inbreeding level.
                - Use crossbreeding to restore genetic diversity when inbreeding is increasing.
                - For purebred improvement, use crossbreeding carefully because it may change breed composition.
                """)
            else:
                st.info("Heterosis data is not available.")


        with tabs[5]:
            render_section_header(
                "Pure Line Pyramid",
                "Simulation of four-line breeding structure for GGPS, GPS, PS, and FS planning.",
                "🏗️"
            )
            render_tab_help(
                "Pure Line",
                [
                    "GGPS/GPS maintain pure lines.",
                    "PS is used for controlled crossing.",
                    "FS is terminal commercial stock."
                ]
            )
            st.subheader("Pure Line Pyramid Simulation")

            st.markdown("""
            This module simulates a safe **four-line breeding pyramid** to support the development of:

            - **GGPS**: Great Grand Parent Stock as the pure-line nucleus.
            - **GPS**: Grand Parent Stock as pure-line multiplication stock.
            - **PS**: Parent Stock from controlled inter-line crossing.
            - **FS**: Final Stock as terminal commercial offspring.

            The goal is to keep the pure lines safe by choosing male and female founders with low relationship values, low predicted offspring inbreeding, and acceptable EBV.
            """)

            st.warning("""
            Important: this is a decision-support simulation. A real pure-line program should ideally use many males and females per line, not only one pair. If the uploaded dataset has limited animals, the system will still show the safest available structure, but it should be validated by a breeding expert before implementation.
            """)

            pl_col1, pl_col2, pl_col3 = st.columns(3)

            with pl_col1:
                pure_max_f = st.slider(
                    "Maximum safe F for pure-line offspring (%)",
                    min_value=0.0,
                    max_value=12.5,
                    value=6.25,
                    step=0.25,
                    help="Lower values are safer for long-term pure-line development."
                )

            with pl_col2:
                required_lines = st.slider(
                    "Number of pure lines",
                    min_value=2,
                    max_value=4,
                    value=4,
                    step=1,
                    help="Four lines are recommended to build a GGPS-GPS-PS-FS pyramid."
                )

            with pl_col3:
                min_founder_note = st.selectbox(
                    "Program objective",
                    [
                        "Balanced safety and EBV",
                        "Prioritize lowest inbreeding",
                        "Prioritize higher EBV with safe F",
                    ],
                )

            pair_pool = build_pair_pool_for_pure_lines(
                res_display_data,
                matrix_df,
                h2_value=h2,
                depression_rate_value=depression_rate,
                max_offspring_f=pure_max_f,
            )

            selected_lines = select_four_safe_pure_lines(
                pair_pool,
                max_offspring_f=pure_max_f,
                required_lines=required_lines,
            )

            if pair_pool.empty or selected_lines.empty:
                st.error(
                    "The system could not build a pure-line simulation from the current data. Please provide more animals with clear sire/dam information or add more unrelated male and female candidates."
                )
            else:
                st.markdown("### 1. Selected GGPS Pure-Line Founders")

                founder_display = selected_lines[[
                    "Line",
                    "GGPS_Male",
                    "Sire_Role",
                    "GGPS_Female",
                    "Dam_Role",
                    "Relationship_A",
                    "GGPS_Expected_F_%",
                    "GGPS_Expected_EBV",
                    "Risk_Level",
                ]].copy()

                founder_display = founder_display.rename(columns={
                    "GGPS_Male": "GGPS_Male_Sire",
                    "GGPS_Female": "GGPS_Female_Dam",
                    "GGPS_Expected_F_%": "Predicted_GGPS_Offspring_F_%",
                    "GGPS_Expected_EBV": "Expected_GGPS_Offspring_EBV",
                })

                st.dataframe(
                    founder_display,
                    hide_index=True,
                    use_container_width=True,
                )

                safe_count = len(selected_lines[selected_lines["GGPS_Expected_F_%"] <= pure_max_f])
                unique_sires = selected_lines["GGPS_Male"].nunique()
                unique_dams = selected_lines["GGPS_Female"].nunique()

                pm1, pm2, pm3, pm4 = st.columns(4)
                pm1.metric("Selected Lines", f"{len(selected_lines)}")
                pm2.metric("Safe Lines", f"{safe_count}")
                pm3.metric("Unique Males", f"{unique_sires}")
                pm4.metric("Unique Females", f"{unique_dams}")

                if len(selected_lines) < 4:
                    st.warning(
                        "Less than four lines could be selected from the current data. Add more unrelated male and female candidates to build a complete four-line GGPS-GPS-PS-FS structure."
                    )

                if unique_sires < len(selected_lines) or unique_dams < len(selected_lines):
                    st.warning(
                        "Some sires or dams are reused across lines. For a safer pure-line program, each line should ideally have different unrelated male and female founders."
                    )

                st.markdown("### 2. GGPS → GPS → PS → FS Pyramid Simulation")

                pyramid_df = simulate_stock_pyramid_from_lines(
                    selected_lines,
                    max_offspring_f=pure_max_f,
                )

                st.dataframe(
                    pyramid_df,
                    hide_index=True,
                    use_container_width=True,
                )

                st.markdown("### 3. Breeding Pyramid Flow")

                st.graphviz_chart(make_pure_line_flow_dot(pyramid_df))

                st.markdown("### 4. Recommended Four-Line Structure")

                if len(selected_lines) >= 4:
                    line_a = selected_lines.iloc[0]
                    line_b = selected_lines.iloc[1]
                    line_c = selected_lines.iloc[2]
                    line_d = selected_lines.iloc[3]

                    st.success(f"""
                    **Safe Four-Line Pyramid Recommendation**

                    **GGPS Pure Lines**
                    - **Line A:** Male/Sire `{line_a['GGPS_Male']}` × Female/Dam `{line_a['GGPS_Female']}`
                    - **Line B:** Male/Sire `{line_b['GGPS_Male']}` × Female/Dam `{line_b['GGPS_Female']}`
                    - **Line C:** Male/Sire `{line_c['GGPS_Male']}` × Female/Dam `{line_c['GGPS_Female']}`
                    - **Line D:** Male/Sire `{line_d['GGPS_Male']}` × Female/Dam `{line_d['GGPS_Female']}`

                    **GPS Stage**
                    - Maintain each line separately as pure-line multiplication stock.

                    **PS Stage**
                    - Produce **PS Male Line** from `Line A × Line B`.
                    - Produce **PS Female Line** from `Line C × Line D`.

                    **FS Stage**
                    - Produce commercial **Final Stock** from `(Line A × Line B) × (Line C × Line D)`.

                    This structure helps preserve pure-line identity at the GGPS/GPS level while reducing inbreeding risk at PS and FS levels through controlled inter-line crossing.
                    """)
                else:
                    st.info(
                        "A complete four-line recommendation requires four selected lines. The current simulation shows the safest partial structure available from the uploaded dataset."
                    )

                st.markdown("### 5. Safety Rules for Pure-Line Production")

                st.info("""
                **Recommended safety rules:**

                1. Keep **GGPS and GPS** as pure-line nucleus and multiplication stock.
                2. Do not use **FS** as breeding stock. FS should be terminal commercial stock.
                3. Avoid mating animals with predicted offspring F above the selected safety threshold.
                4. Maintain more than one male and one female per line whenever possible.
                5. Use the relationship matrix before every mating decision.
                6. Recalculate the pyramid every generation because relationship values and inbreeding risks will change.
                7. If a line repeatedly produces high F values, introduce unrelated animals into that line or redesign the line structure.
                """)

                st.download_button(
                    "Download Pure Line Founder Plan CSV",
                    founder_display.to_csv(index=False).encode("utf-8"),
                    "pure_line_founder_plan.csv",
                    "text/csv",
                    use_container_width=True,
                )

                st.download_button(
                    "Download GGPS-GPS-PS-FS Pyramid Simulation CSV",
                    pyramid_df.to_csv(index=False).encode("utf-8"),
                    "pure_line_pyramid_simulation.csv",
                    "text/csv",
                    use_container_width=True,
                )


        with tabs[6]:
            render_section_header(
                "Breeder Report",
                "Breeder-friendly conclusion, candidate summary, mating recommendation, pure-line readiness, and official PDF export.",
                "📄"
            )
            render_tab_help(
                "Breeder Report",
                [
                    "Fill in report identity before downloading the PDF.",
                    "Review the status, recommendations, and final decision."
                ]
            )
            st.subheader("Breeder Report Summary")

            st.markdown("""
            This tab provides a practical summary for breeders. It combines population status, inbreeding risk, EBV performance, selection candidates, culling candidates, mating recommendation, and pure-line pyramid readiness into one easy-to-read conclusion.
            """)

            safe_summary_f = st.slider(
                "Safe inbreeding threshold for summary (%)",
                min_value=0.0,
                max_value=12.5,
                value=6.25,
                step=0.25,
                help="This threshold is used to classify safe mating, selection candidates, and pure-line recommendations."
            )

            st.markdown("### Report Identity")

            meta_col1, meta_col2, meta_col3 = st.columns(3)

            with meta_col1:
                report_farm_name = st.text_input(
                    "Farm Name",
                    value="Farm / Breeding Unit",
                    help="This will appear on the PDF cover page."
                )

            with meta_col2:
                report_breeder_name = st.text_input(
                    "Breeder / Manager Name",
                    value="Breeder / Manager",
                    help="This will appear on the PDF cover page."
                )

            with meta_col3:
                report_species = st.selectbox(
                    "Species",
                    ["Chicken", "Cattle", "Goat", "Sheep", "Pig", "Duck", "Quail", "Custom / Other"],
                    index=0,
                    help="This will appear on the PDF report."
                )

            report_notes = st.text_area(
                "Report Notes - Optional",
                value="",
                placeholder="Example: Evaluation for GGPS-GPS-PS-FS breeding program, 2026 batch.",
                help="Optional note that will be printed on the cover page."
            )


            breeder_summary = build_breeder_summary_data(
                res_display_data,
                matrix_df,
                h2_value=h2,
                intensity_value=intensity,
                depression_rate_value=depression_rate,
                safe_f_threshold=safe_summary_f,
            )

            st.markdown("### Executive Status")

            status_col1, status_col2, status_col3, status_col4 = st.columns(4)

            status_col1.metric("Overall Status", breeder_summary["overall_status"])
            status_col2.metric("Total Animals", f"{breeder_summary['total']}")
            status_col3.metric("Average F", f"{breeder_summary['avg_f']:.2f}%")
            status_col4.metric("Average EBV", f"{breeder_summary['avg_ebv']:.4f}")

            st.success(f"**Conclusion:** {breeder_summary['conclusion']}")

            st.markdown("### Population Risk Summary")

            risk_summary_df = pd.DataFrame({
                "Category": [
                    "Not inbred",
                    "Low risk: 0 < F < 6.25%",
                    "Moderate risk: 6.25% <= F < 12.5%",
                    "High risk: 12.5% <= F < 25%",
                    "Very high risk: F >= 25%",
                    "Reproduction warning",
                ],
                "Total": [
                    breeder_summary["total"] - breeder_summary["inbred"],
                    breeder_summary["low_risk"],
                    breeder_summary["moderate_risk"],
                    breeder_summary["high_risk"],
                    breeder_summary["very_high_risk"],
                    breeder_summary["warning_count"],
                ],
            })

            st.dataframe(risk_summary_df, hide_index=True, use_container_width=True)

            st.markdown("### Classification Summary")

            class_summary_df = pd.DataFrame(
                [{"Classification": k, "Total": v} for k, v in breeder_summary["class_counts"].items()]
            )

            if not class_summary_df.empty:
                st.dataframe(class_summary_df, hide_index=True, use_container_width=True)
            else:
                st.info("No classification summary is available.")

            st.markdown("### Phenotype and Selection Response Summary")

            phenotype_info = breeder_summary["phenotype_summary"]

            if phenotype_info["available"]:
                pheno_cols = st.columns(4)
                pheno_cols[0].metric("Average Phenotype", f"{phenotype_info['average']:.4f}" if phenotype_info["average"] is not None else "-")
                pheno_cols[1].metric("Phenotype SD", f"{phenotype_info['sd']:.4f}" if phenotype_info["sd"] is not None else "-")
                pheno_cols[2].metric("Selection Response R", f"{phenotype_info['selection_response']:.4f}" if phenotype_info["selection_response"] is not None else "-")
                pheno_cols[3].metric("Next Gen Estimate", f"{phenotype_info['next_generation_estimate']:.4f}" if phenotype_info["next_generation_estimate"] is not None else "-")
            else:
                st.info("Phenotype data is not available, so selection response cannot be summarized.")

            st.markdown("### Recommended Breeding Candidates")

            selection_df = breeder_summary["selection_df"].head(15)

            if not selection_df.empty:
                st.dataframe(
                    selection_df[[
                        "Animal_ID", "Sex_Role", "Sire_ID", "Dam_ID",
                        "EBV", "Inbreeding_%", "Classification", "Recommendation"
                    ]],
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.warning("No strong breeding candidates were detected under the current criteria.")

            st.markdown("### Priority Non-Breeding / Culling Candidates")

            culling_df = breeder_summary["culling_df"].head(15)

            if not culling_df.empty:
                st.dataframe(
                    culling_df[[
                        "Animal_ID", "Sex_Role", "EBV", "Inbreeding_%",
                        "Classification", "Reproduction_Warning", "Recommendation"
                    ]],
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.success("No priority culling candidates were detected.")

            st.markdown("### Best Mating Recommendation")

            best_pair = breeder_summary["best_pair"]

            if best_pair:
                st.info(f"""
                **Best simulated pair for safe mating:**

                - **Male / Sire:** {best_pair['Suggested_Sire']}
                - **Female / Dam:** {best_pair['Suggested_Dam']}
                - **Predicted offspring F:** {best_pair['Predicted_Offspring_F_%']:.2f}%
                - **Expected offspring EBV:** {best_pair['Expected_Offspring_EBV']:.4f}
                - **Risk level:** {best_pair['Risk_Level']}
                - **Decision:** {best_pair['Decision']}
                """)
            else:
                st.warning("No mating recommendation could be generated from the current dataset.")

            st.markdown("### Pure Line Pyramid Readiness")

            selected_lines = breeder_summary["selected_lines"]
            pyramid_df = breeder_summary["pyramid_df"]

            if not selected_lines.empty:
                st.write("**Selected GGPS founder lines:**")
                st.dataframe(
                    selected_lines[[
                        "Line", "GGPS_Male", "Sire_Role", "GGPS_Female", "Dam_Role",
                        "Relationship_A", "GGPS_Expected_F_%", "GGPS_Expected_EBV", "Risk_Level"
                    ]],
                    hide_index=True,
                    use_container_width=True,
                )

                if len(selected_lines) >= 4:
                    st.success(
                        "The dataset can simulate a four-line GGPS-GPS-PS-FS pyramid. Use it as a planning reference and validate with more founder animals for real implementation."
                    )
                else:
                    st.warning(
                        "The dataset can only simulate a partial pure-line structure. Add more unrelated male and female candidates to build a safer four-line pyramid."
                    )

                with st.expander("Show GGPS-GPS-PS-FS simulation table", expanded=False):
                    st.dataframe(pyramid_df, hide_index=True, use_container_width=True)
            else:
                st.warning(
                    "Pure-line pyramid readiness is low. The current dataset does not provide enough safe male-female founder combinations."
                )

            st.markdown("### Practical Breeder Recommendations")

            for i, rec in enumerate(breeder_summary["recommendations"], start=1):
                st.write(f"{i}. {rec}")

            st.markdown("### Download Complete PDF Summary")

            pdf_summary = generate_breeder_summary_pdf(
                res_display_data,
                matrix_df,
                h2_value=h2,
                intensity_value=intensity,
                depression_rate_value=depression_rate,
                safe_f_threshold=safe_summary_f,
                farm_name=report_farm_name,
                breeder_name=report_breeder_name,
                species=report_species,
                report_notes=report_notes,
            )

            if pdf_summary:
                st.download_button(
                    "Download Official Breeding Report PDF",
                    pdf_summary,
                    "breeder_summary_conclusion_report.pdf",
                    "application/pdf",
                    use_container_width=True,
                )
            else:
                st.error("PDF generation is not available. Please install reportlab: pip install reportlab")

            summary_csv = pd.DataFrame({
                "Item": [
                    "Overall Status",
                    "Conclusion",
                    "Total Animals",
                    "Inbred Animals",
                    "Average F",
                    "Highest F",
                    "Average EBV",
                    "Highest EBV",
                    "Reproduction Warnings",
                    "Selection Candidates",
                    "Culling Candidates",
                ],
                "Value": [
                    breeder_summary["overall_status"],
                    breeder_summary["conclusion"],
                    breeder_summary["total"],
                    breeder_summary["inbred"],
                    f"{breeder_summary['avg_f']:.2f}%",
                    f"{breeder_summary['max_f']:.2f}%",
                    f"{breeder_summary['avg_ebv']:.4f}",
                    f"{breeder_summary['max_ebv']:.4f}",
                    breeder_summary["warning_count"],
                    len(breeder_summary["selection_df"]),
                    len(breeder_summary["culling_df"]),
                ],
            })

            st.download_button(
                "Download Summary CSV",
                summary_csv.to_csv(index=False).encode("utf-8"),
                "breeder_summary.csv",
                "text/csv",
                use_container_width=True,
            )


        with tabs[7]:
            render_section_header(
                "Field Guide",
                "Practical explanation of how pedigree breeding can be implemented in real poultry and cattle programs.",
                "📚"
            )
            render_tab_help(
                "Field Guide",
                [
                    "Field implementation guide.",
                    "Explains limitations, workflow, and species differences."
                ]
            )
            st.subheader("Field Implementation Guide")

            st.markdown("""
            This tab explains how pedigree-based breeding can be applied in real livestock production. 
            The goal is to help breeders understand that pedigree analysis is possible, but it must be treated as a long-term management system, not an instant method for producing elite animals.
            """)

            st.info("""
            **Main conclusion:**  
            Pedigree breeding is possible in real life, but its difficulty depends on the species, population size, recording quality, generation interval, and breeding objective. 
            The system is faster and easier to apply in poultry than in cattle because poultry have shorter generation intervals and larger offspring numbers.
            """)

            st.markdown("### 1. Is Pedigree Breeding Realistically Possible?")

            st.markdown("""
            Yes, pedigree breeding is realistically possible. However, it requires consistent recording over multiple generations. 
            Breeders need to record individual identity, sire, dam, sex, birth date, performance, reproduction status, health status, and mating history.

            This application helps reduce manual difficulty by calculating inbreeding, EBV, relationship matrix, mating simulation, pure-line planning, and breeder-friendly conclusions. 
            It does not replace real field evaluation, but it helps breeders make safer and more organized decisions.
            """)

            st.markdown("### 2. Poultry vs Cattle Implementation")

            species_df = pd.DataFrame({
                "Aspect": [
                    "Generation interval",
                    "Number of offspring",
                    "Speed of selection",
                    "Pure-line system",
                    "Elite identification",
                    "Main challenge",
                ],
                "Poultry / Chicken": [
                    "Shorter, so genetic progress can be observed faster.",
                    "High, allowing more candidates to be evaluated.",
                    "Faster because many individuals can be selected each generation.",
                    "Very suitable for GGPS, GPS, PS, and FS pyramid systems.",
                    "More practical because many candidates are available.",
                    "Requires strict line separation and accurate recording.",
                ],
                "Cattle": [
                    "Longer, so selection decisions take more time.",
                    "Low, usually one calf per pregnancy.",
                    "Slower because offspring need time to mature and be evaluated.",
                    "Possible, but more difficult and expensive.",
                    "Harder because fewer animals are available and evaluation takes years.",
                    "Long generation interval, high cost, and limited population size.",
                ],
            })

            st.dataframe(species_df, hide_index=True, use_container_width=True)

            st.markdown("### 3. Why Finding Elite Animals Is Difficult")

            st.warning("""
            In real breeding programs, elite animals are difficult to find because good performance alone is not enough. 
            An animal may have high performance, but it may not be suitable as breeding stock if its inbreeding is high, its relationship to other animals is too close, or its reproductive performance is poor.
            """)

            difficulty_df = pd.DataFrame({
                "Challenge": [
                    "Long generation interval",
                    "Incomplete recording",
                    "Small population size",
                    "High cost of evaluation",
                    "Performance is not always genetic",
                    "High relationship between candidates",
                ],
                "Practical Meaning": [
                    "Breeders must wait until animals mature before confirming breeding value.",
                    "Missing sire/dam records reduce the accuracy of inbreeding and relationship analysis.",
                    "Few candidates make it harder to find low-risk, high-EBV animals.",
                    "Measuring growth, production, fertility, and health requires time and resources.",
                    "A high-performing animal may perform well because of environment, not genetics.",
                    "Using related animals repeatedly increases inbreeding risk in the next generation.",
                ],
                "How This App Helps": [
                    "Provides early screening using available pedigree and phenotype data.",
                    "Shows additional founders and warnings when parent records are incomplete.",
                    "Ranks candidates using EBV, inbreeding, and classification.",
                    "Summarizes data automatically for faster decision-making.",
                    "Uses phenotype and heritability to estimate EBV and selection response.",
                    "Uses the relationship matrix and mating simulation before breeding.",
                ],
            })

            st.dataframe(difficulty_df, hide_index=True, use_container_width=True)

            st.markdown("### 4. Practical Step-by-Step Implementation")

            st.markdown("""
            A realistic breeding program should be implemented gradually:

            **Stage 1 — Basic Recording**  
            Record `Animal_ID`, `Sire_ID`, `Dam_ID`, sex, birth date, line, and generation.

            **Stage 2 — Performance Recording**  
            Add measurable traits such as body weight, egg production, milk yield, growth rate, fertility, survival, or health score.

            **Stage 3 — Genetic Screening**  
            Use this application to calculate inbreeding, EBV, classification, relationship matrix, and risk status.

            **Stage 4 — Selection and Culling**  
            Select animals with high EBV and low inbreeding as breeding candidates. 
            Avoid using animals with high inbreeding, reproduction warnings, or low genetic value as parents.

            **Stage 5 — Mating Simulation**  
            Simulate sire-dam combinations before mating. 
            Choose pairs with low predicted offspring inbreeding and acceptable expected EBV.

            **Stage 6 — Next Generation Validation**  
            After offspring are born, record them and recalculate all values. 
            Breeding decisions should be updated every generation.
            """)

            st.markdown("### 5. Recommended Breeding Workflow")

            workflow_df = pd.DataFrame({
                "Step": [
                    "1. Identify animals",
                    "2. Record pedigree",
                    "3. Record phenotype",
                    "4. Run analysis",
                    "5. Select candidates",
                    "6. Simulate mating",
                    "7. Produce offspring",
                    "8. Recalculate",
                ],
                "Breeder Action": [
                    "Give every animal a unique ID.",
                    "Record sire and dam for every animal.",
                    "Measure production or performance traits.",
                    "Upload data into this application.",
                    "Choose high-EBV and low-F animals.",
                    "Check predicted offspring F before mating.",
                    "Record new offspring as the next generation.",
                    "Repeat analysis after every generation.",
                ],
                "Expected Benefit": [
                    "Prevents confusion and duplicate records.",
                    "Allows inbreeding and relationship calculation.",
                    "Improves EBV and selection accuracy.",
                    "Produces objective breeding information.",
                    "Improves genetic progress safely.",
                    "Reduces risk of close-relative mating.",
                    "Maintains long-term pedigree continuity.",
                    "Keeps the breeding program updated.",
                ],
            })

            st.dataframe(workflow_df, hide_index=True, use_container_width=True)

            st.markdown("### 6. Practical Notes for Poultry Pure-Line Programs")

            st.success("""
            Poultry is more suitable for pure-line pyramid planning because the population can be expanded faster. 
            The recommended structure is:

            - **GGPS:** pure-line nucleus population.
            - **GPS:** multiplication of each pure line.
            - **PS:** controlled crossing between selected lines.
            - **FS:** final commercial stock, not used as breeding nucleus.

            For safety, keep each pure line separated, avoid uncontrolled mating, and recalculate inbreeding after every generation.
            """)

            st.markdown("### 7. Practical Notes for Cattle Breeding Programs")

            st.info("""
            For cattle, pedigree breeding is still useful but should be understood as a long-term program. 
            Because cattle have a longer generation interval and fewer offspring, the application is most useful for:

            - preventing close-relative mating,
            - choosing safer sire-dam combinations,
            - identifying replacement candidates,
            - supporting artificial insemination planning,
            - avoiding overuse of one popular sire,
            - monitoring long-term inbreeding trends.

            Elite cattle may take years to confirm, so early screening should be combined with field performance, reproduction records, health records, and possibly genomic information when available.
            """)

            st.markdown("### 8. Final Breeder-Friendly Conclusion")

            st.markdown("""
            Pedigree analysis is possible and useful in real breeding programs, but it must be supported by consistent recording and repeated evaluation. 
            For poultry, the system can be applied faster and is suitable for structured pure-line production. 
            For cattle, the system is slower but still valuable for preventing inbreeding and supporting selection decisions.

            The safest practical approach is:

            **pedigree recording + performance recording + inbreeding analysis + EBV estimation + mating simulation + regular re-evaluation.**

            This application should be used as a decision-support tool. It helps breeders make better decisions, but final breeding decisions should still consider animal health, fertility, management conditions, and long-term breeding goals.
            """)

            implementation_summary = pd.DataFrame({
                "Main Point": [
                    "Pedigree breeding is possible",
                    "Poultry is faster",
                    "Cattle is slower",
                    "Elite animals are hard to find",
                    "Recording is essential",
                    "Mating simulation is important",
                    "Re-evaluation is required",
                ],
                "Conclusion": [
                    "It can be implemented if records are consistent across generations.",
                    "Short generation interval and many offspring make poultry suitable for pure-line systems.",
                    "Long generation interval makes cattle breeding slower and more expensive.",
                    "Elite status requires high EBV, low inbreeding, good health, and good reproduction.",
                    "Without accurate sire, dam, and phenotype data, analysis quality decreases.",
                    "Every mating should be checked before implementation to reduce offspring inbreeding.",
                    "New offspring data must be added and recalculated every generation.",
                ],
            })

            st.download_button(
                "Download Real-World Implementation Summary CSV",
                implementation_summary.to_csv(index=False).encode("utf-8"),
                "real_world_pedigree_implementation_summary.csv",
                "text/csv",
                use_container_width=True,
            )


        with tabs[8]:
            render_section_header(
                "Advanced Breeding Management",
                "Additional professional breeding modules. These modules use optional columns when available and show guidance when the current sample data is incomplete.",
                "🧭"
            )
            render_tab_help(
                "Advanced Breeding",
                [
                    "Advanced modules for age, reproduction, health, genomic, economic, trend, calendar, and offspring recording.",
                    "If a column is unavailable, the system only shows a checklist instead of failing."
                ]
            )

            st.markdown("### 1. Missing Advanced Data Checklist")
            missing_cols_df = summarize_missing_advanced_columns(res_display_data)
            st.dataframe(missing_cols_df, hide_index=True, use_container_width=True)

            st.markdown("### 2. Final Breeding Decision Labels")
            decision_cols = [
                "Animal_ID", "Sex_Role", "Age_Eligibility", "Reproductive_Decision",
                "Health_Defect_Decision", "EBV_Reliability_Level", "Combined_Genetic_Value",
                "Expected_Profit", "Economic_Status", "Final_Breeding_Decision"
            ]
            decision_cols = [c for c in decision_cols if c in res_display_data.columns]
            st.dataframe(
                res_display_data[decision_cols].sort_values("Final_Breeding_Decision"),
                hide_index=True,
                use_container_width=True,
            )

            decision_summary = res_display_data["Final_Breeding_Decision"].value_counts().reset_index()
            decision_summary.columns = ["Final_Breeding_Decision", "Total"]
            st.dataframe(decision_summary, hide_index=True, use_container_width=True)

            st.markdown("### 3. Age, Reproduction, Health, and Defect Modules")
            ar_cols = [
                "Animal_ID", "Sex_Role", "Age", "Age_Eligibility",
                "Reproductive_Status", "Reproductive_Decision",
                "Health_Status", "Defect_Status", "Health_Defect_Decision"
            ]
            ar_cols = [c for c in ar_cols if c in res_display_data.columns]
            st.dataframe(res_display_data[ar_cols], hide_index=True, use_container_width=True)

            if "Age" not in res_display_data.columns:
                st.info("Age column is not available. Add an `Age` column to activate age eligibility rules.")
            if "Reproductive_Status" not in res_display_data.columns:
                st.info("Reproductive_Status column is not available. Add it to screen fertility and readiness.")
            if "Health_Status" not in res_display_data.columns and "Defect_Status" not in res_display_data.columns:
                st.info("Health_Status or Defect_Status columns are not available. Add them to prevent unhealthy animals from being recommended.")

            st.markdown("### 4. EBV Reliability and Genomic Support")
            reliability_cols = [
                "Animal_ID", "EBV", "EBV_Reliability_%", "EBV_Reliability_Level",
                "Genomic_EBV", "Genomic_Reliability", "Genomic_Status", "Combined_Genetic_Value"
            ]
            reliability_cols = [c for c in reliability_cols if c in res_display_data.columns]
            st.dataframe(
                res_display_data[reliability_cols].sort_values("Combined_Genetic_Value", ascending=False),
                hide_index=True,
                use_container_width=True,
            )

            st.caption("EBV Reliability here is a practical proxy based on pedigree completeness, phenotype availability, generation record, sex record, and progeny count. It is not a formal BLUP reliability.")

            st.markdown("### 5. Generation Trend")
            generation_trend = calculate_generation_trend(res_display_data)
            if generation_trend.empty:
                st.info("Generation trend is not available. Add a `Generation` column such as Founder, G0, G1, GGPS, GPS, PS, or FS.")
            else:
                st.dataframe(generation_trend, hide_index=True, use_container_width=True)
                st.line_chart(generation_trend.set_index("Generation")[["Average_F", "Average_EBV"]])

            st.markdown("### 6. Line Performance Comparison")
            line_performance = calculate_line_performance(res_display_data)
            if line_performance.empty:
                st.info("Line performance comparison is not available. Add a `Line` column such as Line A, Line B, Line C, or Line D.")
            else:
                st.dataframe(line_performance, hide_index=True, use_container_width=True)
                st.bar_chart(line_performance.set_index("Line")["Average_Selection_Index"])

            st.markdown("### 7. Economic Projection")
            econ_cols = [
                "Animal_ID", "Feed_Cost", "Production_Value", "Replacement_Cost",
                "Culling_Value", "Expected_Profit", "Economic_Status", "Final_Breeding_Decision"
            ]
            econ_cols = [c for c in econ_cols if c in res_display_data.columns]
            st.dataframe(res_display_data[econ_cols], hide_index=True, use_container_width=True)
            if "Feed_Cost" not in res_display_data.columns and "Production_Value" not in res_display_data.columns:
                st.info("Economic projection needs optional columns such as Feed_Cost, Production_Value, Replacement_Cost, or Culling_Value.")

            st.markdown("### 8. Mating Calendar")
            calendar_pairs = simulate_mating_pairs(
                res_display_data,
                matrix_df,
                h2_value=h2,
                depression_rate_value=depression_rate,
                max_offspring_f=6.25,
                max_pairs=20,
            )
            calendar_pairs = apply_blocked_mating_rules_to_pairs(
                calendar_pairs,
                res_display_data,
                matrix_df,
                max_safe_f=6.25,
            )
            calendar_pairs = calendar_pairs[calendar_pairs["Mating_Rule_Status"] != "Blocked"].copy()

            mating_calendar = build_mating_calendar(
                calendar_pairs,
                start_date=calendar_start_date,
                interval_days=mating_interval_days,
                expected_days=expected_offspring_days,
            )

            if mating_calendar.empty:
                st.warning("No safe mating calendar could be generated from the current data.")
            else:
                st.dataframe(mating_calendar, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download Mating Calendar CSV",
                    mating_calendar.to_csv(index=False).encode("utf-8"),
                    "mating_calendar.csv",
                    "text/csv",
                    use_container_width=True,
                )

            st.markdown("### 9. Auto-Generated Next Generation IDs")
            id_col1, id_col2, id_col3 = st.columns(3)
            with id_col1:
                id_prefix = st.text_input("Offspring ID Prefix", value="NEXTGEN")
            with id_col2:
                id_generation = st.text_input("Generation Label", value="G1")
            with id_col3:
                id_count = st.number_input("Number of IDs", min_value=1, max_value=1000, value=20, step=1)

            next_ids = generate_next_generation_ids(id_prefix, id_generation, id_count)
            st.dataframe(next_ids, hide_index=True, use_container_width=True)
            st.download_button(
                "Download Next Generation ID Template CSV",
                next_ids.to_csv(index=False).encode("utf-8"),
                "next_generation_id_template.csv",
                "text/csv",
                use_container_width=True,
            )

            st.markdown("### 10. Pen / Cage / Batch Management")
            location_cols = [
                "Animal_ID", "Sex_Role", "Farm", "House", "Pen", "Cage", "Group", "Batch",
                "Line", "Generation", "Final_Breeding_Decision"
            ]
            location_cols = [c for c in location_cols if c in res_display_data.columns]
            if any(c in res_display_data.columns for c in ["Pen", "Cage", "Batch", "Farm", "House", "Group"]):
                st.dataframe(res_display_data[location_cols], hide_index=True, use_container_width=True)
            else:
                st.info("Location management needs optional columns such as Farm, House, Pen, Cage, Group, or Batch.")

            st.markdown("### 11. Offspring Recording Template")
            offspring_template = pd.DataFrame({
                "Mating_ID": ["MATE_0001"],
                "Sire_ID": ["example_sire"],
                "Dam_ID": ["example_dam"],
                "Offspring_ID": ["NEXTGEN_G1_0001"],
                "Birth_Date": [pd.Timestamp.now().date().strftime("%Y-%m-%d")],
                "Sex": ["Male/Female"],
                "Line": ["Line A"],
                "Generation": ["G1"],
                "Phenotype": [""],
                "Survival_Status": ["Alive"],
                "Health_Status": ["Healthy"],
                "Notes": [""],
            })
            st.dataframe(offspring_template, hide_index=True, use_container_width=True)
            st.download_button(
                "Download Offspring Recording Template CSV",
                offspring_template.to_csv(index=False).encode("utf-8"),
                "offspring_recording_template.csv",
                "text/csv",
                use_container_width=True,
            )

        render_footer()

    except Exception as e:
        st.error("The application could not continue because the uploaded data or selected configuration needs correction.")
        st.write("Please review the data writing rules below, then upload the corrected file again.")

        show_input_validation_messages({
            "errors": [
                "Unexpected processing issue. Please check column mapping, duplicated Animal_ID, parent IDs, phenotype values, and pedigree cycles."
            ],
            "warnings": []
        })

        with st.expander("Technical detail for developer", expanded=False):
            st.write(str(e))


if __name__ == "__main__":
    main()
