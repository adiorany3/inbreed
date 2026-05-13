# 🐂 Livestock Breeding & Inbreeding Analytics

A professional-grade web application built with Streamlit for advanced livestock genetic analysis, pedigree tracking, and selection management.

## ✨ Key Features
- **Accurate Inbreeding Calculation**: Uses the Additive Relationship Matrix (A-Matrix) algorithm to ensure precise inbreeding coefficients ($F$) and kinship values.
- **Estimated Breeding Value (EBV)**: Calculates genetic potential based on phenotype data and heritability.
- **Selection Response (R)**: Predicts genetic progress for the next generation using custom intensity and heritability parameters.
- **Hardy-Weinberg Equilibrium (HWE)**: Diagnoses population-level genetic health and provides strategic management advice.
- **Automated Selection & Culling**: Identify top 25% candidates for breeding and priority animals for culling based on genetic risk.
- **Professional PDF Reports**: Generate comprehensive technical reports including regression analysis, full pedigree tables, and strategic recommendations.
- **Interactive Visualizations**: Dynamic pedigree trees (Graphviz), distribution histograms, and inbreeding vs. phenotype scatter plots.
- **Modern UI**: Fully responsive design with automatic Dark/Light mode support.

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adiorany3/inbreed.git
   cd inbreed
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📊 Data Input Format

Upload a CSV or Excel file with the following columns:
- `Animal_ID`: Unique identifier for each animal.
- `Sire_ID`: ID of the father (use `-` or empty for unknown).
- `Dam_ID`: ID of the mother (use `-` or empty for unknown).
- `Phenotype` (Optional): Numerical performance data (e.g., weight, ADG) for EBV and selection response analysis.

### Example Data:
| Animal_ID | Sire_ID | Dam_ID | Phenotype |
|-----------|---------|--------|-----------|
| BULL_01   | -       | -      | 550       |
| COW_01    | -       | -      | 420       |
| CALF_A    | BULL_01 | COW_01 | 480       |

## 🧪 Technical Analytics Included
- **Linear Regression**: Quantifies Inbreeding Depression (performance loss per 1% increase in $F$).
- **A-Matrix Calculation**: Optimized Henderson's method for large-scale pedigrees.
- **Heterosis Analysis**: Measures hybrid vigor for crossbreeding strategies.

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

## 👨‍💻 Developer
Developed by **Galuh Adi Insani**  
Affiliation: **Universitas Gadjah Mada**

---
Built with Python, Streamlit, and ReportLab.
