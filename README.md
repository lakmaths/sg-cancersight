# SG-CancerSight
## Singapore Cancer Outcomes & Healthcare Utilisation Analysis

**Author:** Dr. Lakshmi C. | PhD Mathematics | Healthcare Operations Research  
**Target Role:** Data Analyst (DDOIT), National Cancer Centre Singapore (NCCS)  
**Dataset:** SEER Breast Cancer Registry (4,024 patients, 15 clinical variables)

---

## Project Overview

SG-CancerSight is a reproducible cancer data analytics portfolio project demonstrating competency across the full analytical lifecycle required for population health and health economics research at NCCS:

| JD Requirement | Implementation |
|---|---|
| Access, extract, clean large-scale datasets | `scripts/etl_pipeline.py` |
| Harmonise and link multi-source data | Column renaming, sentinel recoding, imputation |
| Reproducible data pipelines | Logged ETL with full audit trail |
| Data quality checks and validation | `tests/test_data_quality.py` (25 tests) |
| Statistical analysis & survival analysis | Phase 2 (coming) |
| Healthcare utilisation & cost analyses | Phase 3 (coming) |
| High-quality outputs for manuscripts | Phase 4 (coming) |
| Data governance and version control | `DATA_GOVERNANCE.md`, git history |

---

## Project Structure

```
sg-cancersight/
├── data/
│   ├── raw/            seer_raw.csv          (original, never modified)
│   └── processed/      seer_clean.csv        (analysis-ready)
│                       data_dictionary.json  (auto-generated)
│                       etl_run.log           (full audit trail)
├── scripts/
│   └── etl_pipeline.py                       (Phase 1 ETL)
├── notebooks/          (Phase 2–4 analyses)
├── outputs/
│   └── figures/        phase1_eda.png
├── tests/
│   └── test_data_quality.py                  (25 pytest checks)
└── requirements.txt
```

---

## Phase 1 — Data Management & Wrangling

### Cleaning steps performed
1. **Invalid ages** (< 18 or > 100): 4 rows removed
2. **Negative survival months**: 5 rows removed  
3. **Tumour size sentinel (999)**: 3 values recoded → NaN
4. **Column typo**: `Reginol Node Positive` → `Regional Node Positive`
5. **Missing tumour size** (3.2%): imputed with median within T Stage group
6. **Missing node count** (3.1%): imputed with overall median
7. **Missing grade** (3.2%): imputed with mode
8. **Node logical constraint**: nodes positive capped at nodes examined (232 rows)

### Derived variables created
| Variable | Description |
|---|---|
| `Status_Code` | Binary outcome: 1 = Dead (event), 0 = Alive (censored) |
| `Age Group` | 10-year bands: <40, 40–49, 50–59, 60–69, 70+ |
| `Stage Group` | Simplified: Stage I / Stage II / Stage III |
| `Receptor Status` | Hormone Receptor+ vs Triple Negative / HR- |

### Data quality results
- **25 / 25 pytest tests pass**
- 0 missing values in any critical analysis column post-cleaning
- No invalid category values, no sentinel values, no logical violations

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run ETL pipeline
python scripts/etl_pipeline.py

# Run data quality tests
pytest tests/test_data_quality.py -v
```

---

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
pytest>=7.0
```

---

## Data Governance Note

This project uses a synthetic research replica of the SEER dataset structure.  
In a real NCCS deployment, all patient data would be handled under:
- Singapore PDPA (Personal Data Protection Act)
- MOH Data Governance Framework  
- IRB/DSRB approval protocols  
- Role-based access controls and audit logging (implemented in `etl_run.log`)

See `DATA_GOVERNANCE.md` for the full governance framework.
