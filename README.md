# SSLC Risk Batch-Calculator

**English** | **[中文版本](README_zh.md)**

This tool does not rely on a web page or `files/app.py`: models, scripts, dependencies, input templates, and tests are all located in this directory.

## Installation and Usage

Download via:

```powershell
git clone https://github.com/KokuYu-sysu/SSLC_calculator.git
```

Open PowerShell in this directory:

> Note: To avoid conflicts with your global Python environment, it is recommended to create a virtual environment for isolation.

```powershell
python -m pip install -r requirements.txt
python batch_risk.py input.csv
```

By default, it calculates 5-year risk and generates three files:

- `input_with_risk_5y.csv`: Retains all original columns with `risk_5y` appended at the end;
- `input_errors.csv`: An error report where each invalid variable occupies one row;
- `input_schema_report.json`: A consistency report of input columns against model/transformation rules.

To specify a different time window, e.g., 7 years:

```powershell
python batch_risk.py input.csv --time 7
```

The result column will be `risk_7y`. The bundled models collectively support approximately 2–10 years; values beyond this range will be rejected before computation to avoid extrapolating the model into long-term risk. Decimal time windows are also supported, e.g., `--time 7.5` produces a result column `risk_7_5y`.

## Input Format

Use `input_template.csv` as a reference template, fill in your data, and save it as CSV. Each row may belong to a different sex and smoking subgroup. `sex` is required and must be `female` or `male`; `smoking` is required and must be `never` or `ever`. When `smoking=ever`, `smoking_status` is required and must be `former` or `current`.

Only fill in variables that are actually used by the subgroup of the current row; unused variables may be left empty.

| Field | Allowed Values/Range | Description |
| --- | --- | --- |
| `age` | 40–80 | Age |
| `education` | Integer 1–6 | Education level |
| `occupation` | Integer 1–10 | 1 Professional/Technical; 2 Management; 3 Commercial; 4 Farmer; 5 Clerical/Worker; 6 Service; 7 Homemaker; 8 Retired; 9 Unemployed; 10 Other |
| `marital` | `married`, `unmarried`, `others` | Marital status |
| `alcohol` | `never`, `former`, `current` | Alcohol consumption status |
| `height`, `weight` | `(0,250]` cm, `(0,200]` kg respectively | Used to calculate BMI |
| `years_smoking`, `cigarettes_per_day`, `years_since_quit`, `passive_smoking_years` | ≥ 0 | Smoking and passive smoking years; cigarettes/day are automatically converted to packs/day |
| `cancer_history`, `copd`, `chronic_resp`, `diabetes`, `hypertension`, `chd`, `liver`, `tb`, `exercise` | `yes` or `no` | Medical history and regular exercise |
| `family_cancer_count` | Integer 0–20 | Number of first-degree relatives with cancer |
| `pickled`, `protein`, `vegetable` | `never`, `sometimes`, `frequently` | Never = 0; others = 1 |
| `oil_smoke` | `none`, `low`, `moderate`, `high` | None/little = 0; considerable/lots = 1 |
| `menopause` | `premenopausal`, `<40`, `40-45`, `45-50`, `50-55`, `>=55` | Female menopausal age |
| `breastfeeding` | `nulliparous`, `parous_none`, `<6`, `6-12`, `>=12` | Total breastfeeding duration; the first two both map to the same model category |

## Pre-One-Hot-Encoded Tables

The script also accepts tables where model variable names already exist as columns, such as `occupation_1`, `occupation_63`, `marital_have`, `alcohol_no`, `menopause_age_50`, or `bfeed_12`. These columns are strictly validated against `input_features` in the PKL before being passed to the model.

- One-hot columns within a categorical group must be complete, contain only 0/1 values, and be mutually exclusive;
- When both categorical and one-hot columns are provided, the two representations must be consistent;
- When reference categories unused by the model are included, all zeros represent that reference category;
- When column names or values violate the rules, the row's risk is left blank and written to the error report — no silent zero-filling occurs.

Columns such as `patient_id`, outcomes, or other covariates may be retained; by default they are not passed to the model, but will be marked as `unknown_columns` in the `*_schema_report.json`. To enforce that every column is recognized:

```powershell
python batch_risk.py input.csv --strict-columns
```

## Output, Errors, and Safety

The error CSV always contains `source_row_number`, `error_code`, `message`, `subgroup`, and `requested_time_years`. When a subject has multiple issues, there will be multiple rows, facilitating one-time correction of all variables.

Output paths can be customized:

```powershell
python batch_risk.py input.csv --output result.csv --errors problems.csv --schema-report schema.json
```

The script refuses to overwrite the input file, existing output files, or existing risk columns of the same name. Input is read as `utf-8-sig` by default; other encodings exported from hospital systems can be explicitly specified, e.g., `--encoding gb18030`. Output uses UTF-8 BOM, which can be opened directly in Excel with proper Chinese character display.

`models.pkl` is a Python pickle; only use model files distributed with this tool from trusted sources — do not replace it with `.pkl` files from unknown sources.

If you encounter the error: `[existing_output]: refusing to overwrite an existing output file`, you need to delete the previously generated files and re-run the command.

## More Help

```powershell
python batch_risk.py --help
```

## License

MIT
