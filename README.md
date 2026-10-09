# SSLC Risk Batch-Calculator

**English** | **[中文版本](README_zh.md)**

Batch calculation of lung cancer mortality risk from clinical CSV tables. The bundled `models.pkl` is the updated `model_groups_dict_de_compact.pkl`, containing the fitted model, subgroup-specific scaler, and ordered feature lists for each of the four sex/smoking subgroups. This directory works independently of the website.

## Installation and Usage

Open PowerShell in the directory you like. Download via:

```powershell
git clone https://github.com/KokuYu-sysu/SSLC_calculator.git
cd SSLC_calculator
```

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

The result column will be `risk_7y`. The updated models share the inclusive time range **1.4839228235646376–10.55532786885245 years**, so 5, 7, and 10 years are supported. The script reads this range from the fitted models, including `model.time_bins` for XGBSE. Values outside the shared range are rejected before calculation. Decimal horizons are supported, e.g., `--time 7.5` produces `risk_7_5y`. CoxNet uses its fitted survival step function; XGBSE uses the latest available time bin at or before the requested horizon, without interpolation.

Risks are probabilities from 0 to 1, not percentages: `0.01` means 1% lung cancer mortality risk within the requested period.

## Input Format

Use `input_template.csv` as a reference template, fill in your data, and save it as CSV. Each row may belong to a different sex and smoking subgroup. `sex` is required and must be `female` or `male`; `smoking` is required and must be `never` or `ever`. When `smoking=ever`, `smoking_status` is required and must be `former` or `current`.

The updated models use more input variables than the previous version. For categorical input, fill in the common fields below for every row. Female rows additionally require `first_period_age`, `menopause`, `first_live_birth_age`, and `breastfeeding`; these four fields may be blank for male rows. Smokers additionally require `years_smoking` and `cigarettes_per_day`. Former smokers must provide `years_since_quit`; current smokers may leave it blank or enter 0.

For never smokers, `smoking_status`, `years_smoking`, `cigarettes_per_day`, and `years_since_quit` may be blank. Smoking amounts/durations required by their model are derived as 0; a supplied nonzero value in such a model input is a row error. For both female and male current smokers, quitting duration is derived as 0, and a nonzero supplied value is rejected. Missing medical histories or other measured variables are not automatically filled with 0.

| Field | Allowed Values/Range | Description |
| --- | --- | --- |
| `age` | 40–80 | Age |
| `education` | Integer 1–6 | 1 No formal education; 2 Primary; 3 Middle school; 4 High/vocational school; 5 College; 6 University or above |
| `occupation` | Integer 1–10 | 1 Professional/Technical; 2 Management; 3 Commercial; 4 Farmer; 5 Clerical/Worker; 6 Service; 7 Homemaker; 8 Retired; 9 Unemployed; 10 Other |
| `marital` | `married`, `unmarried`, `others` | Marital status |
| `alcohol` | `never`, `former`, `current` | Alcohol consumption status |
| `height`, `weight` | `(0,250]` cm, `(0,200]` kg respectively | Used to calculate BMI |
| `years_smoking`, `cigarettes_per_day`, `years_since_quit`, `passive_smoking_years` | ≥ 0 | Smoking and passive smoking years; cigarettes/day are automatically converted to packs/day |
| `cancer_history`, `copd`, `chronic_resp`, `diabetes`, `hypertension`, `chd`, `liver`, `tb`, `exercise` | `yes` or `no` | Medical history and regular exercise |
| `family_cancer_count` | Integer 0–20 | Number of first-degree relatives with cancer |
| `pickled`, `protein`, `vegetable` | `never`, `sometimes`, `frequently` | Never = 0; others = 1 |
| `oil_smoke` | `none`, `low`, `moderate`, `high` | None/little = 0; considerable/lots = 1 |
| `first_period_age` | Finite number 0–80 | Female age at menarche in years; maps to `first_period_age_new` |
| `menopause` | `premenopausal`, `<40`, `40-45`, `45-50`, `50-55`, `>=55` | Female menopausal age |
| `first_live_birth_age` | `nulliparous`, `<20`, `20-25`, `25-30`, `>=30` | Female age at first live birth; maps to `live_birth_age_*` one-hot columns |
| `breastfeeding` | `nulliparous`, `parous_none`, `<6`, `6-12`, `>=12` | Total breastfeeding duration; the first two both map to the same model category |

For categorical input, `first_live_birth_age=nulliparous` must be paired with `breastfeeding=nulliparous`. Parous rows must use `parous_none`, `<6`, `6-12`, or `>=12`. Their model encoding cannot distinguish nulliparous from parous without breastfeeding using `bfeed_1` alone.

Use the study definitions when classifying clinical data: `smoking=never` includes occasional smokers with fewer than 100 lifetime cigarettes; `alcohol=never` includes drinking less than weekly; regular exercise means at least three sessions per week of 30 minutes each. `chd` includes coronary heart disease, heart failure, stroke, peripheral vascular disease, rheumatic/congenital heart disease, and cardiomyopathy. `family_cancer_count` counts parents, children, and siblings.

## Pre-One-Hot-Encoded Tables

The script also accepts numeric model inputs such as `BMI_new`, `education.y`, and `first_period_age_new`, and complete one-hot groups such as `occupation_*`, `marital_*`, `alcohol_*`, `smoking_*`, `menopause_age_*`, `live_birth_age_*`, and `bfeed_*`. `sex` and `smoking` remain required for subgroup selection, and ever smokers still need `smoking_status`. The script validates scaler and estimator feature names and order, scales the complete `input_features`, then selects `model_features` in the saved order.

- One-hot columns within a categorical group must be complete, contain only 0/1 values, and be mutually exclusive;
- When both categorical and one-hot columns are provided, the two representations must be consistent;
- For a family with a reference column omitted from the saved `input_features`, all 0s in the remaining columns can represent that reference category. When the saved inputs contain the full family, exactly one column must be 1. Columns omitted only from `model_features` are still required by the scaler;
- When column names or values violate the rules, the row's risk is left blank and written to the error report — no silent zero-filling occurs.

The first-birth mapping is `nulliparous` → `live_birth_age_0`, `<20` → `live_birth_age_20`, `20-25` → `live_birth_age_25`, `25-30` → `live_birth_age_30`, and `>=30` → `live_birth_age_30more`. A nulliparous one-hot row must also have `bfeed_1=1`.

`gender`, used by the male never-smoker model, is automatically derived from `sex` using female=1/male=0. You do not need to add it to a categorical CSV; if supplied for that model, it must agree with `sex`. `smoking_pack_day_new` is already in packs/day; only `cigarettes_per_day` is divided by 20.

Columns such as `id`, `patient_id`, outcomes, or other covariates may be retained; by default they are preserved in the output and not passed to the model, but will be marked as `unknown_columns` in the `*_schema_report.json`. To enforce that every column is recognized:

```powershell
python batch_risk.py input.csv --strict-columns
```

## Output, Errors, and Safety

Valid rows continue to be calculated even when other rows fail. Invalid rows are retained with a blank risk. The error CSV contains `source_row_number`, `error_code`, `message`, `subgroup`, and `requested_time_years`; the first data row is numbered 2, including the header. A row may have multiple error entries.

The schema report identifies recognized, semantic, model-ready, and unknown columns, plus `missing_by_subgroup` and `missing_routing_columns`. It is built before row prediction. Column availability alone does not establish valid values: former-smoker quitting duration and subgroup-specific constraints are checked row by row.

Output paths can be customized:

```powershell
python batch_risk.py input.csv --output result.csv --errors problems.csv --schema-report schema.json
```

The script refuses to overwrite the input file, existing output files, or existing risk columns of the same name. Input is read as `utf-8-sig` by default; other encodings exported from hospital systems can be explicitly specified, e.g., `--encoding gb18030`. Output uses UTF-8 BOM, which can be opened directly in Excel with proper Chinese character display.

`models.pkl` is a Python pickle; only use model files distributed with this tool from trusted sources — do not replace it with `.pkl` files from unknown sources.

If `[existing_output]: refusing to overwrite an existing output file` occurs, move the previous outputs or choose unused paths for all three output files.

## Verification

Run the bundled regression tests from this directory:

```powershell
python -B -m unittest discover -s tests -p test_batch_risk.py -v
```

They check the updated artifact, all four subgroups, reference risks at 5/7/10 years, categorical and numeric/one-hot inputs, new female fields, smoking and sex consistency, missing-column reports, and batch CSV/error outputs. These checks verify inference integration; they do not establish clinical calibration or external validity of the trained models.

During the 2026-10-09 update, 160 synthetic clinical profiles (40 per subgroup) were also compared with the updated website encoder and the original fitted models/scalers. All feature encodings matched; the 480 risk values at 5, 7.5, and 10 years had a maximum absolute difference of 0. The bundled model is a byte-for-byte copy of the supplied updated compact artifact; its SHA-256 is `def0d7ae2f87ae924703c767e22a1765cc307f7aa5ea20159c4d3eb08465838c`.

## More Help

```powershell
python batch_risk.py --help
```

## License

MIT
