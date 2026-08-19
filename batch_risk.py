from __future__ import annotations

import argparse
import pickle
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd


class BatchInputError(ValueError):
    """Raised when a batch input value violates the public CLI contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ModelContractError(RuntimeError):
    """Raised when the trusted artifact is incomplete or incompatible."""


MODEL_GROUP_KEYS = {
    "female_nonsmoker",
    "female_smoker",
    "male_nonsmoker",
    "male_smoker",
}
MODEL_PACKAGE_KEYS = {
    "model",
    "scaler",
    "input_features",
    "model_features",
    "metadata",
}
COXNET_IDENTITY = ("sksurv.linear_model.coxnet", "CoxnetSurvivalAnalysis")
XGBSE_IDENTITY = ("xgbse._debiased_bce", "XGBSEDebiasedBCE")
EXPECTED_MODEL_IDENTITIES = {
    "female_nonsmoker": COXNET_IDENTITY,
    "female_smoker": XGBSE_IDENTITY,
    "male_nonsmoker": COXNET_IDENTITY,
    "male_smoker": COXNET_IDENTITY,
}
SUBGROUPS = {
    ("female", "never"): "female_nonsmoker",
    ("female", "ever"): "female_smoker",
    ("male", "never"): "male_nonsmoker",
    ("male", "ever"): "male_smoker",
}
OCCUPATION_FEATURE = {
    "1": "occupation_1", "2": "occupation_2", "3": "occupation_4",
    "4": "occupation_5", "5": "occupation_63", "6": "occupation_7",
    "7": "occupation_8", "8": "occupation_10", "9": "occupation_11",
    "10": "occupation_9",
}
MENOPAUSE_FEATURE = {
    "premenopausal": "menopause_age_0", "<40": "menopause_age_40",
    "40-45": "menopause_age_45", "45-50": "menopause_age_50",
    "50-55": "menopause_age_55", ">=55": "menopause_age_55more",
}
BREASTFEEDING_FEATURE = {
    "nulliparous": "bfeed_1", "parous_none": "bfeed_1",
    "<6": "bfeed_6", "6-12": "bfeed_12", ">=12": "bfeed_12more",
}
ONE_HOT_FAMILIES = {
    "occupation": tuple(OCCUPATION_FEATURE.values()),
    "marital": ("marital_have", "marital_no", "marital_other"),
    "alcohol": ("alcohol_current", "alcohol_ever", "alcohol_no"),
    "smoking": ("smoking_current", "smoking_ever", "smoking_no"),
    "menopause": tuple(dict.fromkeys(MENOPAUSE_FEATURE.values())),
    "breastfeeding": ("bfeed_1", "bfeed_6", "bfeed_12", "bfeed_12more"),
}
FAMILY_SEMANTIC_FEATURES = {
    "marital": {"married": "marital_have", "unmarried": "marital_no", "others": "marital_other"},
    "alcohol": {"never": "alcohol_no", "former": "alcohol_ever", "current": "alcohol_current"},
}


@dataclass(frozen=True)
class RowIssue:
    code: str
    message: str


@dataclass(frozen=True)
class EncodedRow:
    subgroup: str
    features: dict[str, float]


@dataclass(frozen=True)
class BatchResult:
    output_path: Path
    errors_path: Path
    schema_report_path: Path
    valid_rows: int
    invalid_rows: int


class _TrustedNumpyCompatibilityUnpickler(pickle.Unpickler):
    """Read the approved NumPy 2 artifact on a NumPy 1 compatible runtime."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core."):
            module = "numpy.core." + module[len("numpy._core.") :]
        return super().find_class(module, name)


def _model_identity(model: object) -> tuple[str, str]:
    model_type = type(model)
    return model_type.__module__, model_type.__qualname__


def _contract_error(subgroup: str, detail: str) -> None:
    raise ModelContractError(f"{subgroup}: {detail}")


def _validate_feature_names(subgroup: str, field: str, features: object) -> list[str]:
    if not isinstance(features, (list, tuple)) or not features:
        _contract_error(subgroup, f"{field} must be a nonempty ordered list")
    names = list(features)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        _contract_error(subgroup, f"{field} must contain only nonempty strings")
    if len(set(names)) != len(names):
        _contract_error(subgroup, f"{field} names must be unique")
    return names


def _integer_contract_value(subgroup: str, field: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        _contract_error(subgroup, f"{field} must be an integer")
    return int(value)


def _model_feature_count(subgroup: str, model: object, identity: tuple[str, str]) -> int:
    if identity == COXNET_IDENTITY:
        try:
            return _integer_contract_value(subgroup, "model.n_features_in_", model.n_features_in_)
        except AttributeError as exc:
            _contract_error(subgroup, f"cannot read model.n_features_in_: {exc}")
    try:
        count = model.feature_extractor.bst.num_features()
    except Exception as exc:  # pragma: no cover - exercised by corrupt artifacts
        _contract_error(subgroup, f"cannot read XGBSE model feature count: {type(exc).__name__}: {exc}")
    return _integer_contract_value(subgroup, "XGBSE model feature count", count)


def validate_model_groups(groups: object) -> Mapping[str, Mapping[str, object]]:
    """Validate all structural and feature-order contracts stored in the PKL."""
    if not isinstance(groups, Mapping):
        _contract_error("<root>", "model groups must be a mapping")
    if set(groups) != MODEL_GROUP_KEYS:
        _contract_error(
            "<root>",
            f"unexpected model groups; missing={sorted(MODEL_GROUP_KEYS - set(groups))!r}, "
            f"extra={sorted(set(groups) - MODEL_GROUP_KEYS)!r}",
        )

    for subgroup in sorted(MODEL_GROUP_KEYS):
        package = groups[subgroup]
        if not isinstance(package, Mapping) or set(package) != MODEL_PACKAGE_KEYS:
            _contract_error(subgroup, "package must contain exactly the compact-model keys")
        input_features = _validate_feature_names(subgroup, "input_features", package["input_features"])
        model_features = _validate_feature_names(subgroup, "model_features", package["model_features"])
        if not set(model_features).issubset(input_features):
            _contract_error(subgroup, "model_features must be a subset of input_features")
        positions = [input_features.index(feature) for feature in model_features]
        if positions != sorted(positions):
            _contract_error(subgroup, "model_features must preserve input feature order")

        scaler = package["scaler"]
        try:
            scaler_features = list(np.asarray(scaler.feature_names_in_, dtype=object))
            scaler_count = _integer_contract_value(subgroup, "scaler.n_features_in_", scaler.n_features_in_)
        except AttributeError as exc:
            _contract_error(subgroup, f"cannot read scaler contract: {exc}")
        if scaler_features != input_features or scaler_count != len(input_features):
            _contract_error(subgroup, "scaler feature names/count disagree with input_features")

        model = package["model"]
        identity = _model_identity(model)
        if identity != EXPECTED_MODEL_IDENTITIES[subgroup]:
            _contract_error(subgroup, f"unexpected model class {'.'.join(identity)}")
        if _model_feature_count(subgroup, model, identity) != len(model_features):
            _contract_error(subgroup, "model feature count disagrees with model_features")
    return groups


def load_model_groups(path: str | Path) -> Mapping[str, Mapping[str, object]]:
    """Load the bundled, trusted compact-model artifact exactly once per run."""
    artifact_path = Path(path)
    try:
        with artifact_path.open("rb") as handle:
            groups = _TrustedNumpyCompatibilityUnpickler(handle).load()
    except OSError as exc:
        raise ModelContractError(f"cannot open model artifact {artifact_path}: {exc}") from exc
    except (pickle.UnpicklingError, EOFError, ImportError, AttributeError) as exc:
        raise ModelContractError(f"cannot unpickle model artifact {artifact_path}: {exc}") from exc
    validated = validate_model_groups(groups)
    for package in validated.values():
        if _model_identity(package["model"]) == XGBSE_IDENTITY:
            package["model"].n_jobs = 1
    return validated


def _group_time_range(subgroup: str, package: Mapping[str, object]) -> tuple[Decimal, Decimal]:
    model = package["model"]
    identity = _model_identity(model)
    if identity == COXNET_IDENTITY:
        try:
            times = np.asarray(model._baseline_models[0].unique_times_, dtype=float)
        except Exception as exc:  # pragma: no cover - corrupt trained state
            _contract_error(subgroup, f"cannot read CoxNet baseline times: {exc}")
    elif identity == XGBSE_IDENTITY:
        try:
            times = np.asarray(package["metadata"]["time_bins"], dtype=float)
        except Exception as exc:  # pragma: no cover - corrupt metadata
            _contract_error(subgroup, f"cannot read XGBSE time bins: {exc}")
    else:  # pragma: no cover - guarded by validate_model_groups
        _contract_error(subgroup, "unsupported survival model")
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        _contract_error(subgroup, "survival time bins must be finite and strictly increasing")
    return Decimal(str(float(times[0]))), Decimal(str(float(times[-1])))


def common_supported_time_range(groups: Mapping[str, Mapping[str, object]]) -> tuple[Decimal, Decimal]:
    """Return the inclusive time range supported by every available subgroup."""
    validate_model_groups(groups)
    ranges = [_group_time_range(key, groups[key]) for key in sorted(MODEL_GROUP_KEYS)]
    lower = max(start for start, _ in ranges)
    upper = min(end for _, end in ranges)
    if lower > upper:
        raise ModelContractError("model time ranges do not overlap")
    return lower, upper


def validate_requested_time(groups: Mapping[str, Mapping[str, object]], value: object) -> Decimal:
    """Reject a requested horizon outside the models' shared observed range."""
    requested = parse_time_years(value)
    lower, upper = common_supported_time_range(groups)
    if requested < lower or requested > upper:
        raise BatchInputError(
            "unsupported_time",
            f"--time must be within the common model range [{lower}, {upper}] years",
        )
    return requested


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _has_value(row: Mapping[str, object], field: str) -> bool:
    return field in row and not _is_missing(row[field]) and row[field] != ""


def _add_issue(issues: list[RowIssue], code: str, message: str) -> None:
    issue = RowIssue(code, message)
    if issue not in issues:
        issues.append(issue)


def _text_value(row: Mapping[str, object], field: str, issues: list[RowIssue]) -> str | None:
    if not _has_value(row, field):
        return None
    value = row[field]
    if not isinstance(value, str):
        value = str(value)
    return value


def _finite_number(
    row: Mapping[str, object],
    field: str,
    issues: list[RowIssue],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> float | None:
    if not _has_value(row, field):
        return None
    value = row[field]
    if isinstance(value, (bool, np.bool_)):
        _add_issue(issues, "invalid_value", f"{field} must be numeric")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        _add_issue(issues, "invalid_value", f"{field} must be numeric")
        return None
    if not math.isfinite(number):
        _add_issue(issues, "invalid_value", f"{field} must be finite")
        return None
    if minimum is not None and number < minimum:
        _add_issue(issues, "invalid_value", f"{field} must be at least {minimum}")
        return None
    if maximum is not None and number > maximum:
        _add_issue(issues, "invalid_value", f"{field} must be at most {maximum}")
        return None
    if integer and not number.is_integer():
        _add_issue(issues, "invalid_value", f"{field} must be an integer")
        return None
    return number


def _binary_value(row: Mapping[str, object], field: str, issues: list[RowIssue]) -> float | None:
    if not _has_value(row, field):
        return None
    value = row[field]
    if isinstance(value, str) and value in {"yes", "no"}:
        return float(value == "yes")
    number = _finite_number(row, field, issues, minimum=0, maximum=1)
    if number is not None and number not in {0.0, 1.0}:
        _add_issue(issues, "invalid_value", f"{field} must be yes/no or 0/1")
        return None
    return number


def _semantic_one_hot(
    family: str, row: Mapping[str, object], issues: list[RowIssue]
) -> dict[str, float] | None:
    if family == "occupation":
        value = _finite_number(row, "occupation", issues, minimum=1, maximum=10, integer=True)
        if value is None:
            return None
        selected = OCCUPATION_FEATURE[str(int(value))]
    elif family == "menopause":
        value = _text_value(row, "menopause", issues)
        if value is None:
            return None
        if value not in MENOPAUSE_FEATURE:
            _add_issue(issues, "invalid_value", "menopause has an unsupported category")
            return None
        selected = MENOPAUSE_FEATURE[value]
    elif family == "breastfeeding":
        value = _text_value(row, "breastfeeding", issues)
        if value is None:
            return None
        if value not in BREASTFEEDING_FEATURE:
            _add_issue(issues, "invalid_value", "breastfeeding has an unsupported category")
            return None
        selected = BREASTFEEDING_FEATURE[value]
    elif family == "smoking":
        smoking = _text_value(row, "smoking", issues)
        if smoking is None:
            return None
        if smoking == "never":
            selected = "smoking_no"
        elif smoking == "ever":
            smoking_status = _text_value(row, "smoking_status", issues)
            selected_by_status = {"former": "smoking_ever", "current": "smoking_current"}
            if smoking_status not in selected_by_status:
                _add_issue(issues, "invalid_value", "smoking_status must be former or current for ever smokers")
                return None
            selected = selected_by_status[smoking_status]
        else:
            _add_issue(issues, "invalid_value", "smoking must be never or ever")
            return None
    else:
        value = _text_value(row, family, issues)
        mapping = FAMILY_SEMANTIC_FEATURES[family]
        if value is None:
            return None
        if value not in mapping:
            _add_issue(issues, "invalid_value", f"{family} has an unsupported category")
            return None
        selected = mapping[value]
    return {feature: float(feature == selected) for feature in ONE_HOT_FAMILIES[family]}


def _resolve_one_hot_family(
    family: str,
    row: Mapping[str, object],
    input_features: set[str],
    issues: list[RowIssue],
) -> dict[str, float]:
    family_features = ONE_HOT_FAMILIES[family]
    relevant = tuple(feature for feature in family_features if feature in input_features)
    if not relevant:
        return {}
    declared = tuple(feature for feature in family_features if feature in row)
    values_present = tuple(feature for feature in declared if _has_value(row, feature))
    direct_values: dict[str, float] | None = None
    if values_present:
        expected = family_features if any(feature not in relevant for feature in declared) else relevant
        missing = [feature for feature in expected if not _has_value(row, feature)]
        if missing:
            _add_issue(issues, "invalid_one_hot", f"{family} one-hot columns are incomplete: {missing!r}")
        else:
            parsed: dict[str, float] = {}
            for feature in expected:
                value = _finite_number(row, feature, issues, minimum=0, maximum=1)
                if value is None or value not in {0.0, 1.0}:
                    _add_issue(issues, "invalid_one_hot", f"{family} one-hot values must be 0 or 1")
                    continue
                parsed[feature] = value
            if len(parsed) == len(expected):
                total = sum(parsed.values())
                if len(expected) == len(family_features):
                    valid_total = total == 1.0
                else:
                    valid_total = total in {0.0, 1.0}
                if not valid_total:
                    _add_issue(issues, "invalid_one_hot", f"{family} one-hot values are not mutually exclusive")
                else:
                    direct_values = {feature: parsed[feature] for feature in relevant}

    semantic_values = _semantic_one_hot(family, row, issues)
    if direct_values is not None and semantic_values is not None:
        if any(direct_values[feature] != semantic_values[feature] for feature in relevant):
            _add_issue(issues, "conflicting_value", f"{family} semantic and one-hot values disagree")
    if direct_values is not None:
        return direct_values
    if semantic_values is not None:
        return {feature: semantic_values[feature] for feature in relevant}
    if not values_present:
        _add_issue(issues, "missing_field", f"missing {family} category or one-hot columns")
    return {}


def _pick_value(
    direct: float | None,
    semantic: float | None,
    direct_name: str,
    semantic_name: str,
    issues: list[RowIssue],
) -> float | None:
    if direct is not None and semantic is not None:
        if not math.isclose(direct, semantic, rel_tol=1e-9, abs_tol=1e-8):
            _add_issue(issues, "conflicting_value", f"{direct_name} and {semantic_name} disagree")
    return direct if direct is not None else semantic


def _frequency_value(row: Mapping[str, object], field: str, issues: list[RowIssue]) -> float | None:
    if not _has_value(row, field):
        return None
    value = _text_value(row, field, issues)
    if value not in {"never", "sometimes", "frequently"}:
        _add_issue(issues, "invalid_value", f"{field} must be never, sometimes or frequently")
        return None
    return float(value != "never")


def _oil_smoke_value(row: Mapping[str, object], issues: list[RowIssue]) -> float | None:
    if not _has_value(row, "oil_smoke"):
        return None
    value = _text_value(row, "oil_smoke", issues)
    if value not in {"none", "low", "moderate", "high"}:
        _add_issue(issues, "invalid_value", "oil_smoke has an unsupported category")
        return None
    return float(value in {"moderate", "high"})


def _resolve_feature(
    row: Mapping[str, object], feature: str, subgroup: str, issues: list[RowIssue]
) -> float | None:
    if feature == "age":
        return _finite_number(row, "age", issues, minimum=40, maximum=80)
    if feature == "BMI_new":
        direct = _finite_number(row, "BMI_new", issues, minimum=np.nextafter(0.0, 1.0))
        height = _finite_number(row, "height", issues, minimum=np.nextafter(0.0, 1.0), maximum=250)
        weight = _finite_number(row, "weight", issues, minimum=np.nextafter(0.0, 1.0), maximum=200)
        derived = weight / (height / 100) ** 2 if height is not None and weight is not None else None
        value = _pick_value(direct, derived, "BMI_new", "height/weight", issues)
        if value is None:
            _add_issue(issues, "missing_field", "missing BMI_new or both height and weight")
        return value

    integer_specs = {
        "education.y": ("education", 1, 6),
        "fdr_ca_num": ("family_cancer_count", 0, 20),
    }
    if feature in integer_specs:
        semantic, minimum, maximum = integer_specs[feature]
        direct = _finite_number(row, feature, issues, minimum=minimum, maximum=maximum, integer=True)
        alternate = _finite_number(row, semantic, issues, minimum=minimum, maximum=maximum, integer=True)
        value = _pick_value(direct, alternate, feature, semantic, issues)
        if value is None:
            _add_issue(issues, "missing_field", f"missing {feature} or {semantic}")
        return value

    continuous_specs = {
        "ps_year_new": ("passive_smoking_years", 1.0),
        "smoke_total_year_new": ("years_smoking", 1.0),
        "smoke_quit_year_new": ("years_since_quit", 1.0),
        "smoking_pack_day_new": ("cigarettes_per_day", 1 / 20),
    }
    if feature in continuous_specs:
        semantic, multiplier = continuous_specs[feature]
        direct = _finite_number(row, feature, issues, minimum=0)
        alternate_raw = _finite_number(row, semantic, issues, minimum=0)
        alternate = alternate_raw * multiplier if alternate_raw is not None else None
        value = _pick_value(direct, alternate, feature, semantic, issues)
        if feature == "smoke_quit_year_new" and subgroup == "male_smoker":
            status = _text_value(row, "smoking_status", issues)
            if status == "current":
                if value is not None and value != 0.0:
                    _add_issue(issues, "conflicting_value", "current smokers must have smoke_quit_year_new equal to 0")
                return 0.0
        if value is None:
            _add_issue(issues, "missing_field", f"missing {feature} or {semantic}")
        return value

    binary_specs = {
        "CVD": "chd", "COPD_self": "copd", "CRD_other": "chronic_resp",
        "liver_condition": "liver", "TB_new": "tb",
    }
    if feature in binary_specs:
        semantic = binary_specs[feature]
        direct = _binary_value(row, feature, issues)
        alternate = _binary_value(row, semantic, issues)
        value = _pick_value(direct, alternate, feature, semantic, issues)
        if value is None:
            _add_issue(issues, "missing_field", f"missing {feature} or {semantic}")
        return value
    if feature in {"cancer_history", "diabetes", "hypertension", "exercise"}:
        value = _binary_value(row, feature, issues)
        if value is None:
            _add_issue(issues, "missing_field", f"missing {feature}")
        return value

    frequency_specs = {
        "Freq_pickled": "pickled",
        "Freq_protein": "protein",
        "Freq_vegetable": "vegetable",
    }
    if feature in frequency_specs:
        semantic = frequency_specs[feature]
        direct = _binary_value(row, feature, issues)
        alternate = _frequency_value(row, semantic, issues)
        value = _pick_value(direct, alternate, feature, semantic, issues)
        if value is None:
            _add_issue(issues, "missing_field", f"missing {feature} or {semantic}")
        return value
    if feature == "oil_smoke":
        if isinstance(row.get("oil_smoke"), str) and row["oil_smoke"] in {"none", "low", "moderate", "high"}:
            value = _oil_smoke_value(row, issues)
        else:
            value = _binary_value(row, "oil_smoke", issues)
        if value is None:
            _add_issue(issues, "missing_field", "missing oil_smoke")
        return value
    _add_issue(issues, "unknown_feature", f"no clinical conversion rule exists for {feature}")
    return None


def _resolve_subgroup(row: Mapping[str, object], issues: list[RowIssue]) -> str | None:
    sex = _text_value(row, "sex", issues)
    smoking = _text_value(row, "smoking", issues)
    if sex is None:
        _add_issue(issues, "missing_field", "missing sex")
    elif sex not in {"female", "male"}:
        _add_issue(issues, "invalid_value", "sex must be female or male")
    if smoking is None:
        _add_issue(issues, "missing_field", "missing smoking")
    elif smoking not in {"never", "ever"}:
        _add_issue(issues, "invalid_value", "smoking must be never or ever")
    if sex not in {"female", "male"} or smoking not in {"never", "ever"}:
        return None
    if smoking == "ever":
        status = _text_value(row, "smoking_status", issues)
        if status not in {"former", "current"}:
            _add_issue(issues, "invalid_value", "smoking_status must be former or current for ever smokers")
    return SUBGROUPS[(sex, smoking)]


def _encode_row(row: Mapping[str, object], groups: Mapping[str, Mapping[str, object]]) -> tuple[EncodedRow | None, list[RowIssue]]:
    issues: list[RowIssue] = []
    subgroup = _resolve_subgroup(row, issues)
    if subgroup is None:
        return None, issues
    package = groups[subgroup]
    input_features = list(package["input_features"])
    input_feature_set = set(input_features)
    features: dict[str, float] = {}
    family_members = set()
    for family in ONE_HOT_FAMILIES:
        members = set(ONE_HOT_FAMILIES[family])
        if members & input_feature_set:
            features.update(_resolve_one_hot_family(family, row, input_feature_set, issues))
            family_members.update(members)
    for feature in input_features:
        if feature not in family_members:
            value = _resolve_feature(row, feature, subgroup, issues)
            if value is not None:
                features[feature] = float(value)
    missing_features = [feature for feature in input_features if feature not in features]
    if missing_features and not issues:
        _add_issue(issues, "missing_field", f"missing model features: {missing_features!r}")
    if issues:
        return None, issues
    return EncodedRow(subgroup=subgroup, features={feature: features[feature] for feature in input_features}), issues


def collect_row_issues(row: Mapping[str, object], groups: Mapping[str, Mapping[str, object]]) -> list[RowIssue]:
    """Return every validation issue for one source row without predicting it."""
    _, issues = _encode_row(row, groups)
    return issues


def encode_row(row: Mapping[str, object], groups: Mapping[str, Mapping[str, object]]) -> EncodedRow:
    """Convert one validated clinical row into its selected package's feature order."""
    encoded, issues = _encode_row(row, groups)
    if issues:
        raise BatchInputError("invalid_row", "; ".join(issue.message for issue in issues))
    if encoded is None:  # pragma: no cover - guarded by the issue branch above
        raise BatchInputError("invalid_row", "row cannot be encoded")
    return encoded


def prepare_model_frame(package: Mapping[str, object], features: Mapping[str, float]) -> pd.DataFrame:
    """Scale one complete package input mapping and project onto model features."""
    input_features = list(package["input_features"])
    model_features = list(package["model_features"])
    missing = [feature for feature in input_features if feature not in features]
    extra = [feature for feature in features if feature not in set(input_features)]
    if missing or extra:
        raise ModelContractError(f"encoded features mismatch; missing={missing!r}, extra={extra!r}")
    values = []
    for feature in input_features:
        try:
            value = float(features[feature])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelContractError(f"{feature} must be numeric") from exc
        if not math.isfinite(value):
            raise ModelContractError(f"{feature} must be finite")
        values.append(value)
    raw_frame = pd.DataFrame([values], columns=input_features)
    scaled = np.asarray(package["scaler"].transform(raw_frame), dtype=float)
    if scaled.shape != (1, len(input_features)) or not np.isfinite(scaled).all():
        raise ModelContractError("scaler returned invalid values")
    scaled_frame = pd.DataFrame(scaled, columns=input_features)
    return scaled_frame.loc[:, model_features]


def predict_risk(package: Mapping[str, object], features: Mapping[str, float], time_years: Decimal) -> float:
    """Predict one finite absolute risk at a previously validated horizon."""
    model_frame = prepare_model_frame(package, features)
    target = float(time_years)
    model = package["model"]
    identity = _model_identity(model)
    if identity == COXNET_IDENTITY:
        functions = model.predict_survival_function(model_frame)
        if len(functions) != 1:
            raise ModelContractError("CoxNet returned an unexpected survival-function count")
        survival_values = np.asarray(functions[0](np.asarray([target])), dtype=float)
        if survival_values.size != 1:
            raise ModelContractError("CoxNet returned an invalid survival shape")
        survival = float(survival_values.reshape(-1)[0])
    elif identity == XGBSE_IDENTITY:
        prediction = model.predict(model_frame)
        if not isinstance(prediction, pd.DataFrame) or prediction.shape[0] != 1:
            raise ModelContractError("XGBSE returned an invalid survival table")
        bins = np.asarray([float(column) for column in prediction.columns], dtype=float)
        values = prediction.iloc[0].to_numpy(dtype=float)
        if bins.ndim != 1 or not np.isfinite(bins).all() or np.any(np.diff(bins) <= 0):
            raise ModelContractError("XGBSE returned invalid time bins")
        index = int(np.searchsorted(bins, target, side="right") - 1)
        if index < 0 or index >= len(values):
            raise ModelContractError("requested time is outside XGBSE prediction bins")
        survival = float(values[index])
    else:  # pragma: no cover - guarded by artifact validation
        raise ModelContractError(f"unsupported survival model {'.'.join(identity)}")
    if not math.isfinite(survival):
        raise ModelContractError("model returned a non-finite survival probability")
    risk = float(np.clip(1.0 - survival, 0.0, 1.0))
    if not math.isfinite(risk):
        raise ModelContractError("model returned a non-finite risk")
    return risk


SEMANTIC_COLUMNS = {
    "sex", "smoking", "smoking_status", "age", "education", "occupation", "marital",
    "years_smoking", "cigarettes_per_day", "years_since_quit", "passive_smoking_years",
    "alcohol", "exercise", "oil_smoke", "pickled", "protein", "vegetable", "height",
    "weight", "cancer_history", "family_cancer_count", "copd", "chronic_resp", "diabetes",
    "hypertension", "chd", "liver", "tb", "menopause", "breastfeeding",
}


def _feature_has_source_column(feature: str, columns: set[str]) -> bool:
    if feature in columns:
        return True
    aliases = {
        "BMI_new": {"height", "weight"}, "education.y": {"education"},
        "fdr_ca_num": {"family_cancer_count"}, "ps_year_new": {"passive_smoking_years"},
        "smoke_total_year_new": {"years_smoking"},
        "smoking_pack_day_new": {"cigarettes_per_day"}, "smoke_quit_year_new": {"years_since_quit"},
        "CVD": {"chd"}, "COPD_self": {"copd"}, "CRD_other": {"chronic_resp"},
        "liver_condition": {"liver"}, "TB_new": {"tb"}, "Freq_pickled": {"pickled"},
        "Freq_protein": {"protein"}, "Freq_vegetable": {"vegetable"},
    }
    for family, members in ONE_HOT_FAMILIES.items():
        if feature in members and family in columns:
            return True
    return aliases.get(feature, set()).issubset(columns)


def build_schema_report(columns: list[str], groups: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Describe what source columns can be recognized before any row is predicted."""
    source_columns = set(columns)
    model_columns = set().union(*(set(package["input_features"]) for package in groups.values()))
    one_hot_columns = set().union(*(set(members) for members in ONE_HOT_FAMILIES.values()))
    recognized = SEMANTIC_COLUMNS | model_columns | one_hot_columns
    missing_by_subgroup = {
        subgroup: [
            feature for feature in package["input_features"]
            if not _feature_has_source_column(feature, source_columns)
        ]
        for subgroup, package in groups.items()
    }
    return {
        "recognized_columns": [column for column in columns if column in recognized],
        "semantic_columns": [column for column in columns if column in SEMANTIC_COLUMNS],
        "model_ready_columns": [column for column in columns if column in model_columns],
        "unknown_columns": [column for column in columns if column not in recognized],
        "missing_by_subgroup": missing_by_subgroup,
    }


def _default_output_paths(input_path: Path, risk_column: str) -> tuple[Path, Path, Path]:
    return (
        input_path.with_name(f"{input_path.stem}_with_{risk_column}.csv"),
        input_path.with_name(f"{input_path.stem}_errors.csv"),
        input_path.with_name(f"{input_path.stem}_schema_report.json"),
    )


def validate_output_paths(
    input_path: str | Path,
    output_path: str | Path,
    errors_path: str | Path,
    schema_report_path: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """Prevent output collisions and overwriting existing clinical files."""
    resolved = tuple(Path(path).resolve() for path in (input_path, output_path, errors_path, schema_report_path))
    if len(set(resolved)) != len(resolved):
        raise BatchInputError("invalid_output_path", "input, result, error and schema paths must differ")
    if any(path.exists() for path in resolved[1:]):
        raise BatchInputError("existing_output", "refusing to overwrite an existing output file")
    if any(not path.parent.exists() for path in resolved[1:]):
        raise BatchInputError("invalid_output_path", "each output directory must already exist")
    return resolved


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=destination.parent, delete=False)
    temporary_path = Path(handle.name)
    handle.close()
    try:
        frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(payload: dict[str, object], destination: Path) -> None:
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False)
    temporary_path = Path(handle.name)
    try:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.close()
        temporary_path.replace(destination)
    finally:
        if not handle.closed:
            handle.close()
        if temporary_path.exists():
            temporary_path.unlink()


def _row_subgroup(row: Mapping[str, object]) -> str:
    return SUBGROUPS.get((row.get("sex"), row.get("smoking")), "")


def run_batch(
    input_path: str | Path,
    *,
    time_years: Decimal | str | float = Decimal("5"),
    output_path: str | Path | None = None,
    errors_path: str | Path | None = None,
    schema_report_path: str | Path | None = None,
    encoding: str = "utf-8-sig",
    strict_columns: bool = False,
) -> BatchResult:
    """Calculate one risk horizon for all valid rows in one clinical CSV."""
    source_path = Path(input_path).resolve()
    if not source_path.is_file():
        raise BatchInputError("missing_input", f"input CSV does not exist: {source_path}")
    groups = load_model_groups(Path(__file__).resolve().with_name("models.pkl"))
    requested_time = validate_requested_time(groups, time_years)
    risk_column = risk_column_name(requested_time)
    defaults = _default_output_paths(source_path, risk_column)
    resolved_input, resolved_output, resolved_errors, resolved_report = validate_output_paths(
        source_path,
        output_path or defaults[0],
        errors_path or defaults[1],
        schema_report_path or defaults[2],
    )
    try:
        source_frame = pd.read_csv(resolved_input, dtype=object, encoding=encoding)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise BatchInputError("invalid_input", f"cannot read input CSV: {exc}") from exc
    if risk_column in source_frame.columns:
        raise BatchInputError("existing_risk_column", f"input already contains {risk_column}")
    schema_report = build_schema_report(list(source_frame.columns), groups)
    if strict_columns and schema_report["unknown_columns"]:
        raise BatchInputError("unknown_columns", f"unknown columns: {schema_report['unknown_columns']!r}")

    risks: list[float | None] = []
    error_rows: list[dict[str, object]] = []
    valid_rows = 0
    for position, (_, series) in enumerate(source_frame.iterrows(), start=2):
        row = series.to_dict()
        encoded, issues = _encode_row(row, groups)
        subgroup = encoded.subgroup if encoded is not None else _row_subgroup(row)
        if issues:
            risks.append(None)
            for issue in issues:
                error_rows.append({
                    "source_row_number": position,
                    "error_code": issue.code,
                    "message": issue.message,
                    "subgroup": subgroup,
                    "requested_time_years": str(requested_time),
                })
            continue
        try:
            risk = predict_risk(groups[encoded.subgroup], encoded.features, requested_time)
        except Exception as exc:
            risks.append(None)
            error_rows.append({
                "source_row_number": position,
                "error_code": "prediction_failed",
                "message": f"prediction failed: {type(exc).__name__}: {exc}",
                "subgroup": encoded.subgroup,
                "requested_time_years": str(requested_time),
            })
            continue
        risks.append(risk)
        valid_rows += 1

    result_frame = source_frame.copy()
    result_frame[risk_column] = risks
    error_columns = ["source_row_number", "error_code", "message", "subgroup", "requested_time_years"]
    errors_frame = pd.DataFrame(error_rows, columns=error_columns)
    schema_report.update({
        "requested_time_years": str(requested_time),
        "risk_column": risk_column,
        "valid_rows": valid_rows,
        "invalid_rows": len(source_frame) - valid_rows,
    })
    _atomic_write_csv(result_frame, resolved_output)
    _atomic_write_csv(errors_frame, resolved_errors)
    _atomic_write_json(schema_report, resolved_report)
    return BatchResult(
        output_path=resolved_output,
        errors_path=resolved_errors,
        schema_report_path=resolved_report,
        valid_rows=valid_rows,
        invalid_rows=len(source_frame) - valid_rows,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the public command-line interface without executing a calculation."""
    parser = argparse.ArgumentParser(
        description="批量计算 SSLC 临床表格的肺癌死亡风险。默认输出 5 年风险。",
    )
    parser.add_argument("input_csv", help="输入 CSV 文件路径")
    parser.add_argument("--time", default="5", help="预测时间窗（年）；默认 5")
    parser.add_argument("--output", help="结果 CSV 路径")
    parser.add_argument("--errors", help="逐项错误 CSV 路径")
    parser.add_argument("--schema-report", help="列名一致性 JSON 报告路径")
    parser.add_argument("--encoding", default="utf-8-sig", help="输入 CSV 编码；默认 utf-8-sig")
    parser.add_argument("--strict-columns", action="store_true", help="把未知列作为全表预检错误")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the batch calculator and return a shell-friendly status code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        result = run_batch(
            args.input_csv,
            time_years=args.time,
            output_path=args.output,
            errors_path=args.errors,
            schema_report_path=args.schema_report,
            encoding=args.encoding,
            strict_columns=args.strict_columns,
        )
    except BatchInputError as exc:
        print(f"输入错误 [{exc.code}]：{exc}", file=sys.stderr)
        return 2
    except ModelContractError as exc:
        print(f"模型错误：{exc}", file=sys.stderr)
        return 3
    print(f"完成：有效行 {result.valid_rows}，无效行 {result.invalid_rows}")
    print(f"结果 CSV：{result.output_path}")
    print(f"错误 CSV：{result.errors_path}")
    print(f"列名报告：{result.schema_report_path}")
    return 0


def parse_time_years(value: object) -> Decimal:
    """Return one finite, strictly positive prediction horizon in years."""
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BatchInputError("invalid_time", "--time must be a positive finite number") from exc
    if not result.is_finite() or result <= 0:
        raise BatchInputError("invalid_time", "--time must be a positive finite number")
    return result


def format_time_label(value: object) -> str:
    """Format a numeric horizon for use in portable filenames and columns."""
    normalized = parse_time_years(value).normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", "_")


def risk_column_name(value: object) -> str:
    """Return the output risk column for one horizon."""
    return f"risk_{format_time_label(value)}y"


if __name__ == "__main__":
    raise SystemExit(main())
