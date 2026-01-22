from __future__ import annotations
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_DB = Path(
    "/home/haris/Documents/ProjectBzarre/preprocessing_pipeline/check_multicolinearity/all_preprocessed_sources.db"
)
DEFAULT_TABLE = "merged_train"
DEFAULT_OUTPUT_CSV = Path(
    "/home/haris/Documents/ProjectBzarre/regression_pipeline/log_candidates.csv"
)

# Heuristic filters for non-continuous features (explicit, conservative).
NAME_EXCLUDE_TOKENS = {
    "time",
    "timestamp",
    "date",
    "id",
    "flag",
    "bucket",
    "regime",
    "source",
    "cat",
    "category",
}


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _is_integer_coded(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return True
    if pd.api.types.is_integer_dtype(series):
        return True
    vals = series.dropna().to_numpy()
    if vals.size == 0:
        return True
    return np.allclose(vals, np.round(vals))


def _is_flag_like(series: pd.Series) -> bool:
    vals = series.dropna().unique()
    if vals.size == 0:
        return True
    if vals.size <= 2:
        return True
    if np.all(np.isin(vals, [0, 1])):
        return True
    return False


def _exclude_reason(name: str, series: pd.Series) -> str | None:
    lowered = name.lower()
    if any(token in lowered for token in NAME_EXCLUDE_TOKENS):
        return "name_match"
    if _is_flag_like(series):
        return "flag_like"
    if _is_integer_coded(series):
        unique_count = series.dropna().nunique()
        if unique_count <= 20:
            return "integer_coded_low_cardinality"
    return None


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 3:
        return np.nan
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])


def _sample_series(series: pd.Series, max_n: int = 5000) -> np.ndarray:
    vals = series.dropna().to_numpy()
    if vals.size <= max_n:
        return vals
    idx = np.linspace(0, vals.size - 1, num=max_n, dtype=int)
    return vals[idx]


def _qq_r_value(series: pd.Series) -> float:
    sample = _sample_series(series)
    if sample.size < 3:
        return np.nan
    _, (_, _, r) = stats.probplot(sample, dist="norm")
    return float(r)


def _summarize_feature(name: str, series: pd.Series, rolling_window: int) -> dict:
    clean = series.dropna().astype(float)
    if clean.empty:
        raise RuntimeError(f"No data available for feature '{name}'.")

    non_positive_frac = float((clean <= 0).mean())
    min_val = float(clean.min())
    max_val = float(clean.max())

    orders_of_mag = np.nan
    if min_val > 0 and max_val > 0:
        orders_of_mag = float(np.log10(max_val / min_val)) if max_val > min_val else 0.0

    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=0))

    # Skewness and kurtosis indicate tail behavior and asymmetry.
    skew_val = float(stats.skew(clean, bias=False))
    kurt_val = float(stats.kurtosis(clean, fisher=True, bias=False))

    # Rolling variance/absolute deviation capture scale-dependent noise.
    roll_var = clean.rolling(rolling_window, min_periods=max(3, rolling_window // 2)).var()
    roll_abs = clean.rolling(rolling_window, min_periods=max(3, rolling_window // 2)).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    corr_var = _safe_corr(clean, roll_var)
    corr_abs = _safe_corr(clean, roll_abs)

    has_negatives = (clean < 0).any()
    log_transform = None
    if has_negatives:
        log_transform = None
    else:
        if (clean == 0).any():
            log_transform = np.log1p(clean)
        else:
            log_transform = np.log(clean)

    log_skew = np.nan
    log_kurt = np.nan
    qq_r = _qq_r_value(clean)
    qq_r_log = np.nan
    if log_transform is not None:
        log_skew = float(stats.skew(log_transform, bias=False))
        log_kurt = float(stats.kurtosis(log_transform, fisher=True, bias=False))
        qq_r_log = _qq_r_value(log_transform)

    skew_reduction = np.nan if log_transform is None else abs(skew_val) - abs(log_skew)
    kurt_reduction = np.nan if log_transform is None else abs(kurt_val) - abs(log_kurt)
    qq_r_improvement = np.nan if log_transform is None else qq_r_log - qq_r

    # Conservative heuristic: only suggest log if strictly non-negative,
    # high skew and wide scale, and diagnostics improve meaningfully.
    log_candidate = False
    if log_transform is not None:
        if (
            non_positive_frac <= 0.01
            and abs(skew_val) >= 1.0
            and orders_of_mag is not np.nan
            and orders_of_mag >= 2.0
            and skew_reduction >= 0.3
            and kurt_reduction >= 0.3
            and qq_r_improvement >= 0.02
        ):
            log_candidate = True

    return {
        "feature": name,
        "non_positive_frac": non_positive_frac,
        "min": min_val,
        "max": max_val,
        "orders_of_magnitude": orders_of_mag,
        "mean": mean_val,
        "std": std_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "corr_value_roll_var": corr_var,
        "corr_value_roll_abs_dev": corr_abs,
        "log_skewness": log_skew,
        "log_kurtosis": log_kurt,
        "qq_r": qq_r,
        "qq_r_log": qq_r_log,
        "skewness_reduction": skew_reduction,
        "kurtosis_reduction": kurt_reduction,
        "qq_r_improvement": qq_r_improvement,
        "log_candidate": log_candidate,
        "log_possible": log_transform is not None,
        "has_negatives": has_negatives,
    }


def main() -> None:
    db_path = DEFAULT_DB
    table = DEFAULT_TABLE
    rolling_window = 24
    output_csv = DEFAULT_OUTPUT_CSV

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    df = _load_table(db_path, table)
    if df.empty:
        raise RuntimeError(f"Table '{table}' is empty.")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise RuntimeError("No numeric columns found to analyze.")

    rows = []
    excluded = []
    for col in numeric_cols:
        reason = _exclude_reason(col, df[col])
        if reason is not None:
            excluded.append((col, reason))
            continue
        rows.append(_summarize_feature(col, df[col], rolling_window))

    report = pd.DataFrame(rows).sort_values(
        ["log_candidate", "skewness_reduction", "kurtosis_reduction"],
        ascending=[False, False, False],
    )

    print(f"[INFO] Analyzed table: {table}")
    print(f"[INFO] Numeric columns: {len(numeric_cols)}")
    print(f"[INFO] Excluded columns: {len(excluded)}")
    if excluded:
        print("[INFO] Exclusions (feature -> reason):")
        for name, reason in excluded:
            print(f"  - {name} -> {reason}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n[REPORT]")
    print(report.to_string(index=False))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
    print(f"\n[OK] CSV report written to {output_csv}")


if __name__ == "__main__":
    main()
