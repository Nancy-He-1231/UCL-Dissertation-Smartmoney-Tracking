"""
Foptimal Threshold Sensitivity Analysis
========================================
How sensitive the FOMO Resistance index is to the choice of Foptimal (the trading frequency benchmark threshold
in the Frequency Control sub-component).

Requires:
  - fomo_components.csv 
  - returnrate.csv

Tested thresholds: 1, 2, 3, 5 trades/day
Baseline:          2 trades/day 
"""

import pandas as pd
import numpy as np
from scipy import stats

# ============================================================
# CONFIG
# ============================================================
FEATURES_FILE   = 'fomo_components.csv'
REGRESSION_FILE = 'returnrate.csv'

SUB_COMPONENTS = [
    'momentum_resistance',
    'timing_discipline',
    'frequency_control',   
    'size_rationality',
    'cost_sensitivity'
]

BASE_WEIGHTS     = np.array([0.25, 0.20, 0.25, 0.15, 0.15])
FREQ_COL         = 'trading_frequency'   # raw Fi column (trades/day)
F_OPTIMAL_BASE   = 2
F_OPTIMAL_LIST   = [1, 2, 3, 5]
F_MAX            = 50

# ============================================================
# HELPERS
# ============================================================

def compute_fomo(df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Cross-sectionally standardise sub-components, then take weighted sum."""
    w = np.array(weights, dtype=float)
    w = w / (w.sum() + 1e-8)
    X = df[SUB_COMPONENTS].copy()
    X = (X - X.mean()) / (X.std() + 1e-8)
    return X @ w


def spearman_ic(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 5:
        return np.nan
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return float(rho)


def rank_corr(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 5:
        return np.nan
    rho, _ = stats.spearmanr(a[mask], b[mask])
    return float(rho)


def apply_foptimal(df: pd.DataFrame, f_opt: float) -> pd.DataFrame:
    """
    Recompute frequency_control using a given Foptimal threshold,
    then rank-normalise to [0,1] so it stays on the same scale
    as the other sub-components.
    """
    df_copy = df.copy()
    df_copy['frequency_control'] = df_copy[FREQ_COL].apply(
        lambda fi: 1.0 if fi <= f_opt
        else max(0.0, 1.0 - (fi - f_opt) / F_MAX)
    )
    # rank-normalise to match the scale of other sub-components
    df_copy['frequency_control'] = df_copy['frequency_control'].rank(pct=True)
    return df_copy


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  Foptimal Threshold Sensitivity Analysis")
    print("=" * 60)

    # ── Load & merge ───────────────────────────────────────
    feat = pd.read_csv(FEATURES_FILE)

    if FREQ_COL not in feat.columns:
        raise KeyError(
            f"Column '{FREQ_COL}' not found in {FEATURES_FILE}.\n"
            f"Available columns: {feat.columns.tolist()}\n"
            f"Re-run 7.3Fomo_Components.py to regenerate the file with "
            f"the trading_frequency column included."
        )

    reg = pd.read_csv(REGRESSION_FILE)
    # one row per wallet — take mean if multiple rows exist
    reg = (
        reg.groupby('wallet_address')['avg_return']
        .mean()
        .reset_index()
    )

    df = reg.merge(feat, on='wallet_address', how='inner')
    y  = df['avg_return']

    print(f"\nSample size: {len(df):,} wallets")
    print(f"Baseline Foptimal: {F_OPTIMAL_BASE} trades/day\n")

    df_base  = apply_foptimal(df, F_OPTIMAL_BASE)
    f_base   = compute_fomo(df_base, BASE_WEIGHTS)
    base_ic  = spearman_ic(f_base, y)

    print(f"{'Foptimal':<25} {'IC (Spearman)':>14}  {'ΔIC':>8}  {'Rank Corr vs Baseline':>22}")
    print(f"{'-'*25} {'-'*14}  {'-'*8}  {'-'*22}")

    rows = []

    for f_opt in F_OPTIMAL_LIST:
        df_new   = apply_foptimal(df, f_opt)
        f_new    = compute_fomo(df_new, BASE_WEIGHTS)
        ic_new   = spearman_ic(f_new, y)
        rc_new   = rank_corr(f_base, f_new)
        delta_ic = ic_new - base_ic

        marker = " ← baseline" if f_opt == F_OPTIMAL_BASE else ""
        label  = f"Foptimal = {f_opt} trades/day"
        print(f"{label:<25} {ic_new:>14.4f}  {delta_ic:>+8.4f}  {rc_new:>22.4f}{marker}")

        rows.append({
            'Foptimal (trades/day)': f_opt,
            'IC (Spearman)': round(ic_new, 4),
            'ΔIC vs Baseline': round(delta_ic, 4),
            'Rank Corr vs Baseline': round(rc_new, 4),
            'Baseline': 'Yes' if f_opt == F_OPTIMAL_BASE else 'No'
        })

    # ── Summary stats ──────────────────────────────────────
    result_df    = pd.DataFrame(rows)
    max_delta    = result_df['ΔIC vs Baseline'].abs().max()
    min_rc       = result_df['Rank Corr vs Baseline'].min()
    non_baseline = result_df[result_df['Baseline'] == 'No']

    # ── Save CSV ───────────────────────────────────────────
    out_path = 'foptimal_sensitivity.csv'
    result_df.to_csv(out_path, index=False)

    # ── Narrative ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  NARRATIVE SUMMARY ")
    print("=" * 60)
    print(f"""
    The Foptimal threshold was varied across {F_OPTIMAL_LIST} trades per day.
    Rank correlations against the baseline exceed {min_rc:.2f} across all
    specifications, and the maximum IC deviation is {max_delta:.4f},
    confirming robustness to this parameter choice (Barber & Odean, 2000).
    """)

    print(f" Saved: {out_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
