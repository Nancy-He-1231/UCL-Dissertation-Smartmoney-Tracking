"""
=============================================================================
SCREENING CRITERIA SENSITIVITY ANALYSIS — BOT FILTER
=============================================================================
Purpose : Tests whether XGBoost results are sensitive to the bot filter
          threshold (100 tx/day in baseline).

Input   : filtered_trades.csv   (output of your cleaning pipeline)
          advanced_features.csv (output of feature engineering)
          smart_money_labels.csv(your ground truth labels)

Output  : sensitivity_bot_filter.csv  — results across bot thresholds
          sensitivity_summary.txt     — human-readable summary + thesis text
          sensitivity_plots.png       — visualisation
=============================================================================
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import hashlib
import warnings
import logging
from typing import Dict, Optional

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

TRADES_FILE   = 'filtered_trades.csv'
FEATURES_FILE = 'advanced_features.csv'
LABELS_FILE   = 'smart_money_labels.csv'

WALLET_COL  = 'wallet_address'
TIME_COL    = 'block_time'
AMOUNT_COL  = 'amount_usd'

BASELINE_BOT_THRESHOLD = 100
BASELINE_VALUE_CAP     = 5_000_000   # kept only to hold other screening constant
BASELINE_VALUE_FLOOR   = 1
MIN_TRADES             = 10
MIN_ACTIVE_DAYS        = 5

BOT_THRESHOLDS = [50, 75, 100, 150, 200, None]   # None = no filter

XGB_PARAMS = {
    'objective'        : 'binary:logistic',
    'eval_metric'      : 'auc',
    'max_depth'        : 6,
    'learning_rate'    : 0.1,
    'n_estimators'     : 300,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
    'random_state'     : 42,
    'n_jobs'           : -1,
}
CV_FOLDS     = 5
RANDOM_STATE = 42


# =============================================================================
# HELPERS
# =============================================================================

def wallet_to_group(wallet_addr: str) -> int:
    return int(hashlib.md5(wallet_addr.encode()).hexdigest()[:4], 16) % 20


def apply_screening(trades_df: pd.DataFrame,
                    bot_threshold: Optional[int],
                    value_cap: Optional[float] = BASELINE_VALUE_CAP,
                    value_floor: float = BASELINE_VALUE_FLOOR,
                    min_trades: int = MIN_TRADES,
                    min_active_days: int = MIN_ACTIVE_DAYS):
    df = trades_df.copy()
    df[TIME_COL]   = pd.to_datetime(df[TIME_COL], utc=True, errors='coerce')
    df[AMOUNT_COL] = pd.to_numeric(df[AMOUNT_COL], errors='coerce')

    df = df[df[AMOUNT_COL] > value_floor]

    if value_cap is not None:
        df = df[df[AMOUNT_COL] < value_cap]

    if bot_threshold is not None:
        df['_date'] = df[TIME_COL].dt.date
        daily_counts = (df.groupby([WALLET_COL, '_date'])
                          .size()
                          .reset_index(name='_daily_tx'))
        bot_wallets = daily_counts[
            daily_counts['_daily_tx'] > bot_threshold
        ][WALLET_COL].unique()
        df = df[~df[WALLET_COL].isin(bot_wallets)]
        df = df.drop(columns=['_date'])

    wallet_stats = df.groupby(WALLET_COL).agg(
        total_trades = (AMOUNT_COL, 'count'),
        active_days  = (TIME_COL, lambda x: x.dt.date.nunique())
    ).reset_index()

    qualifying = wallet_stats[
        (wallet_stats['total_trades'] >= min_trades) &
        (wallet_stats['active_days']  >= min_active_days)
    ][WALLET_COL]

    return qualifying


def train_and_evaluate(features_df: pd.DataFrame,
                       labels: np.ndarray,
                       random_state: int = RANDOM_STATE) -> Dict:
    numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return {}

    X      = features_df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y      = labels
    groups = np.array([wallet_to_group(w) for w in features_df[WALLET_COL]])

    unique_groups = np.unique(groups)
    np.random.seed(random_state)
    test_groups    = np.random.choice(unique_groups,
                                      size=max(1, int(len(unique_groups) * 0.2)),
                                      replace=False)
    test_mask      = np.isin(groups, test_groups)
    train_val_mask = ~test_mask

    X_tv, X_te = X[train_val_mask], X[test_mask]
    y_tv, y_te = y[train_val_mask], y[test_mask]
    g_tv       = groups[train_val_mask]

    scaler = StandardScaler()
    X_tv   = scaler.fit_transform(X_tv)
    X_te   = scaler.transform(X_te)

    pos   = y_tv.sum()
    neg   = len(y_tv) - pos
    scale = neg / pos if pos > 0 else 1.0

    params = XGB_PARAMS.copy()
    params['scale_pos_weight'] = scale

    gkf     = GroupKFold(n_splits=min(CV_FOLDS, len(np.unique(g_tv))))
    cv_aucs = []
    for tr_idx, val_idx in gkf.split(X_tv, y_tv, g_tv):
        m = xgb.XGBClassifier(**params)
        m.fit(X_tv[tr_idx], y_tv[tr_idx],
              eval_set=[(X_tv[val_idx], y_tv[val_idx])],
              verbose=False)
        cv_aucs.append(roc_auc_score(y_tv[val_idx],
                                     m.predict_proba(X_tv[val_idx])[:, 1]))

    model = xgb.XGBClassifier(**params)
    model.fit(X_tv, y_tv)
    proba = model.predict_proba(X_te)[:, 1]
    pred  = model.predict(X_te)

    return {
        'cv_auc_mean' : np.mean(cv_aucs),
        'cv_auc_std'  : np.std(cv_aucs),
        'test_auc'    : roc_auc_score(y_te, proba),
        'precision'   : precision_score(y_te, pred, zero_division=0),
        'recall'      : recall_score(y_te, pred, zero_division=0),
        'f1'          : f1_score(y_te, pred, zero_division=0),
        'n_wallets'   : len(features_df),
        'n_test'      : int(test_mask.sum()),
    }


def load_labels(features_df: pd.DataFrame) -> np.ndarray:
    for fname in [LABELS_FILE, 'wallet_labels.csv', 'ground_truth_labels.csv']:
        try:
            ldf    = pd.read_csv(fname)
            merged = features_df.merge(ldf, on=WALLET_COL, how='left')
            for col in ['is_smart_money', 'smart_money', 'label', 'target']:
                if col in merged.columns:
                    logger.info(f"  Labels loaded from '{fname}' column '{col}'")
                    return merged[col].fillna(0).astype(int).values
        except FileNotFoundError:
            continue
    logger.warning("  Label file not found — using random demo labels (5% positive).")
    np.random.seed(RANDOM_STATE)
    return np.random.choice([0, 1], size=len(features_df), p=[0.95, 0.05])


# =============================================================================
# SENSITIVITY TEST — BOT FILTER THRESHOLD
# =============================================================================

def sensitivity_bot_filter(trades_df: pd.DataFrame,
                            features_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n[TEST] Bot filter threshold sensitivity...")
    rows = []

    for threshold in BOT_THRESHOLDS:
        label = str(threshold) if threshold is not None else 'No filter'
        logger.info(f"  Threshold = {label} tx/day")

        qualifying   = apply_screening(trades_df, bot_threshold=threshold)
        sub_features = features_df[features_df[WALLET_COL].isin(qualifying)].copy()
        n_wallets    = len(sub_features)

        if n_wallets < 30:
            logger.warning(f"    Only {n_wallets} wallets — skipping.")
            continue

        labels  = load_labels(sub_features)
        metrics = train_and_evaluate(sub_features, labels)

        rows.append({
            'Bot_Threshold' : label,
            'Is_Baseline'   : 'YES' if threshold == BASELINE_BOT_THRESHOLD else '',
            'N_Wallets'     : n_wallets,
            'CV_AUC'        : round(metrics.get('cv_auc_mean', np.nan), 4),
            'CV_AUC_SD'     : round(metrics.get('cv_auc_std',  np.nan), 4),
            'Test_AUC'      : round(metrics.get('test_auc',    np.nan), 4),
            'Precision'     : round(metrics.get('precision',   np.nan), 4),
            'Recall'        : round(metrics.get('recall',      np.nan), 4),
            'F1'            : round(metrics.get('f1',          np.nan), 4),
        })
        logger.info(f"    N={n_wallets:,}  Test AUC={metrics.get('test_auc', 0):.4f}")

    return pd.DataFrame(rows)


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_sensitivity(bot_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))

    palette  = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#44BBA4']
    base_col = '#E63946'

    x      = np.arange(len(bot_df))
    colors = [base_col if v == 'YES' else palette[i % len(palette)]
              for i, v in enumerate(bot_df['Is_Baseline'])]

    bars = ax.bar(x, bot_df['Test_AUC'], color=colors, alpha=0.85, width=0.55,
                  edgecolor='white', linewidth=0.8)
    ax.errorbar(x, bot_df['Test_AUC'], yerr=bot_df['CV_AUC_SD'],
                fmt='none', color='#333333', capsize=4, linewidth=1.2)

    for bar, val in zip(bars, bot_df['Test_AUC']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(bot_df['Bot_Threshold'], rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Test AUC', fontsize=12)
    ax.set_xlabel('Bot Filter Threshold (Max Daily Transactions)', fontsize=12)
    ax.set_title('Sensitivity to Bot Filter Threshold', fontsize=13,
                 fontweight='bold', pad=12)
    ax.set_ylim(max(0, bot_df['Test_AUC'].min() - 0.05),
                min(1, bot_df['Test_AUC'].max() + 0.07))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=base_col, label='Baseline (100 tx/day)')],
              fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig('sensitivity_plots.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    logger.info("  Saved: sensitivity_plots.png")


# =============================================================================
# SUMMARY
# =============================================================================

def write_summary(bot_df: pd.DataFrame):
    auc_min  = bot_df['Test_AUC'].min()
    auc_max  = bot_df['Test_AUC'].max()
    auc_spread = auc_max - auc_min
    robust   = auc_spread <= 0.03

    # baseline and no-filter rows
    base_row     = bot_df[bot_df['Is_Baseline'] == 'YES']
    nofilter_row = bot_df[bot_df['Bot_Threshold'] == 'No filter']
    base_auc     = base_row['Test_AUC'].values[0]     if not base_row.empty     else float('nan')
    nofilter_auc = nofilter_row['Test_AUC'].values[0] if not nofilter_row.empty else float('nan')
    delta_auc    = abs(base_auc - nofilter_auc)

    lines = []
    lines.append("SCREENING CRITERIA SENSITIVITY ANALYSIS — SUMMARY")
    lines.append("=" * 65)
    lines.append("\nTEST: Bot Filter Threshold (baseline = 100 tx/day)")
    lines.append("-" * 65)
    lines.append(f"{'Threshold':<18} {'N Wallets':>10} {'Test AUC':>10} "
                 f"{'Precision':>10} {'Recall':>10} {'F1':>8}")
    lines.append("-" * 65)

    for _, row in bot_df.iterrows():
        marker = " <-- baseline" if row['Is_Baseline'] == 'YES' else ""
        lines.append(
            f"{row['Bot_Threshold']:<18} {row['N_Wallets']:>10,} "
            f"{row['Test_AUC']:>10.4f} {row['Precision']:>10.4f} "
            f"{row['Recall']:>10.4f} {row['F1']:>8.4f}{marker}"
        )

    lines.append(f"\nAUC range : [{auc_min:.4f}, {auc_max:.4f}]")
    lines.append(f"AUC spread: {auc_spread:.4f}")
    lines.append(f"Verdict   : {'ROBUST — spread <= 0.03' if robust else 'SENSITIVE — spread > 0.03, discuss in limitations'}")

    lines.append("\n\nSUGGESTED THESIS TEXT")
    lines.append("-" * 65)
    lines.append(f"""
Removing it entirely yields a Test AUC of {nofilter_auc:.4f}, compared to {base_auc:.4f} under the baseline threshold
of 100 daily transactions — a difference of {delta_auc:.4f}.

Across six alternative specifications — 50, 75, 100 (baseline), 150, 200 daily
transactions, and no filter — Test AUC values range from {auc_min:.4f} to
{auc_max:.4f}, a spread of {auc_spread:.4f}. 
""")

    text = "\n".join(lines)
    with open('sensitivity_summary.txt', 'w') as f:
        f.write(text)
    print("\n" + text)
    logger.info("  Saved: sensitivity_summary.txt")


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 65)
    logger.info("  BOT FILTER THRESHOLD SENSITIVITY ANALYSIS")
    logger.info("=" * 65)

    logger.info("\n[1/4] Loading trades data...")
    try:
        trades_df = pd.read_csv(TRADES_FILE, low_memory=False)
        logger.info(f"  Loaded {len(trades_df):,} trade records.")
    except FileNotFoundError:
        logger.error(f"  '{TRADES_FILE}' not found.")
        return

    logger.info("\n[2/4] Loading wallet features...")
    try:
        features_df = pd.read_csv(FEATURES_FILE)
        logger.info(f"  Loaded features for {len(features_df):,} wallets.")
    except FileNotFoundError:
        logger.error(f"  '{FEATURES_FILE}' not found.")
        return

    for col in [WALLET_COL, TIME_COL, AMOUNT_COL]:
        if col not in trades_df.columns:
            logger.error(f"  Column '{col}' missing from trades file.")
            return

    logger.info("\n[3/4] Running bot filter sensitivity test...")
    bot_df = sensitivity_bot_filter(trades_df, features_df)
    bot_df.to_csv('sensitivity_bot_filter.csv', index=False)
    logger.info("  Saved: sensitivity_bot_filter.csv")

    logger.info("\n[4/4] Generating plot and summary...")
    plot_sensitivity(bot_df)
    write_summary(bot_df)

    logger.info("\n" + "=" * 65)
    logger.info("  Done! Outputs:")
    logger.info("    sensitivity_bot_filter.csv")
    logger.info("    sensitivity_plots.png")
    logger.info("    sensitivity_summary.txt")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
