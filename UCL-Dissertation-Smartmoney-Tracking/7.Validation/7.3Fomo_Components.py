import pandas as pd
import numpy as np

print("=== Extracting FOMO Sub-components ===")

# =========================
# 1. LOAD DATA
# =========================
trades = pd.read_csv("filtered_trades.csv")
prices = pd.read_csv("price_usd.csv")

trades['block_time'] = pd.to_datetime(trades['block_time'])
prices['block_time'] = pd.to_datetime(prices['block_time'])

trades['trade_action'] = trades['trade_direction'].map({
    'buy_stable': 1,
    'sell_stable': -1,
    'buy_eth': 1,
    'sell_eth': -1
})

# =========================
# 2. FOMO SUB-COMPONENTS 
# =========================

def momentum_resistance(wallet_df):
    if len(wallet_df) < 3:
        return np.nan

    actions = wallet_df['trade_action'].dropna()
    if len(actions) < 3:
        return np.nan

    shifted = actions.shift(1).dropna()
    aligned = actions.loc[shifted.index]

    if len(aligned) < 2:
        return np.nan

    std1 = aligned.std()
    std2 = shifted.std()

    if std1 == 0 or std2 == 0:
        return np.nan

    corr = aligned.corr(shifted)
    if np.isnan(corr):
        return np.nan

    return 1 - abs(corr)


def timing_discipline(wallet_df):
    wallet_df = wallet_df.copy()

    wallet_df['hour'] = wallet_df['block_time'].dt.hour
    counts = wallet_df['hour'].value_counts()
    total = len(wallet_df)

    if total <= 1:
        return np.nan

    hhi = sum((c / total) ** 2 for c in counts)
    return 1 - hhi


def frequency_control(wallet_df):
    if len(wallet_df) < 2:
        return np.nan

    days = (wallet_df['block_time'].max() - wallet_df['block_time'].min()).days + 1
    freq = len(wallet_df) / max(days, 1)

    return 1 / (1 + np.log1p(freq))


def size_rationality(wallet_df):
    sizes = wallet_df['amount_usd'].replace(0, np.nan).dropna()

    if len(sizes) < 2:
        return np.nan

    log_sizes = np.log1p(sizes)

    mean = log_sizes.mean()
    std = log_sizes.std()

    if mean == 0:
        return np.nan

    cv = std / (mean + 1e-8)

    return 1 / (1 + cv)


def cost_sensitivity(wallet_df):
    if 'tx_cost_eth' not in wallet_df.columns:
        return np.nan

    ratio = (wallet_df['tx_cost_eth'] / wallet_df['amount_usd']) \
        .replace([np.inf, -np.inf], np.nan) \
        .dropna()

    if len(ratio) == 0:
        return np.nan

    # raw signal
    return ratio.median()


# =========================
# 3. LOOP WALLETS
# =========================

results = []

wallets = trades['wallet_address'].unique()

for i, w in enumerate(wallets):
    if i % 50 == 0:
        print(f"Processing {i}/{len(wallets)}")

    df = trades[trades['wallet_address'] == w].copy()

    results.append({
        'wallet_address': w,
        'momentum_resistance': momentum_resistance(df),
        'timing_discipline': timing_discipline(df),
        'frequency_control': frequency_control(df),
        'size_rationality': size_rationality(df),
        'cost_sensitivity': cost_sensitivity(df)
    })

# =========================
# 4. CREATE DATAFRAME
# =========================

fomo_df = pd.DataFrame(results)

# =========================
# 5. CROSS-SECTIONAL NORMALIZATION
# =========================

features = [
    'momentum_resistance',
    'timing_discipline',
    'frequency_control',
    'size_rationality',
    'cost_sensitivity'
]

# rank transform (0~1)
for col in features:
    fomo_df[col] = fomo_df[col].rank(pct=True)

# fill missing after rank
fomo_df = fomo_df.fillna(0)

# =========================
# 6. SAVE
# =========================

fomo_df.to_csv("fomo_components.csv", index=False)

print("\n Saved: fomo_components.csv")
print(fomo_df.describe())
