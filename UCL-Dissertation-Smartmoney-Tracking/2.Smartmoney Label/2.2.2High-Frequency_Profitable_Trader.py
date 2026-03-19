# ===============================================
# FINAL SCRIPT: Wallet-level HFT Analysis
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load wallet-level CSV
# -------------------------------
df = pd.read_csv("ethereum_addresses.csv")

# Basic sanity check
required_cols = [
    "wallet_address",
    "total_trades",
    "unique_tokens",
    "active_days",
    "total_volume_usd",
    "avg_trade_value"
]
df = df[required_cols].dropna()

print("Wallet-level dataset loaded:")
print(df.head())

# -------------------------------
# 2. Sort wallets by transaction frequency
# -------------------------------
df = df.sort_values("total_trades", ascending=False).reset_index(drop=True)

# Define HFT percentiles (top X%)
hft_percentiles = [0.005, 0.01, 0.02, 0.05]  # 0.5%, 1%, 2%, 5%

# -------------------------------
# 3. Assign HFT labels 
# -------------------------------
for p in hft_percentiles:
    cutoff = int(len(df) * p)
    col = f"is_hft_top{int(p*1000)/10}"
    
    df[col] = False
    df.loc[:cutoff-1, col] = True

# -------------------------------
# 4. Descriptive statistics(median)
# -------------------------------
summary_list = []

for p in hft_percentiles:
    col = f"is_hft_top{int(p*1000)/10}"
    
    stats = df.groupby(col)[
        ["total_trades",
         "total_volume_usd",
         "unique_tokens",
         "active_days",
         "avg_trade_value"]
    ].median()
    
    stats["top_percentile"] = p
    summary_list.append(stats)

summary_df = pd.concat(summary_list).reset_index()

print("\nMedian descriptive stats by HFT percentile:")
print(summary_df)

# -------------------------------
# 5. Percentile cutoff table
# -------------------------------
cutoff_table = []

for p in hft_percentiles:
    cutoff_idx = int(len(df) * p)
    cutoff_table.append({
        "top_percentile": p,
        "min_total_trades": df.loc[cutoff_idx-1, "total_trades"]
    })

cutoff_df = pd.DataFrame(cutoff_table)
print("\nTransaction count cutoffs by percentile:")
print(cutoff_df)

# -------------------------------
# 6. Distribution plots (log scale)
# -------------------------------
plt.figure(figsize=(10, 5))
plt.hist(np.log1p(df["total_trades"]), bins=50)
for p in hft_percentiles:
    cutoff_idx = int(len(df) * p)
    cutoff_value = np.log1p(df.loc[cutoff_idx-1, "total_trades"])
    plt.axvline(cutoff_value, linestyle="dashed", alpha=0.6)
plt.xlabel("log(1 + total_trades)")
plt.ylabel("Number of wallets")
plt.title("Distribution of Wallet Transaction Counts")
plt.show()

# -------------------------------
# 7. HFT share of total volume
# -------------------------------
volume_stats = []

total_volume = df["total_volume_usd"].sum()

for p in hft_percentiles:
    col = f"is_hft_top{int(p*1000)/10}"
    
    hft_volume = df.loc[df[col], "total_volume_usd"].sum()
    hft_wallets_pct = df[col].mean() * 100
    hft_volume_pct = hft_volume / total_volume * 100
    
    volume_stats.append({
        "top_percentile": p,
        "wallet_share_pct": hft_wallets_pct,
        "volume_share_pct": hft_volume_pct
    })

volume_df = pd.DataFrame(volume_stats)
print("\nHFT wallet and volume shares:")
print(volume_df)

# -------------------------------
# 8. Save outputs
# -------------------------------
df.to_csv("wallet_level_HFT_FINAL.csv", index=False)
summary_df.to_csv("HFT_summary_by_percentile_FINAL.csv", index=False)
cutoff_df.to_csv("HFT_trade_cutoffs_FINAL.csv", index=False)

print("\nSaved final output files.")
