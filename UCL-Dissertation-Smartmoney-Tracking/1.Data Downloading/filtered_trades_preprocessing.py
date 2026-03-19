import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt

# 1. Load Data
# Assuming the filename remains the same
eth_df = pd.read_csv('filtered_trades (2).csv')

print(f"Ethereum Data: {len(eth_df)} records")
print("\nEthereum Columns:", eth_df.columns.tolist())

# 2. Standardization Logic
def standardize_chain_data(df):
    """Standardize Ethereum chain data format"""
    df['chain'] = 'ethereum'
    
    # Unify time format
    df['block_time'] = pd.to_datetime(df['block_time'], utc=True)
    
    # Standardize ETH/WETH naming
    df['token_bought_symbol'] = df['token_bought_symbol'].replace({'ETH': 'WETH'})
    df['token_sold_symbol'] = df['token_sold_symbol'].replace({'ETH': 'WETH'})
    
    # Replace Zero Address with WETH address (Ethereum)
    eth_zero_address = '0x0000000000000000000000000000000000000000'
    weth_address = '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'
    
    df.loc[df['token_bought_address'] == eth_zero_address, 'token_bought_address'] = weth_address
    df.loc[df['token_sold_address'] == eth_zero_address, 'token_sold_address'] = weth_address
    
    # Ensure numeric types are correct
    numeric_cols = ['token_bought_amount', 'token_sold_amount', 'amount_usd', 'gas_used', 'gas_price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate Gas Cost (ETH)
    if 'gas_used' in df.columns and 'gas_price' in df.columns:
        df['tx_cost_eth'] = (df['gas_used'] * df['gas_price']) / 1e18
    
    return df

eth_df = standardize_chain_data(eth_df)

# 3. Deduplication Logic
def remove_duplicates(df):
    """Remove duplicate transactions"""
    # Remove exact event duplicates
    df_dedup = df.drop_duplicates(subset=['tx_hash', 'evt_index'], keep='first')
    
    # Check for potential duplicates (same wallet, time, and amounts)
    duplicate_check = df_dedup.duplicated(
        subset=['wallet_address', 'block_time', 'token_bought_address', 'token_sold_address', 'amount_usd'],
        keep=False
    )
    
    if duplicate_check.sum() > 0:
        print(f"Detected {duplicate_check.sum()} potential duplicate transactions")
        df_dedup = df_dedup.sort_values('tx_cost_eth').drop_duplicates(
            subset=['wallet_address', 'block_time', 'token_bought_address', 'token_sold_address'],
            keep='first'
        )
    
    print(f"Remaining records after deduplication: {len(df_dedup)} (Original: {len(df)})")
    return df_dedup

eth_df = remove_duplicates(eth_df)

# 4. Data Cleaning
def clean_data(df):
    """Data Cleaning"""
    initial_len = len(df)
    
    # Filter out anomalous amounts
    df = df[(df['amount_usd'] > 0.01) & (df['amount_usd'] < 10_000_000)]
    df = df[(df['token_bought_amount'] > 0) & (df['token_sold_amount'] > 0)]
    
    print("\nEthereum Missing Values Statistics:")
    print(df.isnull().sum())
    
    df['token_bought_symbol'] = df['token_bought_symbol'].fillna('UNKNOWN')
    df['token_sold_symbol'] = df['token_sold_symbol'].fillna('UNKNOWN')
    
    # Filter extreme Gas costs
    if 'tx_cost_eth' in df.columns:
        df = df[df['tx_cost_eth'] <= 2]
    
    print(f"\nRemaining records after cleaning: {len(df)} (Removed {initial_len - len(df)})")
    return df

eth_df = clean_data(eth_df)

# 5. Token Mapping & Feature Engineering
TOKEN_MAPPING = {
    'USDC': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
    'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7',
    'WETH': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'
}

def add_derived_features(df):
    """Add derived features for analysis"""
    # Unify token identification
    rev_mapping = {v.lower(): k for k, v in TOKEN_MAPPING.items()}
    df['unified_token_bought'] = df.apply(lambda r: rev_mapping.get(r['token_bought_address'].lower(), r['token_bought_symbol']), axis=1)
    df['unified_token_sold'] = df.apply(lambda r: rev_mapping.get(r['token_sold_address'].lower(), r['token_sold_symbol']), axis=1)

    # Define Trade Direction
    df['trade_direction'] = df.apply(
        lambda row: 'buy_stable' if row['unified_token_bought'] in ['USDC', 'USDT', 'DAI'] else
                    'sell_stable' if row['unified_token_sold'] in ['USDC', 'USDT', 'DAI'] else
                    'buy_eth' if row['unified_token_bought'] == 'WETH' else
                    'sell_eth' if row['unified_token_sold'] == 'WETH' else 'token_swap', axis=1
    )
    
    # Time Features
    df['hour'] = df['block_time'].dt.hour
    df['day_of_week'] = df['block_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # Pricing and Classification
    df['price_bought'] = df['amount_usd'] / df['token_bought_amount']
    df['price_sold'] = df['amount_usd'] / df['token_sold_amount']
    
    df['trade_size_category'] = pd.cut(
        df['amount_usd'], bins=[0, 100, 1000, 10000, 100000, float('inf')],
        labels=['micro', 'small', 'medium', 'large', 'whale']
    )
    
    # Gas Efficiency (Assumption: 1 ETH = $2500)
    df['gas_efficiency'] = df['amount_usd'] / (df['tx_cost_eth'] * 2500)
    return df

eth_df = add_derived_features(eth_df)

# 6. Anomaly Detection (Rapid Trading)
def detect_anomalies(df):
    """Detect anomalous trading behavior"""
    df = df.sort_values(['wallet_address', 'block_time'])
    df['time_since_last'] = df.groupby('wallet_address')['block_time'].diff()
    
    # Flags trades happening within 10 seconds of each other
    anomalies = df[df['time_since_last'] < pd.Timedelta(seconds=10)].copy()
    anomalies['anomaly_type'] = 'rapid_trading'
    
    print(f"\nDetected {len(anomalies)} rapid trading anomalies")
    return anomalies

anomalies = detect_anomalies(eth_df)

# 7. Data Storage & Wallet Summary
eth_df.to_csv('preprocessed_ethereum_trades.csv', index=False)

def create_wallet_summary(df):
    """Create a summary per wallet address"""
    agg_dict = {
        'amount_usd': ['sum', 'count', 'mean'],
        'tx_cost_eth': ['sum', 'mean'],
        'block_time': ['min', 'max']
    }
    summary = df.groupby('wallet_address').agg(agg_dict).round(4)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Additional Statistics
    summary['total_trades'] = df.groupby('wallet_address').size()
    summary['unique_tokens'] = df.groupby('wallet_address')['token_bought_symbol'].nunique()
    
    # Trading Period
    summary['trading_days'] = ((pd.to_datetime(summary['block_time_max']) - 
                                pd.to_datetime(summary['block_time_min'])).dt.days + 1)
    
    # Distribution of trade directions and DEX usage
    for wallet in summary.index:
        w_data = df[df['wallet_address'] == wallet]
        summary.loc[wallet, 'trade_direction_dist'] = str(w_data['trade_direction'].value_counts().to_dict())
        summary.loc[wallet, 'dex_usage_top3'] = str(w_data['dex_name'].value_counts().head(3).to_dict())
        
    return summary

eth_wallet_summary = create_wallet_summary(eth_df)
eth_wallet_summary.to_csv('wallet_summary_ethereum.csv')

# 8. Data Quality Report
report = {
    'chain': 'ethereum',
    'total_records': len(eth_df),
    'unique_wallets': eth_df['wallet_address'].nunique(),
    'total_volume_usd': float(eth_df['amount_usd'].sum()),
    'avg_gas_cost_eth': float(eth_df['tx_cost_eth'].mean()),
    'top_dexes': eth_df['dex_name'].value_counts().head(5).to_dict(),
    'preprocessing_timestamp': datetime.now().isoformat()
}

with open('data_quality_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

# 9. Visualization
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Trading Volume History
    eth_df.groupby(eth_df['block_time'].dt.date)['amount_usd'].sum().plot(
        ax=ax1, title='Daily Ethereum Trading Volume (USD)'
    )
    ax1.set_ylabel('Volume in USD')
    
    # Trade Size Distribution
    eth_df['trade_size_category'].value_counts().plot(
        kind='pie', ax=ax2, autopct='%1.1f%%', title='Trade Size Distribution'
    )
    
    plt.tight_layout()
    plt.savefig('preprocessing_summary.png')
    print("\nVisualization charts generated successfully.")
except Exception as e:
    print(f"Visualization skipped: {e}")

print(f"\n=== Processing Complete: Ethereum {len(eth_df)} records total ===")
