WITH wallet_list AS (
    SELECT DISTINCT wallet_address
    FROM dune.your_username.dataset_ethereum_addresses  -- <-- Replace with the actual path to your uploaded table
),

wallet_activity AS (
    SELECT 
        taker AS wallet_address,
        blockchain,
        COUNT(DISTINCT tx_hash) AS total_trades,
        COUNT(DISTINCT DATE(block_time)) AS active_days,
        COUNT(DISTINCT token_bought_address) AS unique_tokens_bought,
        COUNT(DISTINCT token_sold_address) AS unique_tokens_sold,
        COUNT(DISTINCT project) AS dex_count,
        SUM(amount_usd) AS total_volume_usd,
        AVG(amount_usd) AS avg_trade_size,
        STDDEV(amount_usd) AS trade_size_stddev,
        MIN(block_time) AS first_trade,
        MAX(block_time) AS last_trade,
        
        -- Calculate daily averages
        COUNT(DISTINCT tx_hash) / NULLIF(COUNT(DISTINCT DATE(block_time)), 0) AS avg_trades_per_day,
        SUM(amount_usd) / NULLIF(COUNT(DISTINCT DATE(block_time)), 0) AS avg_volume_per_day,
        
        -- Trading time distribution
        COUNT(DISTINCT CASE WHEN EXTRACT(HOUR FROM block_time) BETWEEN 0 AND 5 THEN DATE(block_time) END) AS late_night_days,
        COUNT(DISTINCT CASE WHEN EXTRACT(HOUR FROM block_time) BETWEEN 6 AND 11 THEN DATE(block_time) END) AS morning_days,
        COUNT(DISTINCT CASE WHEN EXTRACT(HOUR FROM block_time) BETWEEN 12 AND 17 THEN DATE(block_time) END) AS afternoon_days,
        COUNT(DISTINCT CASE WHEN EXTRACT(HOUR FROM block_time) BETWEEN 18 AND 23 THEN DATE(block_time) END) AS evening_days,
        
        -- Categorize wallets
        CASE 
            WHEN COUNT(DISTINCT tx_hash) / NULLIF(COUNT(DISTINCT DATE(block_time)), 0) > 50 THEN 'ultra_high_freq'
            WHEN COUNT(DISTINCT tx_hash) / NULLIF(COUNT(DISTINCT DATE(block_time)), 0) > 20 THEN 'high_freq'
            WHEN COUNT(DISTINCT tx_hash) / NULLIF(COUNT(DISTINCT DATE(block_time)), 0) > 5 THEN 'medium_freq'
            WHEN COUNT(DISTINCT tx_hash) / NULLIF(COUNT(DISTINCT DATE(block_time)), 0) > 1 THEN 'low_freq'
            ELSE 'occasional'
        END AS trader_category,
        
        -- Recent activity
        COUNT(CASE WHEN block_time >= CURRENT_DATE - INTERVAL '7' day THEN 1 END) AS trades_last_7d,
        COUNT(CASE WHEN block_time >= CURRENT_DATE - INTERVAL '30' day THEN 1 END) AS trades_last_30d
        
    FROM dex.trades
    WHERE blockchain IN ('ethereum', 'base')
        AND block_time >= CURRENT_DATE - INTERVAL '90' day
        AND taker IN (SELECT wallet_address FROM wallet_list)
        AND amount_usd > 0
    GROUP BY taker, blockchain
)

SELECT 
    wallet_address,
    blockchain,
    trader_category,
    total_trades,
    active_days,
    avg_trades_per_day,
    total_volume_usd,
    avg_trade_size,
    unique_tokens_bought + unique_tokens_sold AS total_unique_tokens,
    dex_count,
    first_trade,
    last_trade,
    trades_last_7d,
    trades_last_30d,
    
    -- Calculate activity score
    (total_trades * 0.3 + 
     active_days * 0.3 + 
     LN(total_volume_usd + 1) * 0.2 + 
     (unique_tokens_bought + unique_tokens_sold) * 0.2) AS activity_score,
     
    -- Trading time preference
    CASE 
        WHEN late_night_days > GREATEST(morning_days, afternoon_days, evening_days) THEN 'late_night'
        WHEN morning_days > GREATEST(afternoon_days, evening_days) THEN 'morning'
        WHEN afternoon_days > evening_days THEN 'afternoon'
        ELSE 'evening'
    END AS preferred_trading_time

FROM wallet_activity
ORDER BY activity_score DESC
