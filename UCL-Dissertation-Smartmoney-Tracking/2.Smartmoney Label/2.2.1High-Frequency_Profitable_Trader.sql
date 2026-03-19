WITH wallet_stats AS (
    SELECT 
        tx_from AS wallet_address,
        COUNT(*) AS trade_count,
        SUM(amount_usd) AS total_volume,
        COUNT(DISTINCT token_bought_symbol) AS unique_tokens
    FROM dex.trades
    WHERE block_time >= TIMESTAMP '2025-01-01'
      AND block_time <  TIMESTAMP '2025-08-01'
      AND amount_usd > 1000
    GROUP BY 1
)

SELECT
    approx_percentile(trade_count, 0.7) AS trade_p70,
    approx_percentile(trade_count, 0.8) AS trade_p80,
    approx_percentile(trade_count, 0.9) AS trade_p90,
    approx_percentile(trade_count, 0.99) AS trade_p99,
    approx_percentile(total_volume, 0.7) AS volume_p70,
    approx_percentile(total_volume, 0.8) AS volume_p80,
    approx_percentile(total_volume, 0.9) AS volume_p90,
    approx_percentile(total_volume, 0.99) AS volume_p99
FROM wallet_stats;