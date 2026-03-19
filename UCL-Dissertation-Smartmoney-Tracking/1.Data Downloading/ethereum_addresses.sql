WITH 

excluded_cex AS (
  SELECT address 
  FROM labels.cex
  WHERE category = 'CEX'
),

high_freq_bots AS (
  SELECT "from" AS address
  FROM ethereum.transactions
  WHERE block_time >= now() - interval '30' day
    AND success = true
  GROUP BY "from", date_trunc('day', block_time)
  HAVING COUNT(*) > 100
),

defi_trades AS (
  SELECT 
    tx_from AS wallet_address,
    COUNT(*) AS total_trades,
    COUNT(DISTINCT token_sold_address) + COUNT(DISTINCT token_bought_address) AS unique_tokens,
    COUNT(DISTINCT date_trunc('day', block_time)) AS active_days,
    SUM(amount_usd) AS total_volume_usd,
    AVG(amount_usd) AS avg_trade_size,
    MAX(block_time) AS last_trade_time
  FROM dex.trades
  WHERE blockchain = 'ethereum'
    AND block_time >= now() - interval '90' day
    AND tx_from IS NOT NULL
    AND amount_usd > 1
    AND tx_from NOT IN (SELECT address FROM excluded_cex)
    AND tx_from NOT IN (SELECT address FROM high_freq_bots)
  GROUP BY tx_from
  HAVING 
    COUNT(*) > 10
    AND COUNT(DISTINCT date_trunc('day', block_time)) >= 5
)

SELECT 
  wallet_address,
  total_trades,
  unique_tokens,
  active_days,
  total_volume_usd,
  avg_trade_size,
  total_volume_usd / total_trades AS avg_trade_value,
  last_trade_time
FROM defi_trades
ORDER BY total_volume_usd DESC
LIMIT 50000;
