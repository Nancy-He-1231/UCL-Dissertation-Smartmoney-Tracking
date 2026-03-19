-- ============================================================
-- SMART MONEY LABEL (ONLY SMART MONEY WALLETS)
-- Contract-Creation Anchored
-- ============================================================

WITH token_creation AS (
  SELECT
    address AS token_address,
    created_at AS creation_time
  FROM ethereum.contracts
  WHERE created_at BETWEEN TIMESTAMP '2025-03-01'
                        AND TIMESTAMP '2025-08-01'
),

early_trades AS (
  SELECT
    t.tx_from AS wallet_address,
    t.token_bought_address AS token_address,
    MIN(t.block_time) AS first_trade_time
  FROM dex.trades t
  JOIN token_creation c
    ON t.token_bought_address = c.token_address
  WHERE t.block_time BETWEEN c.creation_time
                          AND c.creation_time + INTERVAL '24' HOUR
  GROUP BY 1,2
)

SELECT
  wallet_address,
  COUNT(DISTINCT token_address) AS early_tokens_count,
  1 AS smart_money_label
FROM early_trades
GROUP BY 1
HAVING COUNT(DISTINCT token_address) >= 2;