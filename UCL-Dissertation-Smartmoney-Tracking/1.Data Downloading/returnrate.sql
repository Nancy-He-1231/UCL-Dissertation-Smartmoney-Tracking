WITH numbered_wallets AS (
    SELECT 
        wallet_address,
        ROW_NUMBER() OVER (ORDER BY wallet_address) AS rn
    FROM dune.your_username.dataset_ethereum_addresses  -- <-- Replace with the actual path to your uploaded table
),

target_wallets AS (
    SELECT wallet_address
    FROM numbered_wallets
    WHERE rn BETWEEN 1 AND 1000
),

wallet_positions AS (
    SELECT 
        t.taker AS wallet_address,
        t.token_bought_address AS token,
        t.tx_hash,
        t.block_time,
        t.token_bought_amount AS amount,
        t.amount_usd,
        t.amount_usd / NULLIF(t.token_bought_amount, 0) AS entry_price,
        CAST(NULL AS DOUBLE) AS exit_price,
        'BUY' AS action
    FROM dex.trades t
    INNER JOIN target_wallets tw 
        ON t.taker = tw.wallet_address
    
    UNION ALL
    
    SELECT 
        t.taker AS wallet_address,
        t.token_sold_address AS token,
        t.tx_hash,
        t.block_time,
        -t.token_sold_amount AS amount,
        t.amount_usd,
        CAST(NULL AS DOUBLE) AS entry_price,
        t.amount_usd / NULLIF(t.token_sold_amount, 0) AS exit_price,
        'SELL' AS action
    FROM dex.trades t
    INNER JOIN target_wallets tw 
        ON t.taker = tw.wallet_address
),

paired_trades AS (
    SELECT 
        b.wallet_address,
        b.token,
        b.tx_hash AS buy_tx,
        s.tx_hash AS sell_tx,
        b.block_time AS buy_time,
        s.block_time AS sell_time,
        b.entry_price,
        s.exit_price,
        b.amount AS buy_amount,
        ABS(s.amount) AS sell_amount,
        LEAST(b.amount, ABS(s.amount)) AS matched_amount,
        (s.exit_price - b.entry_price) 
            * LEAST(b.amount, ABS(s.amount)) AS pnl_usd,
        (s.exit_price - b.entry_price) 
            / NULLIF(b.entry_price, 0) * 100 AS return_pct,
        DATE_DIFF('second', b.block_time, s.block_time) / 60.0 
            AS holding_minutes
    FROM wallet_positions b
    INNER JOIN wallet_positions s
        ON b.wallet_address = s.wallet_address
        AND b.token = s.token
        AND b.action = 'BUY'
        AND s.action = 'SELL'
        AND s.block_time > b.block_time
    WHERE b.amount > 0 
      AND s.amount < 0
)

SELECT 
    wallet_address,
    token,
    COUNT(*) AS total_trades,
    SUM(pnl_usd) AS total_pnl_usd,
    AVG(return_pct) AS avg_return,
    STDDEV(return_pct) AS return_volatility,
    AVG(CASE WHEN return_pct < 0 THEN holding_minutes END) 
        AS avg_loss_holding_minutes,
    AVG(CASE WHEN return_pct > 0 THEN holding_minutes END) 
        AS avg_profit_holding_minutes,
    SUM(CASE WHEN return_pct < -10 THEN 1 ELSE 0 END) 
        AS deep_loss_count,
    AVG(CASE WHEN return_pct < 0 THEN return_pct END) 
        AS avg_loss_pct
FROM paired_trades
GROUP BY 1, 2;
