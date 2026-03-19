WITH
/* -------------------------------------------------
   1. Sample wallets (pre-selected universe)
-------------------------------------------------- */
sample_wallets AS (
    SELECT wallet_address
    FROM dune.zinan_team_3604.dataset_ethereum_addresses
),

/* -------------------------------------------------
   2. Daily net token flows from DEX trades
-------------------------------------------------- */
daily_flows AS (
    -- Token purchases (positive inventory change)
    SELECT
        t.tx_from AS wallet_address,
        DATE_TRUNC('day', t.block_time) AS day,
        t.token_bought_address AS token_address,
        SUM(t.token_bought_amount) AS net_amount
    FROM dex.trades t
    JOIN sample_wallets s
      ON t.tx_from = s.wallet_address
    WHERE t.block_time BETWEEN TIMESTAMP '2025-05-09'
                           AND TIMESTAMP '2025-08-01'
    GROUP BY 1,2,3

    UNION ALL

    -- Token sales (negative inventory change)
    SELECT
        t.tx_from AS wallet_address,
        DATE_TRUNC('day', t.block_time) AS day,
        t.token_sold_address AS token_address,
        -SUM(t.token_sold_amount) AS net_amount
    FROM dex.trades t
    JOIN sample_wallets s
      ON t.tx_from = s.wallet_address
    WHERE t.block_time BETWEEN TIMESTAMP '2025-05-09'
                           AND TIMESTAMP '2025-08-01'
    GROUP BY 1,2,3
),

/* -------------------------------------------------
   3. Daily token holdings (cumulative inventory)
-------------------------------------------------- */
daily_holdings AS (
    SELECT
        wallet_address,
        token_address,
        day,
        SUM(net_amount) OVER (
            PARTITION BY wallet_address, token_address
            ORDER BY day
        ) AS token_balance
    FROM daily_flows
),

/* -------------------------------------------------
   4. Daily USD token prices
-------------------------------------------------- */
daily_prices AS (
    SELECT
        contract_address AS token_address,
        DATE_TRUNC('day', minute) AS day,
        AVG(price) AS price_usd
    FROM prices.usd
    WHERE minute BETWEEN TIMESTAMP '2025-05-09'
                      AND TIMESTAMP '2025-08-01'
    GROUP BY 1,2
),

/* -------------------------------------------------
   5. Daily mark-to-market portfolio value
-------------------------------------------------- */
daily_portfolio_value AS (
    SELECT
        h.wallet_address,
        h.day,
        SUM(h.token_balance * p.price_usd) AS portfolio_value
    FROM daily_holdings h
    JOIN daily_prices p
      ON h.token_address = p.token_address
     AND h.day = p.day
    GROUP BY 1,2
),

/* -------------------------------------------------
   6. Daily profit and loss (PnL)
-------------------------------------------------- */
daily_pnl AS (
    SELECT
        wallet_address,
        day,
        portfolio_value
        - LAG(portfolio_value) OVER (
            PARTITION BY wallet_address
            ORDER BY day
        ) AS daily_pnl
    FROM daily_portfolio_value
),

/* -------------------------------------------------
   7. Wallet-level performance statistics
-------------------------------------------------- */
wallet_pnl_stats AS (
    SELECT
        wallet_address,
        COUNT(day) AS trading_days,
        SUM(daily_pnl) AS total_pnl,
        AVG(daily_pnl) AS avg_daily_pnl,
        STDDEV(daily_pnl) AS pnl_volatility,
        CASE
            WHEN STDDEV(daily_pnl) > 0
            THEN AVG(daily_pnl) / STDDEV(daily_pnl)
            ELSE NULL
        END AS sharpe_ratio,
        SUM(CASE WHEN daily_pnl > 0 THEN 1 ELSE 0 END) * 1.0
            / COUNT(day) AS win_rate
    FROM daily_pnl
    WHERE daily_pnl IS NOT NULL
    GROUP BY 1
),

/* -------------------------------------------------
   8. Active wallets filter (noise reduction)
-------------------------------------------------- */
active_wallets AS (
    SELECT *
    FROM wallet_pnl_stats
    WHERE trading_days >= 40
),

/* -------------------------------------------------
   9. Cross-sectional percentile thresholds
-------------------------------------------------- */
thresholds AS (
    SELECT
        APPROX_PERCENTILE(total_pnl, 0.90) AS pnl_cut,
        APPROX_PERCENTILE(sharpe_ratio, 0.90) AS sharpe_cut,
        APPROX_PERCENTILE(win_rate, 0.90) AS winrate_cut
    FROM active_wallets
)

/* -------------------------------------------------
   10. Sustained Profitability Label (Percentile-based)
-------------------------------------------------- */
SELECT
    a.wallet_address,
    a.trading_days,
    a.total_pnl,
    a.sharpe_ratio,
    a.win_rate,
    1 AS sustained_profitability_label
FROM active_wallets a
JOIN thresholds t ON TRUE
WHERE
    a.total_pnl >= t.pnl_cut
    AND a.sharpe_ratio >= t.sharpe_cut
    AND a.win_rate >= t.winrate_cut
ORDER BY a.total_pnl DESC;
