-- Query Name: SmartMoney_NetworkAnalysis

WITH wallet_tokens AS (
    SELECT DISTINCT
        taker AS wallet_address,
        token_bought_address AS token_address,
        MIN(block_time) AS first_trade_time,
        blockchain
    FROM dex.trades
    WHERE taker IN (
        (0x7547cfdfc1ec39da52d7a28e5f24f545f5d9516b),
        (0x32fb1743426f37d40d0d50a30cdda1b158e6ba24),
        (0xd833bbeaea157ea5b6aae3b59de8b9b8864d8d10),
        (0x34d4b978f6b73e9365de2e20ad37122272982af4),
        (0x404ae894ab8a5bc77b6b8bbb69c8b240ab8b1d9c),
        (0x2bad6f49838f8a28bddc1c81424744bd89459c66),
        (0x381b7d64303a2a5251ac12ee147ffdb337da5969),
        (0x67bb6879b16ef6bc9035e92dd6c24facacb931b6),
        (0x74de399993e4777e4e3b245aa473a12ba670e29d),
        (0x8c50b33d138486f23b3472f69b5890745ba54e84)
        -- (wallet list truncated for readability)
    )
      AND block_time >= CURRENT_DATE - INTERVAL '90' DAY
      AND token_bought_address != 0x0000000000000000000000000000000000000000
    GROUP BY taker, token_bought_address, blockchain
),

-- Token-level statistics: earliest trade time and total number of traders
token_popularity AS (
    SELECT 
        token_bought_address AS token_address,
        blockchain,
        COUNT(DISTINCT taker) AS total_traders,
        MIN(block_time) AS token_first_trade
    FROM dex.trades
    WHERE block_time >= CURRENT_DATE - INTERVAL '90' DAY
      AND token_bought_address != 0x0000000000000000000000000000000000000000
    GROUP BY token_bought_address, blockchain
),

-- Wallet discovery behaviour relative to token launch
wallet_discovery_metrics AS (
    SELECT 
        wt.wallet_address,
        wt.token_address,
        wt.first_trade_time,
        tp.token_first_trade,
        tp.total_traders,
        DATE_DIFF('minute', tp.token_first_trade, wt.first_trade_time) AS minutes_after_launch,

        CASE WHEN DATE_DIFF('minute', tp.token_first_trade, wt.first_trade_time) <= 5 THEN 1 ELSE 0 END AS ultra_early_discoveries,
        CASE WHEN DATE_DIFF('minute', tp.token_first_trade, wt.first_trade_time) <= 60 THEN 1 ELSE 0 END AS first_hour_discoveries,
        CASE WHEN DATE_DIFF('hour', tp.token_first_trade, wt.first_trade_time) < 24 THEN 1 ELSE 0 END AS is_early_adopter,

        CASE 
            WHEN wt.first_trade_time < tp.token_first_trade + INTERVAL '1' HOUR THEN 100
            WHEN wt.first_trade_time < tp.token_first_trade + INTERVAL '1' DAY THEN 80
            WHEN wt.first_trade_time < tp.token_first_trade + INTERVAL '7' DAY THEN 60
            WHEN wt.first_trade_time < tp.token_first_trade + INTERVAL '30' DAY THEN 40
            ELSE 20
        END AS discovery_score,

        ROW_NUMBER() OVER (
            PARTITION BY wt.token_address, wt.blockchain 
            ORDER BY wt.first_trade_time
        ) AS trader_rank
    FROM wallet_tokens wt
    INNER JOIN token_popularity tp 
        ON wt.token_address = tp.token_address 
       AND wt.blockchain = tp.blockchain
),

-- Pairwise wallet connectivity and synchronisation metrics
wallet_connections AS (
    SELECT 
        w1.wallet_address AS wallet_a,
        w2.wallet_address AS wallet_b,
        COUNT(DISTINCT w1.token_address) AS common_tokens,
        AVG(ABS(DATE_DIFF('minute', w1.first_trade_time, w2.first_trade_time))) AS avg_time_diff_minutes,

        CAST(
            SUM(CASE WHEN w1.first_trade_time < w2.first_trade_time THEN 1 ELSE 0 END) 
            AS DOUBLE
        ) / NULLIF(COUNT(*), 0) AS wallet_a_first_ratio,

        CASE 
            WHEN AVG(ABS(DATE_DIFF('minute', w1.first_trade_time, w2.first_trade_time))) <= 5 THEN 100
            WHEN AVG(ABS(DATE_DIFF('minute', w1.first_trade_time, w2.first_trade_time))) <= 30 THEN 80
            WHEN AVG(ABS(DATE_DIFF('minute', w1.first_trade_time, w2.first_trade_time))) <= 180 THEN 60
            ELSE 20
        END AS synchronicity_score,

        SUM(
            CASE 
                WHEN ABS(DATE_DIFF('minute', w1.first_trade_time, w2.first_trade_time)) <= 5 
                THEN 1 ELSE 0 
            END
        ) AS sync_trades_5min
    FROM wallet_tokens w1
    INNER JOIN wallet_tokens w2 
        ON w1.token_address = w2.token_address 
       AND w1.blockchain = w2.blockchain
       AND w1.wallet_address < w2.wallet_address
    GROUP BY w1.wallet_address, w2.wallet_address
),

-- Wallet-level influence, leadership, and behavioural aggregation
wallet_influence_metrics AS (
    SELECT 
        wd.wallet_address,
        COUNT(DISTINCT wd.token_address) AS tokens_traded,
        SUM(wd.is_early_adopter) AS early_adoptions,
        SUM(wd.ultra_early_discoveries) AS ultra_early_discoveries,
        SUM(wd.first_hour_discoveries) AS first_hour_discoveries,
        AVG(wd.discovery_score) AS avg_discovery_score,
        AVG(wd.trader_rank) AS avg_trader_rank,

        COUNT(DISTINCT CASE WHEN wd.trader_rank <= 10 THEN wd.token_address END) AS top10_discoveries,
        COUNT(DISTINCT CASE WHEN wd.trader_rank = 1 THEN wd.token_address END) AS first_discoveries,

        COUNT(DISTINCT 
            CASE 
                WHEN wc.wallet_a = wd.wallet_address THEN wc.wallet_b 
                WHEN wc.wallet_b = wd.wallet_address THEN wc.wallet_a 
            END
        ) AS connected_wallets,

        AVG(
            CASE 
                WHEN wc.wallet_a = wd.wallet_address THEN wc.wallet_a_first_ratio 
                WHEN wc.wallet_b = wd.wallet_address THEN 1 - wc.wallet_a_first_ratio 
            END
        ) AS lead_ratio,

        AVG(wc.synchronicity_score) AS avg_synchronicity_score,
        SUM(wc.sync_trades_5min) AS sync_trades_5min,
        AVG(wd.minutes_after_launch) AS avg_minutes_after_launch,
        COUNT(*) AS total_connections
    FROM wallet_discovery_metrics wd
    LEFT JOIN wallet_connections wc 
        ON wd.wallet_address = wc.wallet_a 
        OR wd.wallet_address = wc.wallet_b
    GROUP BY wd.wallet_address
),

-- Final scoring and role classification
final_network_analysis AS (
    SELECT 
        *,

        -- 1. Discovery capability (0–40)
        LEAST(40,
            CASE WHEN tokens_traded > 0 
                THEN (ultra_early_discoveries * 1.0 / tokens_traded) * 20 
                ELSE 0 END
            +
            CASE WHEN tokens_traded > 0 
                THEN (first_hour_discoveries * 1.0 / tokens_traded) * 20 
                ELSE 0 END
        ) AS discovery_ability_score,

        -- 2. Network influence (0–30)
        LEAST(30,
            LEAST(15, connected_wallets * 1.5) +
            CASE 
                WHEN lead_ratio > 0.8 THEN 15
                WHEN lead_ratio > 0.6 THEN lead_ratio * 18.75
                WHEN lead_ratio < 0.3 THEN -5
                ELSE 5
            END
        ) AS influence_score,

        -- 3. Independence (0–20)
        LEAST(20,
            CASE WHEN tokens_traded > 0 
                THEN (first_discoveries * 1.0 / tokens_traded) * 10 
                ELSE 0 END
            +
            CASE 
                WHEN avg_synchronicity_score < 30 THEN 10 
                WHEN avg_synchronicity_score < 50 THEN 5 
                ELSE 0 
            END
        ) AS independence_score,

        -- 4. Reaction speed (0–10)
        CASE
            WHEN avg_minutes_after_launch < 30 THEN 10
            WHEN avg_minutes_after_launch < 180 THEN 8
            WHEN avg_minutes_after_launch < 720 THEN 5
            WHEN avg_minutes_after_launch < 1440 THEN 3
            ELSE 1
        END AS reaction_speed_score,

        -- Composite network influence score
        (
            discovery_ability_score * 0.4 +
            influence_score * 0.3 +
            independence_score * 0.2 +
            reaction_speed_score * 0.1
        ) AS network_influence_score,

        -- Independent decision-making ratio
        1 - (avg_synchronicity_score / 100.0) AS independent_decision_ratio,

        -- Behavioural role classification
        CASE 
            WHEN (ultra_early_discoveries * 1.0 / NULLIF(tokens_traded, 0)) > 0.2
                AND lead_ratio > 0.7 
                AND connected_wallets > 10 THEN 'Alpha Leader'

            WHEN (first_hour_discoveries * 1.0 / NULLIF(tokens_traded, 0)) > 0.3
                AND avg_discovery_score > 75 
                AND connected_wallets < 10 THEN 'Smart Money'

            WHEN sync_trades_5min > total_connections * 0.4
                AND lead_ratio < 0.4 
                AND avg_synchronicity_score > 70 THEN 'Copy Trader'

            WHEN first_hour_discoveries > 5 
                AND avg_discovery_score > 60 THEN 'Early Adopter'

            WHEN lead_ratio < 0.3 
                AND connected_wallets > 5 THEN 'Follower'

            WHEN connected_wallets < 3 
                AND tokens_traded > 10 THEN 'Lone Wolf'

            WHEN tokens_traded < 5 THEN 'Inactive'
            ELSE 'Regular Trader'
        END AS network_role
    FROM wallet_influence_metrics
)

SELECT *
FROM final_network_analysis
ORDER BY network_influence_score DESC;