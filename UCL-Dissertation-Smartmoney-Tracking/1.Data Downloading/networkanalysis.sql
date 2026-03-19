-- Query Name: SmartMoney_NetworkAnalysis_Core
-- Description: Calculate network influence score and independent decision ratio

WITH wallet_tokens AS (
    SELECT DISTINCT
        taker AS wallet_address,
        token_bought_address AS token_address,
        MIN(block_time) AS first_trade_time,
        blockchain
    FROM dex.trades
    WHERE taker IN (
        SELECT wallet_address
        FROM dune.your_username.dataset_ethereum_addresses  -- <-- Replace with the actual path to your uploaded table
    )
    AND block_time >= CURRENT_DATE - INTERVAL '90' DAY
    AND token_bought_address != 0x0000000000000000000000000000000000000000
    GROUP BY taker, token_bought_address, blockchain
),

-- Get first trade time for each token globally
token_first_trades AS (
    SELECT 
        token_bought_address AS token_address,
        blockchain,
        MIN(block_time) AS token_first_trade
    FROM dex.trades
    WHERE block_time >= CURRENT_DATE - INTERVAL '90' DAY
      AND token_bought_address != 0x0000000000000000000000000000000000000000
    GROUP BY token_bought_address, blockchain
),

-- Calculate wallet discovery metrics
wallet_discovery_metrics AS (
    SELECT 
        wt.wallet_address,
        wt.token_address,
        wt.first_trade_time,
        tft.token_first_trade,
        DATE_DIFF('minute', tft.token_first_trade, wt.first_trade_time) AS minutes_after_launch,
        CASE WHEN DATE_DIFF('minute', tft.token_first_trade, wt.first_trade_time) <= 5 THEN 1 ELSE 0 END AS ultra_early_discoveries,
        CASE WHEN DATE_DIFF('minute', tft.token_first_trade, wt.first_trade_time) <= 60 THEN 1 ELSE 0 END AS first_hour_discoveries,
        ROW_NUMBER() OVER (PARTITION BY wt.token_address, wt.blockchain ORDER BY wt.first_trade_time) AS trader_rank
    FROM wallet_tokens wt
    INNER JOIN token_first_trades tft 
        ON wt.token_address = tft.token_address 
       AND wt.blockchain = tft.blockchain
),

-- Calculate wallet connections and timing relationships
wallet_connections AS (
    SELECT 
        w1.wallet_address AS wallet_a,
        w2.wallet_address AS wallet_b,
        COUNT(DISTINCT w1.token_address) AS common_tokens,
        CAST(SUM(CASE WHEN w1.first_trade_time < w2.first_trade_time THEN 1 ELSE 0 END) AS DOUBLE) 
            / NULLIF(COUNT(*), 0) AS wallet_a_first_ratio,
        AVG(ABS(DATE_DIFF('minute', w1.first_trade_time, w2.first_trade_time))) AS avg_time_diff_minutes
    FROM wallet_tokens w1
    INNER JOIN wallet_tokens w2 
        ON w1.token_address = w2.token_address 
       AND w1.blockchain = w2.blockchain
       AND w1.wallet_address < w2.wallet_address
    GROUP BY w1.wallet_address, w2.wallet_address
),

-- Aggregate wallet-level metrics
wallet_aggregates AS (
    SELECT 
        wd.wallet_address,
        COUNT(DISTINCT wd.token_address) AS tokens_traded,
        SUM(wd.ultra_early_discoveries) AS ultra_early_discoveries,
        SUM(wd.first_hour_discoveries) AS first_hour_discoveries,
        COUNT(DISTINCT CASE WHEN wd.trader_rank = 1 THEN wd.token_address END) AS first_discoveries,
        AVG(wd.minutes_after_launch) AS avg_minutes_after_launch,
        COUNT(DISTINCT CASE WHEN wc.wallet_a = wd.wallet_address THEN wc.wallet_b 
                           WHEN wc.wallet_b = wd.wallet_address THEN wc.wallet_a END) AS connected_wallets,
        AVG(CASE WHEN wc.wallet_a = wd.wallet_address THEN wc.wallet_a_first_ratio 
                WHEN wc.wallet_b = wd.wallet_address THEN 1 - wc.wallet_a_first_ratio END) AS independent_decision_ratio,
        AVG(CASE 
            WHEN wc.avg_time_diff_minutes <= 5 THEN 100
            WHEN wc.avg_time_diff_minutes <= 30 THEN 80
            WHEN wc.avg_time_diff_minutes <= 180 THEN 60
            ELSE 20
        END) AS avg_synchronicity_score
    FROM wallet_discovery_metrics wd
    LEFT JOIN wallet_connections wc 
        ON wd.wallet_address = wc.wallet_a OR wd.wallet_address = wc.wallet_b
    GROUP BY wd.wallet_address
),

-- Final calculation with only network influence score and independent decision ratio
final_calculation AS (
    SELECT 
        wallet_address,
        tokens_traded,
        connected_wallets,
        independent_decision_ratio,
        avg_synchronicity_score,
        
        -- Network Influence Score Calculation (0-100 scale)
        (
            -- 1. Discovery Ability (40% weight)
            (
                LEAST(40,
                    CASE WHEN tokens_traded > 0 
                        THEN (ultra_early_discoveries * 1.0 / tokens_traded) * 20 
                        ELSE 0 END +
                    CASE WHEN tokens_traded > 0 
                        THEN (first_hour_discoveries * 1.0 / tokens_traded) * 20 
                        ELSE 0 END
                )
            ) * 0.4 +
            
            -- 2. Network Influence (30% weight)
            (
                LEAST(30,
                    LEAST(15, connected_wallets * 1.5) +
                    CASE 
                        WHEN independent_decision_ratio > 0.8 THEN 15
                        WHEN independent_decision_ratio > 0.6 THEN independent_decision_ratio * 18.75
                        WHEN independent_decision_ratio < 0.3 THEN -5
                        ELSE 5
                    END
                )
            ) * 0.3 +
            
            -- 3. Independence (20% weight)
            (
                LEAST(20,
                    CASE WHEN tokens_traded > 0 
                        THEN (first_discoveries * 1.0 / tokens_traded) * 10 
                        ELSE 0 END +
                    CASE 
                        WHEN avg_synchronicity_score < 30 THEN 10
                        WHEN avg_synchronicity_score < 50 THEN 5
                        ELSE 0
                    END
                )
            ) * 0.2 +
            
            -- 4. Reaction Speed (10% weight)
            (
                CASE
                    WHEN avg_minutes_after_launch < 30 THEN 10
                    WHEN avg_minutes_after_launch < 180 THEN 8
                    WHEN avg_minutes_after_launch < 720 THEN 5
                    WHEN avg_minutes_after_launch < 1440 THEN 3
                    ELSE 1
                END
            ) * 0.1
            
        ) AS network_influence_score
        
    FROM wallet_aggregates
)

-- Final output with only the two required metrics
SELECT 
    wallet_address,
    network_influence_score,
    independent_decision_ratio
FROM final_calculation
ORDER BY network_influence_score DESC;
