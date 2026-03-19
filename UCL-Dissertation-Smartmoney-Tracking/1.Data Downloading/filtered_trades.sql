WITH target_wallets AS (
    SELECT wallet_address
    FROM dune.your_username.dataset_ethereum_addresses  -- <-- Replace with the actual path to your uploaded table
),

ethereum_trades AS (
    SELECT 
        t.blockchain,
        t.block_time,
        t.tx_hash,
        t.evt_index,
        t.taker as wallet_address,
        t.token_bought_address,
        COALESCE(t.token_bought_symbol, 'UNKNOWN') as token_bought_symbol,
        t.token_bought_amount,
        t.token_sold_address,
        COALESCE(t.token_sold_symbol, 'UNKNOWN') as token_sold_symbol,
        t.token_sold_amount,
        t.amount_usd,
        COALESCE(t.project, 'UNKNOWN') as dex_name,
        tx.gas_used,
        tx.gas_price,
        DATE(t.block_time) as date,
        EXTRACT(HOUR FROM t.block_time) as hour_of_day
    FROM dex.trades t
    INNER JOIN target_wallets tw 
        ON t.taker = tw.wallet_address
    LEFT JOIN ethereum.transactions tx 
        ON t.tx_hash = tx.hash
    WHERE t.blockchain = 'ethereum'
        AND t.block_time >= CURRENT_DATE - INTERVAL '90' day
        AND t.amount_usd >= 10
        AND t.amount_usd < 5000000
)

SELECT * 
FROM ethereum_trades
ORDER BY wallet_address, block_time, evt_index
