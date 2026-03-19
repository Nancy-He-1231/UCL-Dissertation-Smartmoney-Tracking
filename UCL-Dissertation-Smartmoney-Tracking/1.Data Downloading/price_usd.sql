-- Query: Efficient retrieval of token prices at specific timestamps
-- Instructions:
-- 1. Upload a CSV file containing the columns 'token_address' and 'minute_timestamp' to Dune.
-- 2. Replace 'dune.your_username.dataset_token_minute_pairs' below with the actual path to your uploaded table.

WITH specific_requests AS (
    -- Read directly from the uploaded CSV table
    SELECT 
        "token_address" AS contract_address,
        CAST("minute_timestamp" AS TIMESTAMP) AS minute -- Ensure correct data type
    FROM dune.medola02_team_3646.dataset_token_minute_pairs500 -- <-- Replace with the actual path to your uploaded table
)

SELECT 
    p.minute AS block_time,
    p.contract_address,
    p.price AS price_usd,
    p.blockchain
FROM prices.usd p
INNER JOIN specific_requests sr 
    ON p.contract_address = sr.contract_address 
    AND p.minute = sr.minute
WHERE p.blockchain IN ('ethereum', 'base')