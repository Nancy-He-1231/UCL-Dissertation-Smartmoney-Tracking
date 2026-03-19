import pandas as pd

def create_token_minute_pairs(input_csv_path: str, output_csv_path: str):
    """
    Extract all unique (token address, minute timestamp) combinations from trading data.

    Parameters:
    input_csv_path (str): Path to the input CSV file containing trading data.
    output_csv_path (str): Path to save the CSV file with unique combinations.
    """
    try:
        # 1. Load your trading data
        print(f"Loading data from '{input_csv_path}'...")
        df = pd.read_csv(input_csv_path)
        print(f"Successfully loaded {len(df)} trading records.")

        # 2. Ensure block_time is in datetime format and round it to the minute
        df['minute_timestamp'] = pd.to_datetime(df['block_time']).dt.floor('T')

        # 3. Process bought and sold tokens separately
        # Extract (bought token, minute) pairs
        bought_pairs = df[['token_bought_address', 'minute_timestamp']].copy()
        bought_pairs.rename(columns={'token_bought_address': 'token_address'}, inplace=True)

        # Extract (sold token, minute) pairs
        sold_pairs = df[['token_sold_address', 'minute_timestamp']].copy()
        sold_pairs.rename(columns={'token_sold_address': 'token_address'}, inplace=True)

        # 4. Combine and deduplicate to get unique (token, minute) combinations
        all_pairs = pd.concat([bought_pairs, sold_pairs])
        all_pairs.dropna(subset=['token_address'], inplace=True)  # Remove rows without addresses
        unique_pairs = all_pairs.drop_duplicates().sort_values(by=['token_address', 'minute_timestamp'])

        # 5. Save to a new CSV file
        unique_pairs.to_csv(output_csv_path, index=False)
        
        print(f"\nProcessing complete!")
        print(f"Found {len(unique_pairs)} unique (token, minute) combinations.")
        print(f"Results saved to '{output_csv_path}'.")

    except FileNotFoundError:
        print(f"Error: File '{input_csv_path}' not found.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == '__main__':
    INPUT_FILE = 'filtered_trades_add20000.csv'  # Your input file name
    OUTPUT_FILE = 'token_minute_pairs20000.csv'  # This is the file we need to upload to Dune
    
    create_token_minute_pairs(INPUT_FILE, OUTPUT_FILE)
