import pandas as pd
import os
import numpy as np
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')

class DuneSmartMoneyProcessor:
    """
    Process Smart Money data downloaded from Dune Analytics
    Merge multiple CSV files and generate labels
    """
    
    def __init__(self, folder_path: str = r"C:\Users\11240\Desktop\Smartmoneydata"):
        self.folder_path = folder_path
        self.all_smart_money_addresses = set()
        self.dune_data_summary = []
        
    def scan_csv_files(self) -> List[str]:
        """
        Scan all CSV files in the folder
        """
        print(f"Scanning folder: {self.folder_path}")
        
        csv_files = []
        if os.path.exists(self.folder_path):
            for file in os.listdir(self.folder_path):
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(self.folder_path, file))
                    print(f"  ✓ Found CSV file: {file}")
        else:
            print(f"❌ Folder does not exist: {self.folder_path}")
            
        print(f"Total CSV files found: {len(csv_files)}")
        return csv_files
    
    def analyze_csv_structure(self, file_path: str) -> Dict:
        """
        Analyze the structure of a single CSV file
        """
        try:
            df = pd.read_csv(file_path)
            file_name = os.path.basename(file_path)
            
            analysis = {
                'file_name': file_name,
                'file_path': file_path,
                'rows': len(df),
                'columns': list(df.columns),
                'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
            }
            
            print(f"\n📊 Analyzing file: {file_name}")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Failed to read file {file_path}: {e}")
            return None
    
    def extract_addresses_from_csv(self, file_path: str, file_analysis: Dict) -> set:
        """
        Extract wallet addresses from CSV file
        """
        try:
            df = pd.read_csv(file_path)
            addresses = set()
            
            # Common address column names
            possible_address_columns = [
                'address', 'wallet_address', 'trader', 'from', 'to', 
                'user_address', 'account', 'wallet', 'owner'
            ]
            
            # Find address column
            address_column = None
            for col in df.columns:
                if col.lower() in [pc.lower() for pc in possible_address_columns]:
                    address_column = col
                    break
                # Check if column contains data starting with 0x
                elif df[col].dtype == 'object':
                    sample_values = df[col].dropna().head(5).astype(str)
                    if any(str(val).startswith('0x') and len(str(val)) == 42 for val in sample_values):
                        address_column = col
                        break
            
            if address_column:
                # Extract and clean addresses
                raw_addresses = df[address_column].dropna().astype(str)
                for addr in raw_addresses:
                    addr = addr.strip().lower()
                    if addr.startswith('0x') and len(addr) == 42:
                        addresses.add(addr)
                
                print(f"  ✓ Extracted {len(addresses)} addresses from column '{address_column}'")
            else:
                print(f"  ❌ No address column found")
                print(f"     Available columns: {list(df.columns)}")
                print(f"     Please check data format")
            
            return addresses
            
        except Exception as e:
            print(f"❌ Failed to process file {file_path}: {e}")
            return set()
    
    def process_all_csv_files(self) -> set:
        """
        Process all CSV files and merge Smart Money addresses
        """
        print("=" * 60)
        print("🚀 Starting Dune Analytics CSV file processing")
        print("=" * 60)
        
        csv_files = self.scan_csv_files()
        
        if not csv_files:
            print("❌ No CSV files found, please check folder path")
            return set()
        
        all_addresses = set()
        
        for file_path in csv_files:
            print(f"\n📁 Processing file: {os.path.basename(file_path)}")
            
            # Analyze file structure
            analysis = self.analyze_csv_structure(file_path)
            if analysis:
                self.dune_data_summary.append(analysis)
                
                # Extract addresses
                addresses = self.extract_addresses_from_csv(file_path, analysis)
                all_addresses.update(addresses)
                
                print(f"  Current total addresses: {len(all_addresses)}")
        
        self.all_smart_money_addresses = all_addresses
        return all_addresses
    
    def load_existing_wallet_features(self) -> pd.DataFrame:
        """
        Load existing wallet feature data
        """
        possible_feature_files = [
            'merged_ethereum_features.csv',
            'final_all_features.csv',
            'combined_hdbscan_lstm_features.csv'
        ]
        
        for file_name in possible_feature_files:
            try:
                df = pd.read_csv(file_name)
                print(f"✓ Loaded feature file: {file_name} - Shape: {df.shape}")
                return df
            except FileNotFoundError:
                continue
        
        print("❌ Feature file not found, please ensure HDBSCAN/LSTM/GAT feature extraction has been run")
        return None
    
    def create_smart_money_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Smart Money labels based on Dune data
        """
        print(f"\n🏷️ Creating Smart Money labels...")
        print(f"Dune Smart Money addresses count: {len(self.all_smart_money_addresses)}")
        print(f"Feature data wallet count: {len(features_df)}")
        
        # Ensure consistent address format (lowercase)
        features_df['wallet_address_lower'] = features_df['wallet_address'].str.lower()
        
        # Create labels
        labels = []
        smart_money_count = 0
        
        for _, row in features_df.iterrows():
            wallet_addr = row['wallet_address_lower']
            is_smart_money = 1 if wallet_addr in self.all_smart_money_addresses else 0
            labels.append(is_smart_money)
            
            if is_smart_money:
                smart_money_count += 1
        
        # Create label DataFrame
        labels_df = pd.DataFrame({
            'wallet_address': features_df['wallet_address'],  # Maintain original format
            'is_smart_money': labels
        })
        
        print(f"✓ Label creation completed:")
        print(f"  Smart Money: {smart_money_count} addresses ({smart_money_count/len(labels_df)*100:.1f}%)")
        print(f"  Ordinary users: {len(labels_df) - smart_money_count} addresses")
        
        return labels_df
    
    def generate_summary_report(self):
        """
        Generate processing summary report
        """
        print("\n" + "=" * 60)
        print("📊 DUNE Data Processing Summary Report")
        print("=" * 60)
        
        print(f"Number of processed CSV files: {len(self.dune_data_summary)}")
        print(f"Total Smart Money addresses extracted: {len(self.all_smart_money_addresses)}")
        
        print(f"\n📁 File details:")
        for i, summary in enumerate(self.dune_data_summary, 1):
            print(f"  {i}. {summary['file_name']}: {summary['rows']} rows")
        
        # Display sample Smart Money addresses
        if self.all_smart_money_addresses:
            sample_addresses = list(self.all_smart_money_addresses)[:5]
            print(f"\n🔍 Smart Money address examples:")
            for addr in sample_addresses:
                print(f"  {addr}")
            if len(self.all_smart_money_addresses) > 5:
                print(f"  ... and {len(self.all_smart_money_addresses)-5} more addresses")
    
    def save_results(self, labels_df: pd.DataFrame):
        """
        Save result files
        """
        # Save label file
        labels_df.to_csv('smart_money_labels.csv', index=False)
        print(f"✅ Label file saved: smart_money_labels.csv")
        
        # Save Smart Money address list
        addresses_df = pd.DataFrame({
            'wallet_address': list(self.all_smart_money_addresses)
        })
        addresses_df.to_csv('dune_smart_money_addresses.csv', index=False)
        print(f"✅ Smart Money address list saved: dune_smart_money_addresses.csv")
        
        # Save processing report
        with open('dune_processing_report.txt', 'w', encoding='utf-8') as f:
            f.write("Dune Analytics Smart Money Data Processing Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing time: {pd.Timestamp.now()}\n")
            f.write(f"Number of processed CSV files: {len(self.dune_data_summary)}\n")
            f.write(f"Number of extracted Smart Money addresses: {len(self.all_smart_money_addresses)}\n\n")
            
            f.write("File details:\n")
            for summary in self.dune_data_summary:
                f.write(f"- {summary['file_name']}: {summary['rows']} rows\n")
        
        print(f"✅ Processing report saved: dune_processing_report.txt")

def main():
    """
    Main processing function
    """
    print("🎯 Dune Analytics Smart Money Label Generator")
    print("=" * 60)
    
    # 1. Initialize processor
    processor = DuneSmartMoneyProcessor()
    
    # 2. Process all CSV files
    smart_money_addresses = processor.process_all_csv_files()
    
    if not smart_money_addresses:
        print("❌ No Smart Money addresses extracted, please check CSV files")
        return
    
    # 3. Load existing feature data
    features_df = processor.load_existing_wallet_features()
    
    if features_df is None:
        print("❌ Unable to load feature data")
        return
    
    # 4. Create labels
    labels_df = processor.create_smart_money_labels(features_df)
    
    # 5. Generate report
    processor.generate_summary_report()
    
    # 6. Save results
    processor.save_results(labels_df)
    
    print("\n🎉 Processing completed! Generated files:")
    print("  📄 smart_money_labels.csv - Label file for XGBoost training")
    print("  📄 dune_smart_money_addresses.csv - Smart Money address list")
    print("  📄 dune_processing_report.txt - Processing report")
    
    print(f"\n✅ Now you can run the XGBoost model!")
    print("   python smart_money_xgboost.py")

if __name__ == "__main__":
    main()
