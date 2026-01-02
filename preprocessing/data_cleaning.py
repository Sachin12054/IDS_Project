"""
Data Preprocessing Pipeline for Time Series IDS
Handles CSE-CIC-IDS2018 Dataset - Cleaning and Preprocessing without Spark
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_CSV_DIR, PROCESSED_DIR, CHUNK_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        """Initialize data preprocessor"""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        
        # Ensure output directories exist
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        logger.info("Data Preprocessor initialized")
    
    def load_csv_files(self, data_path=RAW_CSV_DIR, sample_fraction=0.1):
        """Load CSV files with memory-efficient streaming approach"""
        try:
            csv_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])
            logger.info(f"Found {len(csv_files)} CSV files")
            
            if not csv_files:
                raise ValueError("No CSV files found in the specified directory")
            
            # Process files one at a time with sampling to reduce memory
            processed_chunks = []
            total_rows = 0
            
            for csv_file in csv_files:
                file_path = os.path.join(data_path, csv_file)
                logger.info(f"Processing: {csv_file}")
                
                # Read file in chunks and sample to reduce memory
                file_chunks = []
                for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=True):
                    # Remove header rows that might be repeated in data
                    chunk = chunk[chunk['Label'] != 'Label']
                    
                    # Sample to reduce data size
                    if len(chunk) > 0:
                        sampled_chunk = chunk.sample(frac=sample_fraction, random_state=42)
                        sampled_chunk['source_file'] = csv_file
                        file_chunks.append(sampled_chunk)
                        total_rows += len(sampled_chunk)
                
                # Concatenate chunks from this file only
                if file_chunks:
                    file_df = pd.concat(file_chunks, ignore_index=True)
                    processed_chunks.append(file_df)
                    logger.info(f"  - Loaded {len(file_df)} rows from {csv_file}")
                    
                    # Clear memory
                    del file_chunks
                    import gc
                    gc.collect()
            
            # Concatenate all processed files
            logger.info("Merging all processed files...")
            merged_df = pd.concat(processed_chunks, ignore_index=True)
            
            # Clear memory
            del processed_chunks
            import gc
            gc.collect()
            
            logger.info(f"Merged dataset shape: {merged_df.shape}")
            logger.info(f"Total rows loaded: {total_rows}")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error loading CSV files: {str(e)}")
            raise
    
    def clean_column_names(self, df):
        """Clean column names by removing spaces and special characters"""
        logger.info("Cleaning column names...")
        
        new_columns = {}
        for col in df.columns:
            new_col = col.strip().replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '').replace('/', '_')
            new_columns[col] = new_col
        
        # Rename in-place to avoid memory copy
        df.rename(columns=new_columns, inplace=True)
        logger.info(f"Column names cleaned. Total columns: {len(df.columns)}")
        
        return df
    
    def convert_to_numeric(self, df):
        """Convert all columns to proper numeric types where possible"""
        logger.info("Converting columns to numeric types...")
        
        # Exclude known string columns
        string_columns = ['Label', 'source_file', 'Timestamp']
        
        converted_count = 0
        for col in df.columns:
            if col not in string_columns and df[col].dtype == 'object':
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                converted_count += 1
        
        logger.info(f"Converted {converted_count} columns to numeric types")
        
        # Count numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Total numeric columns: {len(numeric_cols)}")
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # Get initial missing value count
        initial_missing = df.isnull().sum().sum()
        logger.info(f"Initial missing values: {initial_missing}")
        
        # Separate numeric and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Remove source_file from processing
        if 'source_file' in categorical_columns:
            categorical_columns.remove('source_file')
        
        # Handle numeric columns with median imputation (in place)
        if numeric_columns:
            # Process in batches to reduce memory
            batch_size = 20
            for i in range(0, len(numeric_columns), batch_size):
                batch_cols = numeric_columns[i:i+batch_size]
                df[batch_cols] = df[batch_cols].fillna(df[batch_cols].median())
        
        # Handle categorical columns with mode imputation
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col] = df[col].fillna(mode_value)
        
        # Final missing value count
        final_missing = df.isnull().sum().sum()
        logger.info(f"Final missing values: {final_missing}")
        
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        logger.info("Removing duplicates...")
        
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        final_count = len(df)
        
        logger.info(f"Removed {initial_count - final_count} duplicate rows")
        
        return df
    
    def handle_infinite_values(self, df):
        """Handle infinite values in numeric columns"""
        logger.info("Handling infinite values...")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Replace infinite values with NaN
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column median
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        logger.info("Infinite values handled")
        
        return df
    
    def encode_labels(self, df):
        """Encode categorical labels"""
        logger.info("Encoding labels...")
        
        # Find label column
        label_column = None
        possible_labels = ['Label', 'label', 'attack', 'Attack', 'class', 'Class']
        
        for col in possible_labels:
            if col in df.columns:
                label_column = col
                break
        
        if label_column:
            logger.info(f"Found label column: {label_column}")
            
            # Check unique values
            unique_labels = df[label_column].unique()
            logger.info(f"Unique labels: {unique_labels}")
            
            # If labels are strings, encode them
            if df[label_column].dtype == 'object':
                df[f'{label_column}_encoded'] = self.label_encoder.fit_transform(df[label_column])
                logger.info(f"Label encoding completed. Classes: {self.label_encoder.classes_}")
            
            # Create binary attack indicator
            benign_labels = ['BENIGN', 'benign', 'normal', 'Normal', '0', 'Benign']
            df['is_attack'] = (~df[label_column].isin(benign_labels)).astype(int)
            
        else:
            logger.warning("No label column found")
            # Create dummy labels
            df['is_attack'] = 0
        
        return df
    
    def feature_selection(self, df, variance_threshold=0.0001):
        """Remove low variance features"""
        logger.info("Performing feature selection...")
        
        # Get numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove label columns from feature selection
        exclude_columns = ['is_attack', 'Label_encoded', 'source_file', 'Label']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        logger.info(f"Total numeric feature columns: {len(feature_columns)}")
        
        # Calculate variance for each feature
        variances = df[feature_columns].var()
        
        # Select features with variance above threshold
        selected_features = variances[variances > variance_threshold].index.tolist()
        
        logger.info(f"Selected {len(selected_features)} features out of {len(feature_columns)}")
        
        # Keep selected features plus required columns (avoid duplicates)
        keep_columns = list(set(selected_features))
        
        # Add metadata columns if they exist
        for col in ['is_attack', 'Label_encoded', 'source_file', 'Label']:
            if col in df.columns and col not in keep_columns:
                keep_columns.append(col)
        
        logger.info(f"Final columns to keep: {len(keep_columns)}")
        
        return df[keep_columns]
    
    def scale_features(self, df):
        """Scale numeric features"""
        logger.info("Scaling features...")
        
        # Get only numeric columns, exclude labels and metadata
        exclude_columns = ['is_attack', 'Label_encoded', 'source_file', 'Label']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_cols if col not in exclude_columns]
        
        logger.info(f"Scaling {len(feature_columns)} numeric features")
        
        # Scale features
        if feature_columns:
            df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        logger.info("Feature scaling completed")
        
        return df
    
    def save_processed_data(self, df, filename="cleaned_features.parquet"):
        """Save processed data"""
        output_path = os.path.join(PROCESSED_DIR, filename)
        logger.info(f"Saving processed data to: {output_path}")
        
        # Save as parquet for efficient storage
        df.to_parquet(output_path, index=False)
        
        # Also save feature names and scaler
        import joblib
        joblib.dump(self.scaler, os.path.join(PROCESSED_DIR, "feature_scaler.pkl"))
        
        if hasattr(self.label_encoder, 'classes_'):
            joblib.dump(self.label_encoder, os.path.join(PROCESSED_DIR, "label_encoder.pkl"))
        
        logger.info(f"Processed data saved. Shape: {df.shape}")
        
        return output_path
    
    def get_data_summary(self, df):
        """Generate data summary"""
        logger.info("=== DATA SUMMARY ===")
        print(f"Dataset Shape: {df.shape}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Attack distribution
        if 'is_attack' in df.columns:
            attack_dist = df['is_attack'].value_counts()
            print(f"\nAttack Distribution:")
            print(f"  Benign: {attack_dist.get(0, 0)} ({attack_dist.get(0, 0)/len(df)*100:.2f}%)")
            print(f"  Attack: {attack_dist.get(1, 0)} ({attack_dist.get(1, 0)/len(df)*100:.2f}%)")
        
        # Data types
        print(f"\nData Types:")
        print(df.dtypes.value_counts())
        
        # Missing values
        missing_values = df.isnull().sum().sum()
        print(f"\nMissing Values: {missing_values}")
        
        return df.describe()

def main():
    """Main preprocessing pipeline"""
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        logger.info("=== STARTING DATA PREPROCESSING PIPELINE ===")
        
        # Step 1: Load data
        logger.info("=== STEP 1: LOADING DATA ===")
        df = preprocessor.load_csv_files()
        
        # Step 2: Clean column names
        logger.info("=== STEP 2: CLEANING COLUMN NAMES ===")
        df = preprocessor.clean_column_names(df)
        
        # Step 2.5: Convert columns to numeric
        logger.info("=== STEP 2.5: CONVERTING TO NUMERIC ===")
        df = preprocessor.convert_to_numeric(df)
        
        # Step 3: Handle missing values
        logger.info("=== STEP 3: HANDLING MISSING VALUES ===")
        df = preprocessor.handle_missing_values(df)
        
        # Step 4: Remove duplicates
        logger.info("=== STEP 4: REMOVING DUPLICATES ===")
        df = preprocessor.remove_duplicates(df)
        
        # Step 5: Handle infinite values
        logger.info("=== STEP 5: HANDLING INFINITE VALUES ===")
        df = preprocessor.handle_infinite_values(df)
        
        # Step 6: Encode labels
        logger.info("=== STEP 6: ENCODING LABELS ===")
        df = preprocessor.encode_labels(df)
        
        # Step 7: Feature selection
        logger.info("=== STEP 7: FEATURE SELECTION ===")
        df = preprocessor.feature_selection(df)
        
        # Step 8: Scale features
        logger.info("=== STEP 8: SCALING FEATURES ===")
        df = preprocessor.scale_features(df)
        
        # Step 9: Save processed data
        logger.info("=== STEP 9: SAVING PROCESSED DATA ===")
        output_path = preprocessor.save_processed_data(df)
        
        # Step 10: Generate summary
        logger.info("=== STEP 10: DATA SUMMARY ===")
        summary = preprocessor.get_data_summary(df)
        
        logger.info("=== DATA PREPROCESSING COMPLETED SUCCESSFULLY ===")
        logger.info(f"Processed data saved to: {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()