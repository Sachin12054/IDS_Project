"""
Big Data Preprocessing Pipeline using Apache Spark
Handles CSE-CIC-IDS2018 Dataset - Merging, Cleaning, and Preprocessing
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparkIDSPreprocessor:
    def __init__(self, app_name="IDS_BigData_Preprocessing"):
        """Initialize Spark Session with optimized configurations"""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        # Suppress INFO logs for cleaner output
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark Session initialized successfully")
    
    def load_csv_files(self, data_path):
        """Load and merge all CSV files from the raw_csv directory"""
        try:
            csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
            logger.info(f"Found {len(csv_files)} CSV files")
            
            if not csv_files:
                raise ValueError("No CSV files found in the specified directory")
            
            # Read all CSV files and union them
            dfs = []
            for csv_file in csv_files:
                file_path = os.path.join(data_path, csv_file)
                logger.info(f"Loading: {csv_file}")
                
                df = self.spark.read.csv(file_path, header=True, inferSchema=True)
                df = df.withColumn("source_file", lit(csv_file))
                dfs.append(df)
            
            # Union all DataFrames
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = merged_df.unionAll(df)
            
            logger.info(f"Merged dataset shape: {merged_df.count()} rows, {len(merged_df.columns)} columns")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error loading CSV files: {str(e)}")
            raise
    
    def clean_data(self, df):
        """Clean the dataset by removing nulls, infinities, and duplicates"""
        logger.info("Starting data cleaning process...")
        
        # Get initial count
        initial_count = df.count()
        logger.info(f"Initial record count: {initial_count}")
        
        # Remove rows with null values in critical columns
        df_clean = df.dropna()
        after_null = df_clean.count()
        logger.info(f"After removing nulls: {after_null} ({initial_count - after_null} removed)")
        
        # Replace infinite values with None and drop those rows
        numeric_columns = [field.name for field in df_clean.schema.fields 
                          if field.dataType in [IntegerType(), LongType(), FloatType(), DoubleType()]]
        
        for col_name in numeric_columns:
            df_clean = df_clean.withColumn(col_name, 
                when(col(col_name).isNull() | isnan(col(col_name)) | 
                     isinf(col(col_name)), None).otherwise(col(col_name)))
        
        df_clean = df_clean.dropna()
        after_inf = df_clean.count()
        logger.info(f"After removing infinities: {after_inf} ({after_null - after_inf} removed)")
        
        # Remove duplicates
        df_clean = df_clean.dropDuplicates()
        final_count = df_clean.count()
        logger.info(f"After removing duplicates: {final_count} ({after_inf - final_count} removed)")
        
        # Clean column names (remove spaces and special characters)
        for old_col in df_clean.columns:
            new_col = old_col.strip().replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '')
            df_clean = df_clean.withColumnRenamed(old_col, new_col)
        
        logger.info("Data cleaning completed successfully")
        return df_clean
    
    def encode_labels(self, df, label_column="Label"):
        """Encode string labels to numeric values"""
        logger.info("Encoding labels...")
        
        # Show label distribution
        label_counts = df.groupBy(label_column).count().orderBy(desc("count"))
        logger.info("Label distribution:")
        label_counts.show()
        
        # Create binary labels (0: Benign, 1: Attack)
        df = df.withColumn("is_attack", 
                          when(col(label_column) == "Benign", 0).otherwise(1))
        
        # Encode specific attack types
        indexer = StringIndexer(inputCol=label_column, outputCol="attack_type_encoded")
        df = indexer.fit(df).transform(df)
        
        return df
    
    def feature_engineering(self, df):
        """Create additional features and prepare for ML"""
        logger.info("Performing feature engineering...")
        
        # Remove non-feature columns
        exclude_cols = ['Label', 'source_file', 'Timestamp']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle any remaining string columns
        string_columns = [field.name for field in df.schema.fields 
                         if field.dataType == StringType() and field.name in feature_cols]
        
        for col_name in string_columns:
            indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_encoded")
            df = indexer.fit(df).transform(df)
            feature_cols.remove(col_name)
            feature_cols.append(f"{col_name}_encoded")
        
        # Create feature vector
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
        df = assembler.transform(df)
        
        # Scale features
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", 
                               withStd=True, withMean=True)
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)
        
        logger.info("Feature engineering completed")
        return df, feature_cols
    
    def save_processed_data(self, df, output_path):
        """Save processed data in Parquet format"""
        logger.info(f"Saving processed data to: {output_path}")
        
        # Coalesce to reduce number of output files
        df.coalesce(4).write.mode("overwrite").parquet(output_path)
        logger.info("Data saved successfully")
    
    def show_data_summary(self, df):
        """Display data summary and statistics"""
        logger.info("=== DATA SUMMARY ===")
        print(f"Total Records: {df.count()}")
        print(f"Total Features: {len(df.columns)}")
        
        # Show schema
        print("\n=== SCHEMA ===")
        df.printSchema()
        
        # Show sample data
        print("\n=== SAMPLE DATA ===")
        df.show(5, truncate=False)
        
        # Show label distribution
        if 'is_attack' in df.columns:
            print("\n=== LABEL DISTRIBUTION ===")
            df.groupBy('is_attack').count().show()
        
        if 'Label' in df.columns:
            print("\n=== ATTACK TYPES ===")
            df.groupBy('Label').count().orderBy(desc('count')).show()
    
    def stop_spark(self):
        """Stop Spark session"""
        self.spark.stop()
        logger.info("Spark session stopped")

def main():
    """Main execution function"""
    try:
        # Initialize processor
        processor = SparkIDSPreprocessor()
        
        # Define paths
        raw_data_path = "../data/raw_csv"
        processed_data_path = "../data/processed"
        
        # Create processed directory if it doesn't exist
        os.makedirs(processed_data_path, exist_ok=True)
        
        # Step 1: Load and merge CSV files
        logger.info("=== STEP 1: LOADING DATA ===")
        df = processor.load_csv_files(raw_data_path)
        
        # Step 2: Clean data
        logger.info("=== STEP 2: CLEANING DATA ===")
        df_clean = processor.clean_data(df)
        
        # Step 3: Encode labels
        logger.info("=== STEP 3: ENCODING LABELS ===")
        df_encoded = processor.encode_labels(df_clean)
        
        # Step 4: Feature engineering
        logger.info("=== STEP 4: FEATURE ENGINEERING ===")
        df_final, feature_columns = processor.feature_engineering(df_encoded)
        
        # Step 5: Save processed data
        logger.info("=== STEP 5: SAVING DATA ===")
        processor.save_processed_data(df_final, 
                                    os.path.join(processed_data_path, "cleaned_features.parquet"))
        
        # Show summary
        logger.info("=== STEP 6: DATA SUMMARY ===")
        processor.show_data_summary(df_final)
        
        # Save feature column names for later use
        with open(os.path.join(processed_data_path, "feature_columns.txt"), 'w') as f:
            f.write('\n'.join(feature_columns))
        
        logger.info("=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        processor.stop_spark()

if __name__ == "__main__":
    main()