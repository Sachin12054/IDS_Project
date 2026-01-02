"""
Advanced Feature Engineering for IDS using PySpark
Creates time-series features and advanced network flow features
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineering:
    def __init__(self):
        """Initialize Spark Session"""
        self.spark = SparkSession.builder \
            .appName("IDS_Feature_Engineering") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("Feature Engineering Spark Session initialized")
    
    def load_processed_data(self, data_path):
        """Load previously cleaned data"""
        logger.info(f"Loading processed data from: {data_path}")
        df = self.spark.read.parquet(data_path)
        logger.info(f"Loaded {df.count()} records with {len(df.columns)} columns")
        return df
    
    def create_temporal_features(self, df):
        """Create time-based features for time-series analysis"""
        logger.info("Creating temporal features...")
        
        # Create timestamp if not exists (using row_number as proxy)
        if 'Timestamp' not in df.columns:
            window_spec = Window.orderBy(monotonically_increasing_id())
            df = df.withColumn("row_id", row_number().over(window_spec))
            df = df.withColumn("Timestamp", 
                              (col("row_id") * 1000).cast("timestamp"))
        
        # Extract time-based features
        df = df.withColumn("hour", hour(col("Timestamp"))) \
               .withColumn("day_of_week", dayofweek(col("Timestamp"))) \
               .withColumn("is_weekend", when(dayofweek(col("Timestamp")).isin([1, 7]), 1).otherwise(0))
        
        return df
    
    def create_flow_features(self, df):
        """Create advanced network flow features"""
        logger.info("Creating advanced flow features...")
        
        # Flow duration and rate features
        if 'Flow_Duration' in df.columns:
            df = df.withColumn("flow_duration_log", log(col("Flow_Duration") + 1))
            df = df.withColumn("packets_per_second", 
                              when(col("Flow_Duration") > 0, 
                                   col("Total_Fwd_Packets") / (col("Flow_Duration") / 1000000)).otherwise(0))
            df = df.withColumn("bytes_per_second", 
                              when(col("Flow_Duration") > 0, 
                                   col("Total_Length_of_Fwd_Packets") / (col("Flow_Duration") / 1000000)).otherwise(0))
        
        # Packet size ratios
        if all(col in df.columns for col in ['Total_Fwd_Packets', 'Total_Backward_Packets']):
            total_packets = col("Total_Fwd_Packets") + col("Total_Backward_Packets")
            df = df.withColumn("fwd_packet_ratio", 
                              when(total_packets > 0, col("Total_Fwd_Packets") / total_packets).otherwise(0))
            df = df.withColumn("bwd_packet_ratio", 
                              when(total_packets > 0, col("Total_Backward_Packets") / total_packets).otherwise(0))
        
        # Byte ratios
        if all(col in df.columns for col in ['Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets']):
            total_bytes = col("Total_Length_of_Fwd_Packets") + col("Total_Length_of_Bwd_Packets")
            df = df.withColumn("fwd_byte_ratio", 
                              when(total_bytes > 0, col("Total_Length_of_Fwd_Packets") / total_bytes).otherwise(0))
            df = df.withColumn("bwd_byte_ratio", 
                              when(total_bytes > 0, col("Total_Length_of_Bwd_Packets") / total_bytes).otherwise(0))
        
        # Packet length statistics
        if all(col in df.columns for col in ['Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min']):
            df = df.withColumn("fwd_packet_length_range", 
                              col("Fwd_Packet_Length_Max") - col("Fwd_Packet_Length_Min"))
        
        return df
    
    def create_statistical_features(self, df):
        """Create statistical aggregation features"""
        logger.info("Creating statistical features...")
        
        # IAT (Inter-Arrival Time) features
        iat_columns = [col_name for col_name in df.columns if 'IAT' in col_name]
        if iat_columns:
            # Create IAT variance and coefficient of variation
            for iat_col in iat_columns:
                if iat_col + '_Mean' in df.columns and iat_col + '_Std' in df.columns:
                    df = df.withColumn(f"{iat_col}_cv", 
                                      when(col(f"{iat_col}_Mean") > 0, 
                                           col(f"{iat_col}_Std") / col(f"{iat_col}_Mean")).otherwise(0))
        
        # Flag-based features
        flag_columns = [col_name for col_name in df.columns if any(flag in col_name.upper() for flag in ['PSH', 'URG', 'FIN', 'SYN', 'RST', 'ACK'])]
        if flag_columns:
            # Create flag combination features
            df = df.withColumn("total_flags", sum([col(flag_col) for flag_col in flag_columns]))
            df = df.withColumn("flag_diversity", 
                              sum([when(col(flag_col) > 0, 1).otherwise(0) for flag_col in flag_columns]))
        
        return df
    
    def create_time_windows(self, df, window_duration="5 seconds"):
        """Create time window aggregations for time-series features"""
        logger.info(f"Creating time windows with duration: {window_duration}")
        
        # Create time windows
        windowed_df = df.withColumn("time_window", 
                                   window(col("Timestamp"), window_duration))
        
        # Aggregate by time window
        time_series_df = windowed_df.groupBy("time_window") \
            .agg(
                count("*").alias("flow_count"),
                sum("is_attack").alias("attack_count"),
                avg("Flow_Duration").alias("avg_flow_duration"),
                avg("Total_Fwd_Packets").alias("avg_fwd_packets"),
                avg("Total_Backward_Packets").alias("avg_bwd_packets"),
                avg("Total_Length_of_Fwd_Packets").alias("avg_fwd_bytes"),
                avg("Total_Length_of_Bwd_Packets").alias("avg_bwd_bytes"),
                max("Flow_Duration").alias("max_flow_duration"),
                stddev("Flow_Duration").alias("std_flow_duration")
            )
        
        # Calculate attack rate
        time_series_df = time_series_df.withColumn("attack_rate", 
                                                  col("attack_count") / col("flow_count"))
        
        # Add temporal features to time series
        time_series_df = time_series_df.withColumn("window_start", col("time_window.start")) \
                                      .withColumn("window_end", col("time_window.end"))
        
        # Sort by time
        time_series_df = time_series_df.orderBy("window_start")
        
        return time_series_df
    
    def create_sequence_features(self, df, sequence_length=10):
        """Create sequence features for LSTM training"""
        logger.info(f"Creating sequence features with length: {sequence_length}")
        
        # Define window for sequence creation
        window_spec = Window.orderBy("window_start")
        
        # Create lag features
        feature_columns = ['flow_count', 'attack_rate', 'avg_flow_duration', 
                          'avg_fwd_packets', 'avg_bwd_packets']
        
        for col_name in feature_columns:
            for lag in range(1, sequence_length + 1):
                df = df.withColumn(f"{col_name}_lag_{lag}", 
                                 lag(col(col_name), lag).over(window_spec))
        
        # Remove rows with null lag features
        df = df.dropna()
        
        return df
    
    def detect_anomalies_statistical(self, df):
        """Simple statistical anomaly detection for labeling"""
        logger.info("Performing statistical anomaly detection...")
        
        # Calculate IQR for key features
        key_features = ['flow_count', 'attack_rate', 'avg_flow_duration']
        
        for feature in key_features:
            if feature in df.columns:
                # Calculate quartiles
                quantiles = df.select(
                    expr(f"percentile_approx({feature}, 0.25)").alias("q1"),
                    expr(f"percentile_approx({feature}, 0.75)").alias("q3")
                ).collect()[0]
                
                q1, q3 = quantiles['q1'], quantiles['q3']
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Mark outliers
                df = df.withColumn(f"{feature}_outlier", 
                                  when((col(feature) < lower_bound) | (col(feature) > upper_bound), 1)
                                  .otherwise(0))
        
        # Create composite anomaly score
        outlier_cols = [col_name for col_name in df.columns if col_name.endswith('_outlier')]
        if outlier_cols:
            df = df.withColumn("anomaly_score", 
                              sum([col(outlier_col) for outlier_col in outlier_cols]))
            df = df.withColumn("is_statistical_anomaly", 
                              when(col("anomaly_score") > 0, 1).otherwise(0))
        
        return df
    
    def save_features(self, df, output_path, feature_type="flow"):
        """Save engineered features"""
        logger.info(f"Saving {feature_type} features to: {output_path}")
        df.coalesce(2).write.mode("overwrite").parquet(output_path)
        logger.info(f"Features saved successfully")
    
    def show_feature_summary(self, df):
        """Display feature summary"""
        logger.info("=== FEATURE ENGINEERING SUMMARY ===")
        print(f"Total Records: {df.count()}")
        print(f"Total Features: {len(df.columns)}")
        
        # Show new features
        new_features = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['ratio', 'rate', 'cv', 'diversity', 'range', 'lag', 'outlier', 'anomaly'])]
        print(f"\nNew Features Created ({len(new_features)}):")
        for feature in new_features[:20]:  # Show first 20
            print(f"  - {feature}")
        
        if len(new_features) > 20:
            print(f"  ... and {len(new_features) - 20} more")
        
        # Show sample data
        print("\n=== SAMPLE ENGINEERED DATA ===")
        df.select(new_features[:10]).show(5)

def main():
    """Main execution function"""
    try:
        # Initialize feature engineering
        fe = AdvancedFeatureEngineering()
        
        # Define paths
        processed_data_path = "../data/processed/cleaned_features.parquet"
        time_series_path = "../data/time_series"
        
        # Create output directory
        os.makedirs(time_series_path, exist_ok=True)
        
        # Load processed data
        logger.info("=== LOADING PROCESSED DATA ===")
        df = fe.load_processed_data(processed_data_path)
        
        # Step 1: Create temporal features
        logger.info("=== STEP 1: TEMPORAL FEATURES ===")
        df = fe.create_temporal_features(df)
        
        # Step 2: Create flow features
        logger.info("=== STEP 2: FLOW FEATURES ===")
        df = fe.create_flow_features(df)
        
        # Step 3: Create statistical features
        logger.info("=== STEP 3: STATISTICAL FEATURES ===")
        df = fe.create_statistical_features(df)
        
        # Save flow-level features
        fe.save_features(df, os.path.join("../data/processed", "engineered_features.parquet"), "flow")
        
        # Step 4: Create time windows for time-series
        logger.info("=== STEP 4: TIME SERIES WINDOWS ===")
        ts_df = fe.create_time_windows(df)
        
        # Step 5: Create sequence features
        logger.info("=== STEP 5: SEQUENCE FEATURES ===")
        ts_df = fe.create_sequence_features(ts_df)
        
        # Step 6: Statistical anomaly detection
        logger.info("=== STEP 6: ANOMALY DETECTION ===")
        ts_df = fe.detect_anomalies_statistical(ts_df)
        
        # Save time-series features
        fe.save_features(ts_df, os.path.join(time_series_path, "time_series_features.parquet"), "time_series")
        
        # Show summary
        logger.info("=== FEATURE SUMMARY ===")
        fe.show_feature_summary(df)
        
        logger.info("=== FEATURE ENGINEERING COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise
    finally:
        fe.spark.stop()

if __name__ == "__main__":
    main()