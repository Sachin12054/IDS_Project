"""
Real-time IDS Spark Streaming
Processes network traffic from Kafka and performs real-time intrusion detection
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.streaming import StreamingContext
from pyspark.sql.streaming import *
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPARK_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeIDS:
    def __init__(self, models_dir="../models"):
        """Initialize real-time IDS with Spark Streaming"""
        
        # Initialize Spark with streaming configurations
        self.spark = SparkSession.builder \
            .appName("RealTime_IDS_Streaming") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/spark_checkpoints") \
            .config("spark.sql.streaming.stateStore.providerClass", 
                    "org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark Streaming session initialized")
        
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        
        # Load models
        self.load_models()
        
        # Define schema for incoming data
        self.define_schema()
        
    def define_schema(self):
        """Define schema for incoming network traffic data"""
        self.traffic_schema = StructType([
            StructField("Timestamp", DoubleType(), True),
            StructField("Flow_Duration", DoubleType(), True),
            StructField("Total_Fwd_Packets", IntegerType(), True),
            StructField("Total_Backward_Packets", IntegerType(), True),
            StructField("Total_Length_of_Fwd_Packets", DoubleType(), True),
            StructField("Total_Length_of_Bwd_Packets", DoubleType(), True),
            StructField("Fwd_Packet_Length_Max", DoubleType(), True),
            StructField("Fwd_Packet_Length_Min", DoubleType(), True),
            StructField("Fwd_Packet_Length_Mean", DoubleType(), True),
            StructField("Fwd_Packet_Length_Std", DoubleType(), True),
            StructField("Bwd_Packet_Length_Max", DoubleType(), True),
            StructField("Bwd_Packet_Length_Min", DoubleType(), True),
            StructField("Bwd_Packet_Length_Mean", DoubleType(), True),
            StructField("Bwd_Packet_Length_Std", DoubleType(), True),
            StructField("Flow_Bytes_per_s", DoubleType(), True),
            StructField("Flow_Packets_per_s", DoubleType(), True),
            StructField("Flow_IAT_Mean", DoubleType(), True),
            StructField("Flow_IAT_Std", DoubleType(), True),
            StructField("Flow_IAT_Max", DoubleType(), True),
            StructField("Flow_IAT_Min", DoubleType(), True),
            StructField("Fwd_IAT_Total", DoubleType(), True),
            StructField("Fwd_IAT_Mean", DoubleType(), True),
            StructField("Fwd_IAT_Std", DoubleType(), True),
            StructField("Fwd_IAT_Max", DoubleType(), True),
            StructField("Fwd_IAT_Min", DoubleType(), True),
            StructField("Bwd_IAT_Total", DoubleType(), True),
            StructField("Bwd_IAT_Mean", DoubleType(), True),
            StructField("Bwd_IAT_Std", DoubleType(), True),
            StructField("Bwd_IAT_Max", DoubleType(), True),
            StructField("Bwd_IAT_Min", DoubleType(), True),
            StructField("is_attack", IntegerType(), True),
            StructField("attack_type", StringType(), True)
        ])
    
    def load_models(self):
        """Load pre-trained models for real-time prediction"""
        logger.info("Loading models for real-time inference...")
        
        try:
            # Load Random Forest model
            rf_path = os.path.join(self.models_dir, "random_forest.pkl")
            if os.path.exists(rf_path):
                self.models['random_forest'] = joblib.load(rf_path)
                logger.info("Random Forest model loaded")
            
            # Load XGBoost model (simplified loading for streaming)
            # Note: For production, consider using ONNX or other streaming-friendly formats
            
            # Load scalers
            scaler_path = os.path.join(self.models_dir, "scaler_rf.pkl")
            if os.path.exists(scaler_path):
                self.scalers['rf'] = joblib.load(scaler_path)
                logger.info("Scaler loaded")
                
        except Exception as e:
            logger.warning(f"Some models could not be loaded: {e}")
            # Continue with available models
    
    def create_kafka_stream(self, kafka_servers="localhost:9092", topic="network_traffic"):
        """Create Kafka stream for reading network traffic"""
        logger.info(f"Creating Kafka stream for topic: {topic}")
        
        # Read from Kafka
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse JSON data
        parsed_df = df.select(
            from_json(col("value").cast("string"), self.traffic_schema).alias("data")
        ).select("data.*")
        
        # Add processing timestamp
        parsed_df = parsed_df.withColumn("processing_time", current_timestamp())
        
        logger.info("Kafka stream created successfully")
        return parsed_df
    
    def feature_engineering_stream(self, df):
        """Apply real-time feature engineering"""
        logger.info("Applying real-time feature engineering...")
        
        # Add derived features
        df = df.withColumn("total_packets", col("Total_Fwd_Packets") + col("Total_Backward_Packets"))
        df = df.withColumn("total_bytes", col("Total_Length_of_Fwd_Packets") + col("Total_Length_of_Bwd_Packets"))
        
        # Packet ratios
        df = df.withColumn("fwd_packet_ratio", 
                          when(col("total_packets") > 0, col("Total_Fwd_Packets") / col("total_packets")).otherwise(0))
        df = df.withColumn("bwd_packet_ratio", 
                          when(col("total_packets") > 0, col("Total_Backward_Packets") / col("total_packets")).otherwise(0))
        
        # Byte ratios
        df = df.withColumn("fwd_byte_ratio", 
                          when(col("total_bytes") > 0, col("Total_Length_of_Fwd_Packets") / col("total_bytes")).otherwise(0))
        df = df.withColumn("bwd_byte_ratio", 
                          when(col("total_bytes") > 0, col("Total_Length_of_Bwd_Packets") / col("total_bytes")).otherwise(0))
        
        # Flow rates
        df = df.withColumn("packets_per_second", 
                          when(col("Flow_Duration") > 0, col("total_packets") / (col("Flow_Duration") / 1000000)).otherwise(0))
        df = df.withColumn("bytes_per_second", 
                          when(col("Flow_Duration") > 0, col("total_bytes") / (col("Flow_Duration") / 1000000)).otherwise(0))
        
        return df
    
    def anomaly_detection_rules(self, df):
        """Apply rule-based anomaly detection"""
        logger.info("Applying rule-based anomaly detection...")
        
        # Statistical anomaly detection
        df = df.withColumn("high_packet_rate", 
                          when(col("Flow_Packets_per_s") > 100, 1).otherwise(0))
        df = df.withColumn("high_byte_rate", 
                          when(col("Flow_Bytes_per_s") > 100000, 1).otherwise(0))
        df = df.withColumn("suspicious_duration", 
                          when((col("Flow_Duration") < 1000) | (col("Flow_Duration") > 10000000), 1).otherwise(0))
        
        # Composite anomaly score
        df = df.withColumn("rule_anomaly_score", 
                          col("high_packet_rate") + col("high_byte_rate") + col("suspicious_duration"))
        df = df.withColumn("rule_based_anomaly", 
                          when(col("rule_anomaly_score") > 0, 1).otherwise(0))
        
        return df
    
    def ml_prediction_udf(self):
        """Create UDF for ML model predictions"""
        if 'random_forest' not in self.models:
            return None
        
        # Broadcast model and scaler
        model_broadcast = self.spark.sparkContext.broadcast(self.models['random_forest'])
        scaler_broadcast = self.spark.sparkContext.broadcast(self.scalers.get('rf'))
        
        def predict_intrusion(features_array):
            try:
                model = model_broadcast.value
                scaler = scaler_broadcast.value
                
                # Convert to numpy array
                features = np.array(features_array).reshape(1, -1)
                
                # Scale features if scaler is available
                if scaler is not None:
                    features = scaler.transform(features)
                
                # Make prediction
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0][1]  # Probability of being attack
                
                return {"prediction": int(prediction), "probability": float(probability)}
                
            except Exception as e:
                return {"prediction": 0, "probability": 0.0, "error": str(e)}
        
        return udf(predict_intrusion, MapType(StringType(), StringType()))
    
    def apply_ml_predictions(self, df):
        """Apply ML model predictions to streaming data"""
        logger.info("Applying ML model predictions...")
        
        # Define feature columns for ML prediction
        feature_columns = [
            "Flow_Duration", "Total_Fwd_Packets", "Total_Backward_Packets",
            "Total_Length_of_Fwd_Packets", "Total_Length_of_Bwd_Packets",
            "Flow_Bytes_per_s", "Flow_Packets_per_s", "Flow_IAT_Mean",
            "fwd_packet_ratio", "bwd_packet_ratio", "packets_per_second"
        ]
        
        # Create feature array
        df = df.withColumn("features_array", array([col(c) for c in feature_columns if c in df.columns]))
        
        # Apply ML prediction UDF
        predict_udf = self.ml_prediction_udf()
        if predict_udf:
            df = df.withColumn("ml_prediction", predict_udf(col("features_array")))
            
            # Extract prediction and probability
            df = df.withColumn("ml_attack_prediction", col("ml_prediction")["prediction"].cast("integer"))
            df = df.withColumn("ml_attack_probability", col("ml_prediction")["probability"].cast("double"))
        else:
            # Fallback if no ML model available
            df = df.withColumn("ml_attack_prediction", lit(0))
            df = df.withColumn("ml_attack_probability", lit(0.0))
        
        return df
    
    def final_decision_logic(self, df):
        """Combine rule-based and ML predictions for final decision"""
        logger.info("Applying final decision logic...")
        
        # Combine predictions
        df = df.withColumn("final_prediction", 
                          when((col("rule_based_anomaly") == 1) | 
                               (col("ml_attack_prediction") == 1) |
                               (col("ml_attack_probability") > 0.7), 1).otherwise(0))
        
        # Confidence score
        df = df.withColumn("confidence_score", 
                          (col("rule_anomaly_score") * 0.3 + col("ml_attack_probability") * 0.7))
        
        # Threat level
        df = df.withColumn("threat_level", 
                          when(col("confidence_score") > 0.8, "HIGH")
                          .when(col("confidence_score") > 0.5, "MEDIUM")
                          .when(col("confidence_score") > 0.2, "LOW")
                          .otherwise("BENIGN"))
        
        return df
    
    def create_alerts(self, df):
        """Create alerts for detected threats"""
        # Filter for threats only
        threats_df = df.filter(col("final_prediction") == 1)
        
        # Add alert information
        threats_df = threats_df.withColumn("alert_id", monotonically_increasing_id())
        threats_df = threats_df.withColumn("alert_timestamp", current_timestamp())
        
        # Select relevant columns for alerts
        alert_columns = [
            "alert_id", "alert_timestamp", "Timestamp", "threat_level", 
            "confidence_score", "attack_type", "ml_attack_probability",
            "rule_anomaly_score", "Flow_Packets_per_s", "Flow_Bytes_per_s"
        ]
        
        return threats_df.select([col for col in alert_columns if col in threats_df.columns])
    
    def start_streaming(self, output_mode="append", trigger_interval="10 seconds"):
        """Start the real-time IDS streaming process"""
        logger.info("=== STARTING REAL-TIME IDS STREAMING ===")
        
        # Create Kafka stream
        raw_stream = self.create_kafka_stream()
        
        # Apply processing pipeline
        processed_stream = raw_stream.transform(self.feature_engineering_stream) \
                                   .transform(self.anomaly_detection_rules) \
                                   .transform(self.apply_ml_predictions) \
                                   .transform(self.final_decision_logic)
        
        # Create alerts stream
        alerts_stream = processed_stream.transform(self.create_alerts)
        
        # Output to console (for demonstration)
        console_query = processed_stream.select(
            "Timestamp", "threat_level", "confidence_score", "attack_type",
            "final_prediction", "ml_attack_probability", "rule_anomaly_score"
        ).writeStream \
        .outputMode(output_mode) \
        .format("console") \
        .option("truncate", False) \
        .option("numRows", 20) \
        .trigger(processingTime=trigger_interval) \
        .start()
        
        # Output alerts to console
        alerts_query = alerts_stream.writeStream \
        .outputMode(output_mode) \
        .format("console") \
        .option("truncate", False) \
        .option("numRows", 10) \
        .trigger(processingTime=trigger_interval) \
        .start()
        
        logger.info("Real-time IDS streaming started")
        logger.info("Monitoring network traffic for intrusions...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            # Wait for termination
            console_query.awaitTermination()
            
        except KeyboardInterrupt:
            logger.info("Stopping streaming...")
            console_query.stop()
            alerts_query.stop()
            
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            
        finally:
            self.spark.stop()
            logger.info("=== REAL-TIME IDS STREAMING STOPPED ===")

def main():
    """Main function to run real-time IDS"""
    try:
        # Initialize real-time IDS
        ids = RealTimeIDS()
        
        # Start streaming with configuration
        ids.start_streaming(
            output_mode="append",
            trigger_interval="5 seconds"
        )
        
    except Exception as e:
        logger.error(f"Error in real-time IDS: {str(e)}")
        raise

if __name__ == "__main__":
    main()