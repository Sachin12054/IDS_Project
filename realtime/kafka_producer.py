"""
Real-time IDS Kafka Producer
Simulates network traffic data for streaming processing
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from kafka import KafkaProducer
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ATTACK_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkTrafficProducer:
    def __init__(self, kafka_config=None):
        """Initialize Kafka producer for network traffic simulation"""
        
        # Default Kafka configuration
        default_config = {
            'bootstrap_servers': ['localhost:9092'],
            'value_serializer': lambda x: json.dumps(x).encode('utf-8'),
            'key_serializer': lambda x: x.encode('utf-8') if x else None
        }
        
        config = kafka_config or default_config
        
        try:
            self.producer = KafkaProducer(**config)
            logger.info("Kafka producer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
        
        self.topic = "network_traffic"
        self.is_running = False
        
        # Load sample data for simulation
        self.sample_data = None
        self.load_sample_data()
    
    def load_sample_data(self):
        """Load sample data for traffic simulation"""
        try:
            # Try to load processed data
            data_path = "../data/processed/engineered_features.parquet"
            if os.path.exists(data_path):
                self.sample_data = pd.read_parquet(data_path)
                logger.info(f"Loaded {len(self.sample_data)} sample records")
            else:
                # Generate synthetic data if no real data available
                logger.warning("No processed data found. Generating synthetic data...")
                self.sample_data = self.generate_synthetic_data()
                
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            self.sample_data = self.generate_synthetic_data()
    
    def generate_synthetic_data(self, num_records=1000):
        """Generate synthetic network traffic data"""
        logger.info("Generating synthetic network traffic data...")
        
        np.random.seed(42)
        
        # Generate synthetic features
        data = {
            'Flow_Duration': np.random.exponential(1000000, num_records),
            'Total_Fwd_Packets': np.random.poisson(10, num_records),
            'Total_Backward_Packets': np.random.poisson(8, num_records),
            'Total_Length_of_Fwd_Packets': np.random.exponential(1500, num_records),
            'Total_Length_of_Bwd_Packets': np.random.exponential(1200, num_records),
            'Fwd_Packet_Length_Max': np.random.exponential(1500, num_records),
            'Fwd_Packet_Length_Min': np.random.exponential(64, num_records),
            'Fwd_Packet_Length_Mean': np.random.exponential(800, num_records),
            'Fwd_Packet_Length_Std': np.random.exponential(200, num_records),
            'Bwd_Packet_Length_Max': np.random.exponential(1200, num_records),
            'Bwd_Packet_Length_Min': np.random.exponential(64, num_records),
            'Bwd_Packet_Length_Mean': np.random.exponential(600, num_records),
            'Bwd_Packet_Length_Std': np.random.exponential(150, num_records),
            'Flow_Bytes_per_s': np.random.exponential(50000, num_records),
            'Flow_Packets_per_s': np.random.exponential(100, num_records),
            'Flow_IAT_Mean': np.random.exponential(10000, num_records),
            'Flow_IAT_Std': np.random.exponential(5000, num_records),
            'Flow_IAT_Max': np.random.exponential(50000, num_records),
            'Flow_IAT_Min': np.random.exponential(1000, num_records),
            'Fwd_IAT_Total': np.random.exponential(100000, num_records),
            'Fwd_IAT_Mean': np.random.exponential(15000, num_records),
            'Fwd_IAT_Std': np.random.exponential(8000, num_records),
            'Fwd_IAT_Max': np.random.exponential(80000, num_records),
            'Fwd_IAT_Min': np.random.exponential(2000, num_records),
            'Bwd_IAT_Total': np.random.exponential(80000, num_records),
            'Bwd_IAT_Mean': np.random.exponential(12000, num_records),
            'Bwd_IAT_Std': np.random.exponential(6000, num_records),
            'Bwd_IAT_Max': np.random.exponential(60000, num_records),
            'Bwd_IAT_Min': np.random.exponential(1500, num_records),
        }
        
        # Add attack labels (90% benign, 10% attacks)
        attack_probability = 0.1
        is_attack = np.random.choice([0, 1], num_records, p=[1-attack_probability, attack_probability])
        data['is_attack'] = is_attack
        
        # Add timestamp
        base_time = time.time()
        data['Timestamp'] = [base_time + i for i in range(num_records)]
        
        return pd.DataFrame(data)
    
    def add_noise_and_variation(self, record):
        """Add realistic noise and variation to data"""
        # Add small random variations to numeric fields
        for key, value in record.items():
            if isinstance(value, (int, float)) and key != 'is_attack' and key != 'Timestamp':
                # Add 5% noise
                noise_factor = 1 + np.random.normal(0, 0.05)
                record[key] = max(0, value * noise_factor)
        
        return record
    
    def simulate_attack_patterns(self, record):
        """Simulate different attack patterns"""
        if record['is_attack'] == 1:
            attack_type = np.random.choice(['dos', 'ddos', 'port_scan', 'brute_force'])
            
            if attack_type == 'dos':
                # DoS: High packet rate, short duration
                record['Flow_Packets_per_s'] *= 10
                record['Flow_Duration'] = min(record['Flow_Duration'], 100000)
            
            elif attack_type == 'ddos':
                # DDoS: Very high packet rate, multiple sources
                record['Flow_Packets_per_s'] *= 20
                record['Total_Fwd_Packets'] *= 5
            
            elif attack_type == 'port_scan':
                # Port Scan: Many small packets
                record['Total_Fwd_Packets'] *= 3
                record['Fwd_Packet_Length_Mean'] = min(record['Fwd_Packet_Length_Mean'], 100)
                
            elif attack_type == 'brute_force':
                # Brute Force: Regular intervals, specific patterns
                record['Flow_IAT_Std'] = max(record['Flow_IAT_Std'] * 0.1, 100)  # More regular intervals
            
            record['attack_type'] = attack_type
        else:
            record['attack_type'] = 'benign'
        
        return record
    
    def create_traffic_record(self):
        """Create a single network traffic record"""
        if self.sample_data is not None and len(self.sample_data) > 0:
            # Select random sample and modify it
            sample_idx = np.random.randint(0, len(self.sample_data))
            record = self.sample_data.iloc[sample_idx].to_dict()
            
            # Update timestamp
            record['Timestamp'] = time.time()
            
            # Add noise and variations
            record = self.add_noise_and_variation(record)
            
            # Apply attack patterns
            record = self.simulate_attack_patterns(record)
            
        else:
            # Fallback to basic synthetic record
            record = {
                'Timestamp': time.time(),
                'Flow_Duration': np.random.exponential(100000),
                'Total_Fwd_Packets': np.random.poisson(5),
                'Total_Backward_Packets': np.random.poisson(3),
                'Flow_Bytes_per_s': np.random.exponential(10000),
                'Flow_Packets_per_s': np.random.exponential(50),
                'is_attack': np.random.choice([0, 1], p=[0.9, 0.1]),
                'attack_type': 'synthetic'
            }
        
        return record
    
    def send_record(self, record, key=None):
        """Send a single record to Kafka"""
        try:
            # Convert numpy types to native Python types for JSON serialization
            serializable_record = {}
            for k, v in record.items():
                if isinstance(v, (np.integer, np.floating)):
                    serializable_record[k] = v.item()
                elif isinstance(v, np.ndarray):
                    serializable_record[k] = v.tolist()
                else:
                    serializable_record[k] = v
            
            future = self.producer.send(self.topic, value=serializable_record, key=key)
            future.get(timeout=10)  # Wait for confirmation
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send record: {e}")
            return False
    
    def start_streaming(self, rate_per_second=10, duration_seconds=None):
        """Start streaming network traffic data"""
        logger.info(f"Starting traffic stream at {rate_per_second} records/second")
        
        self.is_running = True
        sleep_interval = 1.0 / rate_per_second
        
        start_time = time.time()
        records_sent = 0
        
        try:
            while self.is_running:
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    logger.info(f"Duration limit reached. Stopping after {duration_seconds} seconds")
                    break
                
                # Create and send record
                record = self.create_traffic_record()
                success = self.send_record(record, key=f"flow_{records_sent}")
                
                if success:
                    records_sent += 1
                    
                    # Log progress periodically
                    if records_sent % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = records_sent / elapsed
                        attack_count = sum(1 for i in range(max(0, records_sent-100), records_sent) 
                                         if self.create_traffic_record()['is_attack'] == 1)
                        logger.info(f"Sent {records_sent} records, Rate: {rate:.2f}/sec, Recent attacks: {attack_count}%")
                
                # Sleep to maintain rate
                time.sleep(sleep_interval)
                
        except KeyboardInterrupt:
            logger.info("Streaming stopped by user")
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
        finally:
            self.stop_streaming()
        
        logger.info(f"Streaming completed. Total records sent: {records_sent}")
    
    def stop_streaming(self):
        """Stop the streaming process"""
        self.is_running = False
        if hasattr(self, 'producer'):
            self.producer.flush()
            self.producer.close()
        logger.info("Traffic streaming stopped")

def main():
    """Main function to run the producer"""
    try:
        # Initialize producer
        producer = NetworkTrafficProducer()
        
        # Configuration
        rate_per_second = 5  # 5 records per second
        duration_minutes = 10  # Run for 10 minutes
        
        logger.info("=== STARTING REAL-TIME TRAFFIC PRODUCER ===")
        logger.info(f"Topic: {producer.topic}")
        logger.info(f"Rate: {rate_per_second} records/second")
        logger.info(f"Duration: {duration_minutes} minutes")
        logger.info("Press Ctrl+C to stop streaming")
        
        # Start streaming
        producer.start_streaming(
            rate_per_second=rate_per_second,
            duration_seconds=duration_minutes * 60
        )
        
        logger.info("=== TRAFFIC PRODUCER COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Error in traffic producer: {str(e)}")
        raise

if __name__ == "__main__":
    main()