"""
Test the IDS model with threat and non-threat samples
"""
import pandas as pd
import numpy as np
import joblib

print("=" * 60)
print("TESTING IDS WITH THREAT & NON-THREAT SAMPLES")
print("=" * 60)

# Load model
model = joblib.load('models/random_forest.pkl')
scaler = joblib.load('models/scaler_rf.pkl')

def test_file(filepath, expected_label):
    """Test a CSV file and show results"""
    print(f"\n📂 Testing: {filepath}")
    print("-" * 50)
    
    df = pd.read_csv(filepath)
    
    # Prepare features
    X = df.fillna(0).replace([np.inf, -np.inf], 0)
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Results
    attacks_detected = predictions.sum()
    benign_detected = len(predictions) - attacks_detected
    
    print(f"   Total samples: {len(df)}")
    print(f"   🔴 Detected as ATTACK: {attacks_detected}")
    print(f"   🟢 Detected as BENIGN: {benign_detected}")
    print(f"   Expected: {expected_label}")
    
    if expected_label == "ATTACK":
        accuracy = attacks_detected / len(predictions) * 100
        print(f"   ✓ Detection Rate: {accuracy:.1f}%")
    else:
        accuracy = benign_detected / len(predictions) * 100
        print(f"   ✓ Correct Classification: {accuracy:.1f}%")
    
    # Show sample predictions
    print(f"\n   Sample predictions (first 5):")
    for i in range(min(5, len(predictions))):
        pred = "ATTACK" if predictions[i] == 1 else "BENIGN"
        prob = probabilities[i]
        print(f"      Row {i+1}: {pred} (confidence: {max(prob, 1-prob)*100:.1f}%)")

# Test threat file
test_file("test_samples/threat_traffic.csv", "ATTACK")

# Test non-threat file
test_file("test_samples/non_threat_traffic.csv", "BENIGN")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
✓ threat_traffic.csv     → Should be detected as ATTACK
✓ non_threat_traffic.csv → Should be detected as BENIGN

If detection rate is high (>90%), the model is working correctly!
""")
