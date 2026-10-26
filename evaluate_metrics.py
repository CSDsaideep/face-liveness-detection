import numpy as np

def calculate_biometric_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Computes professional presentation attack detection metrics.
    APCER: Attack Presentation Classification Error Rate (Rate of spoofs let in)
    BPCER: Bona Fide Presentation Classification Error Rate (Rate of real faces blocked)
    """
    # Convert probabilities to hard binary classifications
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Isolate attack (spoof) and bona fide (real) indices
    attacks = (y_true == 0)
    bona_fides = (y_true == 1)
    
    total_attacks = np.sum(attacks)
    total_bona_fides = np.sum(bona_fides)
    
    # APCER: Misclassified spoofs divided by total spoofs
    false_accepts = np.sum((y_pred == 1) & attacks)
    apcer = (false_accepts / total_attacks) * 100 if total_attacks > 0 else 0
    
    # BPCER: Misclassified real faces divided by total real faces
    false_rejects = np.sum((y_pred == 0) & bona_fides)
    bpcer = (false_rejects / total_bona_fides) * 100 if total_bona_fides > 0 else 0
    
    return apcer, bpcer

if __name__ == "__main__":
    print("Running biometric performance validation dashboard...")
    
    # Create mock ground truth labels and model predictions to demonstrate verification
    np.random.seed(24)
    mock_true_labels = np.random.choice([0, 1], size=200, p=[0.4, 0.6]) # 40% spoofs, 60% real
    mock_pred_probabilities = np.random.uniform(0.0, 1.0, size=200)
    
    # Let's assume the model performs well (high scores for 1s, low scores for 0s)
    mock_pred_probabilities[mock_true_labels == 1] += 0.25
    mock_pred_probabilities[mock_true_labels == 0] -= 0.25
    mock_pred_probabilities = np.clip(mock_pred_probabilities, 0.0, 1.0)
    
    apcer, bpcer = calculate_biometric_metrics(mock_true_labels, mock_pred_probabilities, threshold=0.5)
    
    print("\n--- Biometric Metric Validation Execution Report ---")
    print(f"Target Classification Threshold: {0.5}")
    print(f"APCER (Attack Presentation Classification Error Rate): {apcer:.2f}%")
    print(f"BPCER (Bona Fide Presentation Classification Error Rate): {bpcer:.2f}%")
    print("-----------------------------------------------------")
