import numpy as np
from calculate_modules import *
from test_evaluation_metrics import calculate_minDCF_EER_CLLR_actDCF
import a_dcf

def read_predictions(file_path):
    """Read predictions from the given file path and return true labels and scores."""
    y_true = []
    y_scores = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' - ')
            y_true.append(float(parts[2][-3:]))
            y_scores.append(float(parts[1][-3:]))
    return np.array(y_true), np.array(y_scores)

def process_predictions_for_metrics(file_path):
    """Process predictions and calculate metrics using the calculate_minDCF_EER_CLLR_actDCF function."""
    y_true, y_scores = read_predictions(file_path)
    print(y_scores[:5])
    # Convert y_true to binary labels (0 or 1)
    y_true = np.array([1 if label > 0 else 0 for label in y_true])
    
    # Calculate metrics using the provided function
    minDCF, eer, cllr, actDCF, accuracy, cm  = calculate_minDCF_EER_CLLR_actDCF(y_scores, y_true)
    
    return minDCF, eer, cllr, actDCF, accuracy, cm 

if __name__ == '__main__':
    # file_path = './prediction_per_fold/LA/final_predictions_50_16_0.001_fold-4.txt'  # Replace with your actual file path
    file_path = './best_model/LA/test_predictions_test_only_cv_to_prosa.txt'
    
    minDCF, eer, cllr, actDCF, accuracy, cm = process_predictions_for_metrics(file_path)
    print(f"minDCF: {minDCF}")
    print(f"EER: {eer}")
    print(f"CLLR: {cllr}")
    print(f"actDCF: {actDCF}")
    print(f"accuracy: {accuracy}")
    print(f"cm: {cm}")
