import argparse
import numpy as np
from calculate_modules import *
from calculate_metrics import calculate_minDCF_EER_CLLR_actDCF
import a_dcf
from datetime import datetime

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
    # Convert y_true to binary labels (0 or 1)
    y_true = np.array([1 if label > 0 else 0 for label in y_true])
    
    # Calculate metrics using the function from calculate_metrics module
    minDCF, eer, cllr, actDCF, accuracy, cm  = calculate_minDCF_EER_CLLR_actDCF(y_scores, y_true)
    
    return minDCF, eer, cllr, actDCF, accuracy, cm 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # set files to scores and the scenario type
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--scores', type=str, required=True)
    parser.add_argument('--output_filename', type=str, default="result_%s" % datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), help="Output file path to write results (without extension)")

    args = parser.parse_args()

    minDCF, eer, cllr, actDCF, accuracy, cm = process_predictions_for_metrics(args.scores)

    output_file_name = ("./results/%s/%s.txt" % (args.scenario, args.output_filename))

    # Write the results to a file
    with open(output_file_name, 'w') as f:
        f.write('\nCM SYSTEM\n')
        f.write('\tmin DCF \t\t= {} '
                    '(min DCF for countermeasure)\n'.format(
                        minDCF))
        f.write('\tEER\t\t= {:8.9f} % '
                    '(EER for countermeasure)\n'.format(
                        eer * 100))
        f.write('\tCLLR\t\t= {:8.9f} bits '
                '(CLLR for countermeasure)\n'.format(
                    cllr))
        f.write('\tactDCF\t\t= {:} '
                '(actual DCF)\n'.format(
                    actDCF))
        f.write('\taccuracy\t\t= {:} '
                '(accuracy)\n'.format(
                    accuracy))
        f.write('\tConfusion Matrix\t\t= ')
        f.write("\t" + str(cm))

