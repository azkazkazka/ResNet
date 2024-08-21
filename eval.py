import os
import glob
import librosa
import argparse
import time
import numpy as np
from lfcc import extract_lfcc
from model import train_and_evaluate_model
from datetime import date
import tensorflow as tf
from sklearn.metrics import roc_curve


def read_file(train_file_path):
    with open(train_file_path, 'r') as file:
        train_filenames = [line.strip() for line in file]
    return train_filenames

def get_audio_files(folder_path, file_extension, train_filenames):
    valid_files = []
    for root, dirs, files in os.walk(folder_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in excluded_folders]  # Exclude certain folders
        for file in files:
            if file.endswith('.' + file_extension):
                full_path = os.path.join(root, file)
                if os.path.splitext(os.path.basename(full_path))[0] in train_filenames:
                    valid_files.append(full_path)
    return valid_files

def generate_labels_from_protocol(path_to_protocol):
    labels_dict = {}
    with open(path_to_protocol, 'r') as file:
        for line in file:
            parts = line.strip().split()
            filename = parts[1]  # Filename without the extension
            label = 1 if parts[4] == 'bonafide' else 0
            labels_dict[filename] = label
    return labels_dict

def preprocess_dataset_with_protocol(audio_files, labels_dict):
    features = []
    labels = []
    max_frame_length = 0
    max_coefficient_length = 1568

    # Calculate the maximum lengths across all features first
    for file in audio_files:
        signal, sr = librosa.load(file, sr=None)
        lfcc_feat = extract_lfcc(signal, sr)
        if lfcc_feat.shape[1] > max_frame_length:
            max_frame_length = lfcc_feat.shape[1]
        if lfcc_feat.shape[0] > max_coefficient_length:
            max_coefficient_length = lfcc_feat.shape[0]

    # Pad and collect features and labels
    for file in audio_files:
        filename = os.path.basename(file).split('.')[0]
        signal, sr = librosa.load(file, sr=None)
        lfcc_feat = extract_lfcc(signal, sr)
        pad_width_time = max_frame_length - lfcc_feat.shape[1]
        pad_width_coefficients = max_coefficient_length - lfcc_feat.shape[0]
        lfcc_feat_padded = np.pad(lfcc_feat, ((0, pad_width_coefficients), (0, pad_width_time)), mode='constant', constant_values=0)
        features.append(lfcc_feat_padded)
        labels.append(labels_dict.get(filename, -1))  # Default label if not found is -1

    # Convert lists to arrays and ensure dimensions for neural network input
    features = np.stack(features, axis=0)
    features = np.expand_dims(features, axis=-1)
    labels = np.array(labels, dtype=np.float32)
    return features, labels

def process_predictions_for_eer(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        y_true = []
        y_scores = []
        for line in lines:
            parts = line.strip().split(' - ')
            y_true.append(float(parts[2][-3:]))
            y_scores.append(float(parts[1][-3:]))
    return calculate_eer(np.array(y_true), np.array(y_scores))

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # Find the nearest point where FPR equals FNR
    eer_threshold_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    return eer

def load_and_predict(best_model_path, X_test, y_test, file_names, scenario):
    model = tf.keras.models.load_model(best_model_path)
    predictions = model.predict(X_test)
    with open(f'./best_result/{scenario}/test_predictions.txt', 'w') as f:
        for idx, prediction in enumerate(predictions):
            filename = file_names[idx]
            f.write(f"{filename} - Predicted: {prediction[0]:.1f} - Actual: {y_test[idx]}\n")
    print(f"Predictions logged successfully to ./best_result/{scenario}/test_predictions.txt")

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--scenario', type=str, default='PA')
parser.add_argument('--ext', type=str, default='wav')

args = parser.parse_args()
scenario = args.scenario
base_path = "/home/sarah.azka/speech/NEW_DATA_" + scenario

data_path = base_path + "/test"
protocol_path = base_path + "/protocol.txt"
test_list = base_path + "/test.lst"

test_filenames = read_file(test_list)
test_audio_files = get_audio_files(data_path, args.ext, test_filenames)
audio_files = test_audio_files

labels_dict = generate_labels_from_protocol(protocol_path)
features, labels = preprocess_dataset_with_protocol(audio_files, labels_dict)

print(f"Data shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

preprocess_time = time.time()
model_path = "best_model_0.0001.h5"

print(f"Predicting using the following model: {model_path}")
load_and_predict(f'./best_result/{scenario}/{model_path}', features, labels, test_filenames, scenario)
predictions_file = f"./best_result/{scenario}/test_predictions.txt"
eer = process_predictions_for_eer(predictions_file)


print(f"EER: {eer}")

end = time.time()
print(f"Time taken for preprocessing: {preprocess_time - start} seconds")
print(f"Time taken to run evaluation test: {end - start} seconds")