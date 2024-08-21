import os
import glob
import librosa
import argparse
import time
import numpy as np
from lfcc import extract_lfcc
from model_fold import train_and_evaluate_model

def read_train_file(train_file_path):
    with open(train_file_path, 'r') as file:
        train_filenames = {line.strip() for line in file}  # Create a set of filenames
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

def generate_labels(file_paths):
    labels = []
    for file_path in file_paths:
        if 'deteksi_suara_palsu' in file_path:
            labels.append(1)  # bonafide
        else:
            labels.append(0)  # spoof
    return np.array(labels, dtype=np.int32)

def preprocess_dataset(audio_files):
    features = []
    max_frame_length = 0
    max_coefficient_length = 0

    # First, determine the maximum size in each dimension
    for file in audio_files:
        signal, sr = librosa.load(file, sr=None)
        lfcc_features = extract_lfcc(signal, sr)
        if lfcc_features.shape[1] > max_frame_length:
            max_frame_length = lfcc_features.shape[1]
        if lfcc_features.shape[0] > max_coefficient_length:
            max_coefficient_length = lfcc_features.shape[0]
        features.append(lfcc_features)

    # Pad each feature array to the maximum size
    padded_features = []
    for feature in features:
        pad_width_time = max_frame_length - feature.shape[1]
        pad_width_coefficients = max_coefficient_length - feature.shape[0]
        padded_feature = np.pad(feature, ((0, pad_width_coefficients), (0, pad_width_time)), mode='constant', constant_values=0)
        padded_features.append(padded_feature)

    # Stack the uniformly sized feature arrays
    if padded_features:
        features = np.stack(padded_features, axis=0)
    else:
        print("No features to process.")
        features = np.array([])  # Return an empty array if no features

    return features

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
    max_coefficient_length = 0

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

start = time.time()
# parser = argparse.ArgumentParser()
# parser.add_argument('--ext', type=str, default='flac')
# parser.add_argument('--path', type=str, default='/home/sarah.azka/speech/RESNET/toy_example/train_dev')
# parser.add_argument('--protocols', type=str, default='/home/sarah.azka/speech/RESNET/toy_example/protocol.txt')

# args = parser.parse_args()
# extension = args.ext
# train_data_path = args.path
# protocols = args.protocols

# audio_files = get_audio_files(train_data_path, extension)
# labels_dict = generate_labels_from_protocol(protocols)

# data, labels = preprocess_dataset(audio_files, labels_dict)

# # Verify shapes
# print(f"Data shape: {data.shape}")
# print(f"Labels shape: {labels.shape}")

# train_and_evaluate_model(data, labels)

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/home/sarah.azka/speech/spoof_data/')
parser.add_argument('--train', type=str, default='/home/sarah.azka/speech/spoof_data/train.lst')
parser.add_argument('--ext', type=str, default='wav')
parser.add_argument('--protocols', type=str, default='/home/sarah.azka/speech/spoof_data/protocol.txt')

args = parser.parse_args()
train_filenames = read_train_file(args.train)
audio_files = get_audio_files(args.path, args.ext, train_filenames)
labels = generate_labels(audio_files)
features = preprocess_dataset(audio_files)

# audio_files = get_audio_files(args.path, args.ext, args.train)
# labels_dict = generate_labels_from_protocol(args.protocols)

# features, labels = preprocess_dataset_with_protocol(audio_files, labels_dict)

preprocess_time = time.time()

train_and_evaluate_model(features, labels)

end = time.time()
print(f"Data shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Time taken for preprocessing: {preprocess_time-start} seconds")
print(f"Time taken to run train/eval model: {end-start} seconds")

