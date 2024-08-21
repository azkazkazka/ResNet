import os
import glob
import librosa
import argparse
import time
import numpy as np
from lfcc import extract_lfcc
from model import train_and_evaluate_model
from datetime import date
from sklearn.metrics import roc_curve
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle


def read_file(train_file_path):
    with open(train_file_path, 'r') as file:
        train_filenames = [line.strip() for line in file]
    return train_filenames

def get_audio_files(folder_path, file_extension, train_filenames):
    valid_files = []
    for root, dirs, files in os.walk(folder_path, topdown=True):
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
            filename = parts[1]
            label = 1 if parts[4] == 'bonafide' else 0
            labels_dict[filename] = label
    return labels_dict

def preprocess_dataset_with_protocol(audio_files, labels_dict):
    try:
        features = []
        labels = []
        max_frame_length = 0
        max_coefficient_length = 0

        # Calculate the maximum lengths across all features first
        for file in audio_files:
            signal, sr = librosa.load(file, sr=None)
            lfcc_feat = extract_lfcc(signal, sr)
            if lfcc_feat.size == 0:
                print(f"LFCC extraction failed for file: {file}")
                continue  # Skip this file if LFCC extraction fails
            if lfcc_feat.shape[1] > max_frame_length:
                max_frame_length = lfcc_feat.shape[1]
            if lfcc_feat.shape[0] > max_coefficient_length:
                max_coefficient_length = lfcc_feat.shape[0]

        # Pad and collect features and labels
        for file in audio_files:
            filename = os.path.basename(file).split('.')[0]
            signal, sr = librosa.load(file, sr=None)
            lfcc_feat = extract_lfcc(signal, sr)
            if lfcc_feat.size == 0:
                print(f"LFCC size is 0 for file: {file}")
                continue  # Skip this file if LFCC extraction fails
            pad_width_time = max_frame_length - lfcc_feat.shape[1]
            pad_width_coefficients = max_coefficient_length - lfcc_feat.shape[0]
            lfcc_feat_padded = np.pad(lfcc_feat, ((0, pad_width_coefficients), (0, pad_width_time)), mode='constant', constant_values=0)
            features.append(lfcc_feat_padded.astype(np.float32))  # Ensure features are float32
            labels.append(labels_dict.get(filename, -1))  # Default label if not found is -1

        features = np.stack(features, axis=0)
        features = np.expand_dims(features, axis=-1)
        labels = np.array(labels, dtype=np.float32)  # Ensure labels are float32
        return features, labels
    except Exception as e:
        print(f"Error in preprocess_dataset_with_protocol: {e}")
        return np.array([]), np.array([])

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
    eer_threshold_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    return eer

def load_and_predict(best_model_path, X_test, y_test, file_names, scenario):
    model = tf.keras.models.load_model(best_model_path)
    predictions = model.predict(X_test)
    with open(f'./best_model/{scenario}/test_predictions_test_only_prosa.txt', 'w') as f:
        for idx, prediction in enumerate(predictions):
            filename = file_names[idx]
            f.write(f"{filename} - Predicted: {prediction[0]:.1f} - Actual: {y_test[idx]}\n")
    print(f"Predictions logged successfully to ./best_model/{scenario}/test_predictions_test_only_prosa.txt")

def plot_eers(param, param_eers, scenario):
    plt.figure(figsize=(10, 5))
    plt.plot(param, param_eers, marker='o')
    plt.title(f'EER for Tuning Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('EER')
    plt.grid(True)
    plt.show()
    plt.savefig(f'./tuning_epochs_{scenario}.png')

def save_checkpoint(param_val, model, scenario, param_name):
    model.save(f'./best_model/{scenario}/model_{param_name}_{param_val}.h5')
    print(f"Model checkpoint saved for param {param_name}")

def train_model(args):
    start = time.time()

    scenario = args.scenario
    base_path = f"/home/sarah.azka/speech/NEW_DATA_{scenario}"
    data_path = os.path.join(base_path, "train_val")
    protocol_path = os.path.join(base_path, "protocol.txt")
    train_list = os.path.join(base_path, "train_prosa_only.lst")
    val_list = os.path.join(base_path, "val_prosa_only.lst")

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"Started training for scenario {scenario} at {timestamp}")

    try:
        print("Start preprocess")
        train_filenames = read_file(train_list)
        val_filenames = read_file(val_list)
        train_audio_files = get_audio_files(data_path, args.ext, train_filenames)
        val_audio_files = get_audio_files(data_path, args.ext, val_filenames)
        # audio_files = val_audio_files + train_audio_files

        labels_dict = generate_labels_from_protocol(protocol_path)
        # features, labels = preprocess_dataset_with_protocol(audio_files, labels_dict)
        print("Start training preprocess")
        X_train, y_train = preprocess_dataset_with_protocol(train_audio_files, labels_dict)
        print("Start validation preprocess")
        X_val, y_val = preprocess_dataset_with_protocol(val_audio_files, labels_dict)

        preprocess_time = time.time()
        print(f"Data preprocessing completed in {preprocess_time - start} seconds")

        best_eer = float('inf')
        best_model_info = {}
        epochs_eers = []

        fold = 1
        learning_rate = [0.001]
        batch_size = [16]
        epochs = [50]

        with open(f"model_performance_{scenario}_{timestamp}.txt", "a") as perf_file:
            perf_file.write(f"\n{date.today()}")

        for e in epochs:
            print(f"Training with epochs: {e}")
            all_histories, eers = train_and_evaluate_model(X_train, y_train, train_audio_files, X_val, y_val, val_audio_files, scenario, epochs=e, batch_size=batch_size[0], learning_rate=learning_rate[0], n_splits=fold)
            epochs_eers.append(np.mean(eers))

            for i in range(fold):
                predictions_file = f"./prediction_per_fold/{scenario}/final_predictions_{e}_{batch_size[0]}_{learning_rate[0]}_fold-{i+1}_for_test.txt"
                eer = process_predictions_for_eer(predictions_file)
                eers.append(eer)

            avg_eer = np.mean(eers)
            min_val_losses = [min(history['val_loss']) for history in all_histories if 'val_loss' in history]
            max_val_accuracies = [max(history['val_accuracy']) for history in all_histories if 'val_accuracy' in history]

            mean_val_loss = np.mean(min_val_losses)
            mean_val_accuracy = np.mean(max_val_accuracies)

            with open(f"model_performance_{scenario}_{timestamp}.txt", "a") as perf_file:
                perf_file.write(f"\nEpochs: {e}\n\t\tAverage EER (4-Fold): {avg_eer}\n\t\tAverage Validation Loss: {mean_val_loss}\n\t\tAverage Validation Accuracy: {mean_val_accuracy}")

            if avg_eer < best_eer:
                best_eer = avg_eer
                best_model_info = {
                    'epochs': e,
                    'batch_size': batch_size[0],
                    'learning_rate': learning_rate[0],
                    'EER': avg_eer,
                    'Mean Validation Loss': mean_val_loss,
                    'Mean Validation Accuracy': mean_val_accuracy
                }
        
        print(f"Best Model based on EER: {best_model_info}")
        plot_eers(epochs, epochs_eers, scenario)

        end = time.time()
        print(f"Time taken to run train/eval model: {end - start} seconds")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def evaluate_model(args):
    scenario = args.scenario
    base_path = f"/home/sarah.azka/speech/NEW_DATA_{scenario}"
    data_path = os.path.join(base_path, "test")
    protocol_path = os.path.join(base_path, "protocol.txt")
    test_list = os.path.join(base_path, "test_prosa_only.lst")

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"Started evaluation for scenario {scenario} at {timestamp}")

    try:
        test_filenames = read_file(test_list)
        test_audio_files = get_audio_files(data_path, args.ext, test_filenames)


        labels_dict = generate_labels_from_protocol(protocol_path)
        print("Start testing preprocess")
        features, labels = preprocess_dataset_with_protocol(test_audio_files, labels_dict)

        print(f"Test data shapes: {features.shape}")

        model_name = args.trained_network
        print(f"Predicting using the following model: {model_name}")
        load_and_predict(model_name, features, labels, test_filenames, scenario)
        predictions_file = f"./best_model/{scenario}/test_predictions_test_only_prosa.txt"
        eer = process_predictions_for_eer(predictions_file)
        print(f"Final EER: {eer}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='PA')
    parser.add_argument('--ext', type=str, default='wav')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--trained_network', type=str, default='None')
    args = parser.parse_args()

    print("MULAI")

    print(args.mode)
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'eval':
        # with open('best_model_info.pkl', 'rb') as f:
        #     best_model_info = pickle.load(f)
        evaluate_model(args)

if __name__ == "__main__":
    # Check and configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Since only GPU 3 is visible, it will be treated as GPU 0
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is the GPU available: ", tf.test.is_gpu_available())
    main()