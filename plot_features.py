import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import librosa
from main import read_file, get_audio_files, generate_labels_from_protocol, preprocess_dataset_with_protocol

def plot_lfcc_2d_tsne(audio_files, labels_dict, scenario):
    # get the lfcc features from dataset
    features, labels = preprocess_dataset_with_protocol(audio_files, labels_dict)

    # flatten features for t-SNE since likely still in 3D (samples, coefficients, 1)
    flattened_features = features.reshape(features.shape[0], -1)

    # t-SNE to reduce to 2 dimensions
    tsne = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200)
    reduced_features = tsne.fit_transform(flattened_features)

    # plotting the visualzization
    plt.figure(figsize=(14, 10), dpi=300)
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, 
                          cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.colorbar(scatter).set_label(label='Class',size=20,weight='bold')
    plt.xlabel('TSNE Component 1', fontsize=20)
    plt.ylabel('TSNE Component 2', fontsize=20)
    plt.title(f't-SNE of LFCC Features for {scenario} Scenario', fontsize=24)
    plt.grid(True)
    # plt.rc('font', size=28)          # controls default text sizes
    # plt.rc('axes', titlesize=32)     # fontsize of the axes title
    # plt.rc('axes', labelsize=32)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=32)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=32)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=36)    # legend fontsize
    # plt.rc('figure', titlesize=36)

    # save to file
    save_path_png = f"./lfcc_2d_plot_{scenario}_newest.png"
    save_path_svg = f"./lfcc_2d_plot_{scenario}_newest.svg"
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
    # plt.savefig(save_path)
    print(f"Plot saved to {save_path_png}")

# exxample of use
scenario = "LA"
base_path = "/home/sarah.azka/speech/NEW_DATA_" + scenario

data_path = base_path + "/train_val"
data_path_test = base_path + "/test"

protocol_path = base_path + "/protocol.txt"
train_list = base_path + "/train.lst"
val_list = base_path + "/val.lst"
test_list = base_path + "/test.lst"

train_filenames = read_file(train_list)
val_filenames = read_file(val_list)
test_filenames = read_file(test_list)

train_audio_files = get_audio_files(data_path, "wav", train_filenames)
val_audio_files = get_audio_files(data_path, "wav", val_filenames)
test_audio_files = get_audio_files(data_path_test, "wav", test_filenames)
audio_files = train_audio_files + val_audio_files + test_audio_files

labels_dict = generate_labels_from_protocol(protocol_path)
plot_lfcc_2d_tsne(audio_files, labels_dict, scenario)