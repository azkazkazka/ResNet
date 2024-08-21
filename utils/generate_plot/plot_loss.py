import re
import matplotlib.pyplot as plt

def parse_logs(file_path):
    fold_data = {}
    current_fold = None

    epoch_pattern = re.compile(r"Epoch (\d+)/50")
    # loss_pattern = re.compile(r"loss: (\d+\.\d+e?-?\d*) - accuracy: \d+\.\d+ - val_loss: (\d+\.\d+e?-?\d*)")
    loss_pattern = re.compile(r"loss: (\d+\.\d+(e[+-]?\d+)?) - accuracy: \d+\.\d+ - val_loss: (\d+\.\d+(e[+-]?\d+)?)")

    
    with open(file_path, 'r') as file:
        for line in file:
            epoch_match = epoch_pattern.search(line)
            loss_match = loss_pattern.search(line)

            if line.startswith("Starting training"):
                if "fold" not in line:
                    current_fold = "1"
                else:
                    current_fold = line.strip()[-1]
                fold_data[current_fold] = {'epoch': [], 'train_loss': [], 'dev_loss': []}
            elif epoch_match:
                epoch_number = epoch_match.group(1)  # This will be '17' as a string
                epoch_number = int(epoch_number.strip())
                fold_data[current_fold]['epoch'].append(epoch_number)
            elif loss_match:
                training_loss = loss_match.group(1)  # This will be '0.0194'
                validation_loss = loss_match.group(3)  # This will be '5.0296e-05'
                print(validation_loss)
                
                training_loss = float(training_loss.strip())
                validation_loss = float(validation_loss.strip())
                fold_data[current_fold]['train_loss'].append(training_loss)
                fold_data[current_fold]['dev_loss'].append(validation_loss)
                
    return fold_data

def plot_loss(fold_data, fold_on=True):
    plt.figure(figsize=(12, 8))
    for fold, data in fold_data.items():
        prefix = ""
        if fold_on:
            prefix = f"Fold {fold} "
        plt.plot(data['epoch'], data['train_loss'], label=f'{prefix}Train Loss')
        plt.plot(data['epoch'], data['dev_loss'], label=f'{prefix}Dev Loss', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Development Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./plot_loss_test.jpg")
    plt.show()

def plot_loss_per_fold(fold_data, save_folder):
    for fold, data in fold_data.items():
        plt.figure(figsize=(12, 8))
        
        plt.plot(data['epoch'], data['train_loss'], label='Train Loss')
        plt.plot(data['epoch'], data['dev_loss'], label='Dev Loss', linestyle='--')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss per Epoch for Fold {fold}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{save_folder}/fold_{fold}_loss_plot.png")
        plt.show()
        


# File path to log file
log_file_path = '../../log_training_for_test.txt'

# Parse the logs
fold_data = parse_logs(log_file_path)

# Plot the loss graphs
plot_loss(fold_data, fold_on=False)


# # Folder to save the plots
# save_folder = '.'

# # Plot the loss graphs for each fold and save the plots
# plot_loss_per_fold(fold_data, save_folder)