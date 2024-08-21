import matplotlib.pyplot as plt

epochs = list(range(1, len(metrics['EER']) + 1))

plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics['EER'], label='EER', marker='o', linestyle='-', color='blue')
plt.plot(epochs, metrics['Mean Validation Loss'], label='Mean Validation Loss', marker='x', linestyle='--', color='red')
plt.plot(epochs, metrics['Mean Validation Accuracy'], label='Mean Validation Accuracy', marker='^', linestyle='-.', color='green')

plt.title('Performance Metrics per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Metric Values')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig('/path/to/save/performance_metrics.png')
