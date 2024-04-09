import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_loss_from_csv(csv_file_path, title):
    data = pd.read_csv(csv_file_path)
    steps = data['Step']
    values = data['Value']

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, label='Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    save_path = os.path.splitext(csv_file_path)[0] + '.png'
    plt.savefig(save_path)
    print(f"Plot saved as: {save_path}")

plot_loss_from_csv('train_loss.csv', 'Train Loss')
plot_loss_from_csv('val_loss.csv', 'Eval Loss')
