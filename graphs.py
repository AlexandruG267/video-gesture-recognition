import matplotlib.pyplot as plt
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# .py file I made for the values
import train_results

epochs = 30

val_accs = {"baseline": {"mean":[9.43396282196045, 9.562453269958496, 13.248123168945312, 16.31162452697754, 20.247514724731445, 28.64002227783203, 31.20308494567871, 33.97578811645508, 36.90403747558594, 35.477108001708984, 41.84756851196289, 37.51944351196289, 42.361534118652344, 44.620277404785156, 42.61851501464844, 32.75850296020508, 42.6388053894043, 42.54412841796875, 43.08514404296875, 43.96429443359375, 43.86961364746094, 43.2474479675293, 44.4444465637207, 44.27537536621094, 44.2280387878418, 44.4309196472168, 44.796104431152344, 45.0057487487793, 44.762290954589844, 44.85696792602539],
                       "abs_diff": [25.042264938354492, 33.854061126708984, 39.974300384521484, 39.724082946777344, 39.805233001708984, 46.006629943847656, 49.03631591796875, 45.4926643371582, 53.614662170410156, 53.2224235534668, 53.72286605834961, 48.7252311706543, 52.20125961303711, 51.673770904541016, 49.09041976928711, 51.36944580078125, 51.552040100097656, 53.263004302978516, 54.865760803222656, 55.880165100097656, 55.59613037109375, 57.1583137512207, 56.8539924621582, 56.8066520690918, 56.3467903137207, 56.90809631347656, 56.691688537597656, 56.779605865478516, 56.63082504272461, 56.745792388916016],
                       "rel_diff": [25.63738441467285, 21.44451332092285, 38.04016876220703, 44.23480224609375, 42.070735931396484, 51.572330474853516, 52.71522521972656, 51.28829574584961, 56.25887680053711, 58.44322967529297, 58.11861801147461, 55.8260612487793, 55.3661994934082, 58.625823974609375, 59.504974365234375, 57.002769470214844, 60.28944396972656, 58.99776840209961, 60.64110565185547, 61.64198684692383, 62.0950813293457, 62.933658599853516, 62.6901969909668, 62.82545471191406, 63.33941650390625, 63.08243942260742, 63.32589340209961, 63.42733383178711, 63.41381072998047, 63.474674224853516]},
            "conv": {
                # "unaltered_loss": train_results.TRAIN_LOSS_UNALTERED_3D,
                "unaltered_acc": train_results.VAL_ACC_UNALTERED_3D,
                # "diff_loss": train_results.TRAIN_LOSS_DIFF_3D,
                "diff_acc": train_results.VAL_ACC_DIFF_3D,
                # "rel_diff_loss": train_results.TRAIN_LOSS_REL_DIFF_3D,
                "rel_diff_acc": train_results.VAL_ACC_REL_DIFF_3D
            }
}


model_types = ["baseline", "conv"]


def generate_accuracy_over_epochs_graph():
    fig, axs = plt.subplots(1, len(model_types), figsize=(16, 8))
    # fig.subplots_adjust(hspace=0.4)

    for i, model in enumerate(model_types):

        x = np.arange(1, epochs + 1)
        print(model)
        # print(val_accs[model])
        for diff_type, y in val_accs[model].items():
            clean_label = diff_type.replace("_acc", "").replace("_", " ").title()
            axs[i].plot(x, y, label=clean_label, linewidth=2)

        axs[i].grid(True, )
        axs[i].legend(fontsize=10, loc='lower right')

        axs[i].set_xlabel("Epoch", fontsize=14)
        axs[i].set_ylabel("Validation set accuracy", fontsize=14)
        axs[i].set_title(f"Validation set accuracy over the epochs for the {model} model", fontsize=18, pad=20)

    plt.tight_layout(pad=3.0, w_pad=4.0)
    plt.show()
    plt.close()


def generate_combined_accuracy_graph():
    # Single figure, large size for clarity
    plt.figure(figsize=(14, 10))
    x = np.arange(1, epochs + 1)

    for model in model_types:
        # VISUAL TIP: Use dashed lines for baseline to distinguish them from the main 3D models
        line_style = '--' if model == 'baseline' else '-'
        linewidth = 2 if model == 'conv' else 1.5 # Make the 3D models slightly thicker

        print(f"Plotting {model}...")
        
        # Iterate through the dictionary for this model (Baseline or Conv)
        # val_accs is defined globally in your graphs.py file
        for data_type, y in val_accs[model].items():
            
            # Optional: Clean up labels (e.g., "unaltered_acc" -> "Unaltered")
            clean_label = data_type.replace("_acc", "").replace("_", " ").title()
            label = f"{model.title()}: {clean_label}"
            
            plt.plot(x, y, label=label, linestyle=line_style, linewidth=linewidth)

    # Styling
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='lower right') # Lower right is usually empty for accuracy graphs
    
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Validation Set Accuracy (%)", fontsize=14)
    
    # Pad adds space between the title and the chart
    plt.title("Model Comparison: 2D Baseline vs. 3D Convolution Performance", fontsize=18, pad=20)
    
    plt.tight_layout()
    plt.show()
    plt.close()

    
def generate_accuracy_over_epochs_simple_graph(model):

    x = np.arange(1, epochs + 1)
    print(model)
    # print(val_accs[model])
    for diff_type, y in val_accs[model].items():
        plt.plot(x, y, label=diff_type)

    plt.grid(True, )
    plt.legend()

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Validation set accuracy", fontsize=14)
    plt.title(f"Validation set accuracy over the epochs for the {model} model", fontsize=18)

    plt.show()
    plt.close()


if __name__ == "__main__":
    # Code for generating the accuract over epochs graph

    # generate_combined_accuracy_graph()
    generate_accuracy_over_epochs_graph()
    # generate_accuracy_over_epochs_simple_graph("baseline")
    # generate_accuracy_over_epochs_simple_graph("conv")

