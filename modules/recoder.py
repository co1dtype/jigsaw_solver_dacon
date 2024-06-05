import csv
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from typing import Dict
import os

# Define the function to record data and save plots without displaying them
def record(data: Dict[str, float], font_path="/home/hyj/ChanHyung/SSL/PUZZLE_AI/OPTITimesRoman-Italic.otf"):
    # Define the filenames for the CSV and plots
    csv_filename = os.path.join(save_folder, 'training_data.csv')
    plot_filenames = {
        'val_loss': os.path.join(save_folder, 'val_loss_plot.png'),
        'val_accuracy': os.path.join(save_folder, 'val_accuracy_plot.png'),
        'val_dacon_score': os.path.join(save_folder, 'val_dacon_score_plot.png')
    }

    # Append the data to the CSV
    fieldnames = list(data.keys())
    try:
        # Check if the CSV already exists to append or write headers
        with open(csv_filename, 'x', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(data)
    except FileExistsError:
        with open(csv_filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(data)

    # Read the updated data for plotting
    with open(csv_filename, newline='') as file:
        reader = csv.DictReader(file)
        epochs = []
        metrics = {key: [] for key in data.keys()}
        for i, row in enumerate(reader):
            epochs.append(i + 1)  # Assuming each row is a new epoch
            for key in data.keys():
                metrics[key].append(float(row[key]))

    # Set up the custom font
    prop = FontProperties(fname=font_path)

    # Create and save the plots without displaying them
    plt.ioff()  # Turn off interactive mode to prevent displaying figures
    
    # Plot val_loss
    plt.figure()
    plt.plot(epochs, metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs', fontproperties=prop)
    plt.ylabel('Validation Loss', fontproperties=prop)
    plt.title('Validation Loss over Epochs', fontproperties=prop)
    plt.legend(prop=prop)
    plt.savefig(plot_filenames['val_loss'])

    # Plot val_1x1, val_2x2, val_3x3, val_4x4 in one graph
    plt.figure()
    plt.plot(epochs, metrics['val_1x1'], label='1x1 Accuracy')
    plt.plot(epochs, metrics['val_2x2'], label='2x2 Accuracy')
    plt.plot(epochs, metrics['val_3x3'], label='3x3 Accuracy')
    plt.plot(epochs, metrics['val_4x4'], label='4x4 Accuracy')
    plt.xlabel('Epochs', fontproperties=prop)
    plt.ylabel('Accuracy', fontproperties=prop)
    plt.title('Validation Accuracy over Epochs', fontproperties=prop)
    plt.legend(prop=prop)
    plt.savefig(plot_filenames['val_accuracy'])

    # Plot val_dacon_score
    plt.figure()
    plt.plot(epochs, metrics['val_dacon_score [AVG]'], label='Dacon Score')
    plt.xlabel('Epochs', fontproperties=prop)
    plt.ylabel('Dacon Score', fontproperties=prop)
    plt.title('Dacon Score over Epochs', fontproperties=prop)
    plt.legend(prop=prop)
    plt.savefig(plot_filenames['val_dacon_score'])

    plt.ion()  # Turn interactive mode back on

    # Return paths of generated files
    return {
        'csv_file': csv_filename,
        'plot_files': plot_filenames
    }


def optimizer_to_yaml_dict(optimizer):
    return {
        'type': optimizer.__class__.__name__,
        'defaults': {k: v if not isinstance(v, torch.Tensor) else 'Tensor' for k, v in optimizer.defaults.items()}
    }

# Function to convert a scheduler to a YAML-compatible dictionary
def scheduler_to_yaml_dict(scheduler):
    return {
        'type': scheduler.__class__.__name__,
        'scheduler_params': {
            'mode': scheduler.mode,
            'factor': scheduler.factor,
            'patience': scheduler.patience,
            'threshold_mode': scheduler.threshold_mode,
            'min_lr': scheduler.min_lrs,
            'verbose': scheduler.verbose
        }
    }

# Function to convert a criterion to a YAML-compatible dictionary
def criterion_to_yaml_dict(criterion):
    return {
        'type': criterion.__class__.__name__
    }


def transform_to_dict(transform):
    transform_list = []
    for t in transform.transforms:
        transform_name = t.__class__.__name__
        params = {k: v for k, v in t.__dict__.items() if not k.startswith('_')}
        transform_list.append({transform_name: params or None})
    return transform_list