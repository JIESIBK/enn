import numpy as np
import optax
from enn import supervised
import dill

def grid_search(model, config, epinet, loss_fn, dataset, seed, logger, num_batch, start_rate, end_rate, num_sector):
    lr_range = np.logspace(np.log10(start_rate), np.log10(end_rate), num_sector).tolist()
    best_losses = []
    final_epochs = []
    for lr in lr_range:
        optimizer = optax.adam(config.learning_rate)
        experiment = supervised.Experiment(epinet, loss_fn, 
                                           optimizer, 
                                           dataset, seed, logger)
        print("Training with lr: ", lr)
        best_loss, final_epoch = experiment.train(num_batch)
        best_losses.append(best_loss)
        final_epochs.append(final_epoch)
    
    print("Learning_rate: ", lr_range)
    print("Best_loss: ", best_losses)
    print("Epoch_elapsed: ", final_epochs)

    min_loss = min(best_losses)
    min_pos = best_losses.index(min_loss)

    if min_pos == 0:
        start_rate = lr_range[0]
        end_rate = lr_range[1]
    elif min_pos == num_batch-1:
        start_rate = lr_range[min_pos-1]
        end_rate = lr_range[min_pos]      
    else:
        if best_losses[min_pos-1] < best_losses[min_pos+1]:
            start_rate = lr_range[min_pos-1]
            end_rate = lr_range[min_pos]
        else:
            start_rate = lr_range[min_pos]
            end_rate = lr_range[min_pos+1]
    
    lr_range = np.linspace(start_rate, end_rate, num_sector).tolist()
    best_losses = []
    final_epochs = []

    for lr in lr_range:
        optimizer = optax.adam(config.learning_rate)
        experiment = supervised.Experiment(epinet, loss_fn, 
                                           optimizer, 
                                           dataset, seed, logger)
        print("Training with lr: ", lr)
        best_loss, final_epoch = experiment.train(num_batch)
        best_losses.append(best_loss)
        final_epochs.append(final_epoch)
    
    print("Learning_rate: ", lr_range)
    print("Best_loss: ", best_losses)
    print("Epoch_elapsed: ", final_epochs)

    return (min(best_losses), lr_range[best_losses.index(min(best_losses))])


def extract_epochs_and_losses(training_log):

    import re

    epochs = []
    losses = []
    epoch_pattern = r"Epoch: (\d+), Loss: ([\d.]+)"
    for line in training_log.split('\n'):
        match = re.search(epoch_pattern, line)
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))
    return epochs, losses


def plot_loss_curve(log_file_path):
    import matplotlib.pyplot as plt

    # Extracting epoch numbers and loss values
    epochs_all = []
    losses_all = []
    training_log = []

    for file_path in log_file_path:
        with open(file_path, 'r') as file:
            training_log.append(file.read())

    # Using regular expressions to find matches
    for log in training_log:
        epochs, losses = extract_epochs_and_losses(log)
        epochs_all.append(epochs)
        losses_all.append(losses)

    # Plotting the loss vs epoch for both logs
    plt.figure(figsize=(20, 12))

    markers = ['o', 'x', '.', ',', '^', '|']
    colors = ['blue', 'red', 'green', 'c', 'm', 'orange']

    for idx in range(len(log_file_path)):    
        plt.plot(epochs_all[idx], losses_all[idx], marker=markers[idx], color=colors[idx], linestyle='-', label='Training Log ' + str(idx))

    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(False)
    
    plt.xticks(epochs_all[0] + epochs_all[1] + epochs_all[2] + epochs_all[3] + epochs_all[4] + epochs_all[5], rotation='vertical')  # Making epoch labels vertical
    
    plt.legend()

    # Save the plot
    save_path = 'training_loss_plot_combined.png'
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    
    log_files = ["/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/training.log",
                 "/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/training_#2.log",
                 "/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/training_#3.log",
                 "/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/training_#4.log",
                 "/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/training_#5.log",
                 "/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/training_#6.log"]
    plot_loss_curve(log_files)
