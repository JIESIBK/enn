def grid_search(model, loss_fn, dataset, seed, logger, num_batch, start_rate, end_rate, num_sector):
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
