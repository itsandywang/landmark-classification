"""
EECS 445 - Introduction to Machine Learning
Fall 2023 - Project 2

Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from model.challenge import Challenge
from train_target import *
from train_common import *
from utils import config
import utils
import copy
from helper import *

            

def main():
    # Data loaders
    if check_for_augmented_data("./data"):
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target", batch_size=config("challenge.batch_size"), augment=True
        )
    else:
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target",
            batch_size=config("challenge.batch_size"),
        )
    # Model
    model = Challenge()
#------------------------------------------------------------------------------
    freeze_one = copy.deepcopy(model)
    freeze_two = copy.deepcopy(model)
    freeze_three = copy.deepcopy(model)

    freeze_layers(model, 1)
    freeze_layers(model, 2)
    freeze_layers(model, 3)

    train(tr_loader, va_loader, te_loader, model, "./checkpoints/target0/", 0)
    train(tr_loader, va_loader, te_loader, freeze_one, "./checkpoints/target1/", 1)
    train(tr_loader, va_loader, te_loader, freeze_two, "./checkpoints/target2/", 2)
    train(tr_loader, va_loader, te_loader, freeze_three, "./checkpoints/target3/", 3)

#------------------------------------------------------------------------------

    # TODO: Define loss function and optimizer. Replace "None" with the appropriate definitions.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=10**-3)

    # Attempts to restore the latest checkpoint if exists
    print("Loading challenge...")
    model, start_epoch, stats = restore_checkpoint(
        model, config("challenge.checkpoint")
    )

    axes = utils.make_training_plot()

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats
    )

    # initial val loss for early stopping
    global_min_loss = stats[0][1]

    # TODO: Define patience for early stopping. Replace "None" with the patience value.
    patience = 5
    curr_count_to_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    
    min_validation_loss = float("inf")
    best_epoch = None
    best_epoch_stats = None
    
    while curr_count_to_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("challenge.checkpoint"), stats)

        best_epoch, best_epoch_stats, min_validation_loss = update_best_epoch(epoch, min_validation_loss, best_epoch_stats, best_epoch, stats)

        # TODO: Implement early stopping
        curr_count_to_patience, global_min_loss = early_stopping(
            stats, curr_count_to_patience, global_min_loss
        )
        #
        epoch += 1
    print("Finished Training")
    print_best_epoch_stats(best_epoch, best_epoch_stats, stats)
    

    # Save figure and keep plot open
    utils.save_challenge_training_plot()
    utils.hold_training_plot()


if __name__ == "__main__":
    main()
