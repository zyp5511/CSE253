import os
import sys
import pickle as pkl

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def store_checkpoints(model, opts):
    opts.model = model
    torch.save(model.state_dict(), opts.checkpoints_dir+'/best_model')
        
def restore_checkpoints(model, opts):
    model.load_state_dict(torch.load(opts.checkpoints_dir+'/best_model'))
    opts.model = model
    return model


def store_loss_plots(train_losses, val_losses, opts):
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label = 'Training Set Loss')
    plt.plot(range(len(val_losses)), val_losses, label = 'Validation Set Loss')
    plt.legend(loc='upper right')
    plt.title('batch_size={}, lr={}, hidden_size={}'.format(opts.batch_size, opts.learning_rate, opts.hidden_size), fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(opts.checkpoints_dir, 'loss_plot.pdf'))
    plt.close()
    
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
