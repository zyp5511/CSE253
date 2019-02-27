# %%
import os
import numpy as np
import math

from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data_handler import *
import utils

from keras.utils import to_categorical

# %%
opts = EasyDict()
opts.data_directory = "./Data/"
opts.train_data_file = "train.txt"
opts.val_data_file = "val.txt"
opts.n_epochs = 100
opts.batch_size = 1  # TODO minibatch size 1 for simplifying / taking chunks of 100
opts.seq_length = 100
opts.learning_rate = 0.1
opts.lr_decay = 0.99
opts.hidden_size = 100
opts.generate_seq_length = 10
opts.temperature = 0.6
opts.model_name = "LSTM"
opts.checkpoints_dir = "./checkpoints/" + opts.model_name


# %%
def read_batches(list_index, opts):
    # discards the last chunks
    for itr in range(0, len(list_index)):
        song = list_index[itr]
        song_length = len(song)
        num_batch = math.ceil(1.0 * song_length / opts.seq_length)
        for batch_index in range(0, num_batch):
            if batch_index == num_batch - 1:
                temp = song[batch_index * opts.seq_length:]
                if len(temp) <= 1:
                    inputs = temp
                    targets = []
                else:
                    inputs = temp
                    targets = temp[1:]
                inputs = np.pad(np.array(inputs), (0, opts.seq_length - len(inputs)), 'constant',
                                constant_values=(0, 93))
                targets = np.pad(np.array(targets), (0, opts.seq_length - len(targets)), 'constant',
                                 constant_values=(0, 93))
            else:
                inputs = np.array(song[batch_index * opts.seq_length:(batch_index + 1) * opts.seq_length])
                targets = np.array(song[batch_index * opts.seq_length + 1:(batch_index + 1) * opts.seq_length + 1])
            inputs = np.expand_dims(inputs, axis=0)
            targets = np.expand_dims(targets, axis=0) 
            input_tensors = torch.LongTensor(inputs)
            target_tensors = torch.LongTensor(targets)
            yield input_tensors, target_tensors
    """

    chunk = opts.batch_size*opts.seq_length

    for i in range(0, batch_num): 

        if (i != batch_num-1):
            start = i * chunk
            end = start + chunk

            #TODO check for chunking
            inputs = all_characters[start:end]
            targets = all_characters[start+1:end+1]

            #TODO how you want to define batches?
            input_tensors = torch.LongTensor(inputs).view(opts.batch_size, opts.seq_length)
            output_tensors = torch.LongTensor(targets).view(opts.batch_size, opts.seq_length)

        #TODO how do you want to tackle the last part
        else:
            start = i * chunk

            inputs = all_characters[start:]
            targets = np.concatenate((all_characters[start+1:], [idx_dict['end_token']]))


            input_tensors = torch.LongTensor(inputs).view(1, len(inputs))
            output_tensors = torch.LongTensor(targets).view(1, len(targets))


        yield input_tensors, output_tensors
    """


# %%
class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(RNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(vocab_size, hidden_size)

        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first = True)
        self.lstm = nn.LSTM(vocab_size, hidden_size, batch_first=True)

        self.out = nn.Linear(hidden_size, vocab_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden=None):
        # encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        # output, hidden = self.lstm(encoded, hidden)
        output, hidden = self.lstm(inputs, hidden)
        outputs = self.out(output)

        return outputs, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(1, bsz, 100),weight.new_zeros(1, bsz, 100))

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def training_model(train_characters, val_characters, vocab_size, idx_dict, model, opts, epochs=8):
    model_optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate)
    criterion = nn.CrossEntropyLoss().to(computing_device)

    loss_log = open(os.path.join(opts.checkpoints_dir, 'loss_log.txt'), 'w')

    best_val_loss = 1e6
    train_losses = []
    val_losses = []

    loss_log.write('started training')
    hidden = model.init_hidden(1)
    for epoch in range(opts.n_epochs):

        model_optimizer.param_groups[0]['lr'] *= opts.lr_decay

        epoch_losses = []
        print("Epoch {}/{}".format(epoch + 1, epochs))

        for i, (inputs, targets) in enumerate(read_batches(train_characters, opts)):

            model_optimizer.zero_grad()
            
            inputs_embed = torch.FloatTensor(to_categorical(inputs, num_classes=vocab_size)).to(computing_device)
            targets = targets.to(computing_device)
            hidden = repackage_hidden(hidden)
            outputs, hidden = model.forward(inputs_embed, hidden)
            
            loss = 0.0
            
            # TODO check
            for j in range(targets.shape[1]):
                loss += criterion(outputs[:, j, :], targets[:, j])

            loss /= float(targets.shape[1])

            # TODO check
            loss.backward()

            model_optimizer.step()

            epoch_losses.append(loss.item())
            
            if 93 in inputs.cpu().detach().numpy()[0]:
                hidden = model.init_hidden(1)

            if (i % 100 == 0):
                print("Batch: {}, Loss: {}".format(i + 1, loss.item()))

            if (i % 1000 == 0):

                train_loss = np.mean(epoch_losses)
                val_loss = evaluate(val_characters, model, idx_dict, criterion, opts)

                if val_loss < best_val_loss:
                    utils.store_checkpoints(model, opts)

                generate_tune = generate_sequence('<start>', idx_dict, model,
                                                  opts)  # TODO should we choose the best model here?
                print(
                    "Epoch: {:3d}| Batch: {:3d} | Train loss: {:.3f} | Val loss: {:.3f} | Gen: {:20s}".format(epoch, i,
                                                                                                              train_loss,
                                                                                                              val_loss,
                                                                                                              generate_tune))

                loss_log.write('{} {} {}\n'.format(epoch, train_loss, val_loss))
                loss_log.flush()

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                utils.store_loss_plots(train_losses, val_losses, opts)

                epoch_losses = []


def evaluate(data, model, idx_dict, criterion, opts):
    losses = []
    hidden = None

    for i, (inputs, targets) in enumerate(read_batches(data, opts)):

        inputs = torch.FloatTensor(to_categorical(inputs, num_classes=vocab_size)).to(computing_device)
        targets = targets.to(computing_device)

        outputs, hidden = model.forward(inputs, hidden)

        loss = 0.0

        for j in range(targets.shape[1]):
            loss += criterion(outputs[:, j, :], targets[:, j])

        loss /= float(targets.shape[1])
        losses.append(loss.item())

    mean_loss = np.mean(losses)

    return mean_loss


# %%
def generate_sequence(start_string, idx_dict, model, opts):
    char_to_index = idx_dict['char_to_index']
    index_to_char = idx_dict['index_to_char']

    # start_string = 'abc' #for a better hidden state, in our case should be stated with <start>
    start_characters = np.asarray([char_to_index[c] for c in start_string], dtype=np.int32)

    inputs = np.zeros((1, len(start_string)))

    for i in range(0, len(start_string)):  # TODO limit here if want to start only with <
        inputs[0, i] = start_characters[i]

    inputs = torch.LongTensor(inputs)
    hidden = None
    inputs = torch.FloatTensor(to_categorical(inputs, num_classes=vocab_size)).to(computing_device)
    outputs, hidden = model.forward(inputs, hidden)
    output = outputs[:, -1:, ]

    final_output_sequence = []

    for i in range(opts.generate_seq_length):
        probabilities = F.softmax(output.div(opts.temperature).squeeze(0).squeeze(0))

        current_input = torch.multinomial(probabilities.data, 1)

        final_output_sequence.append(current_input.data)

        current_input = torch.cuda.FloatTensor(
            to_categorical(current_input.cpu().detach().numpy(), num_classes=vocab_size)).unsqueeze(
            0)  # .cpu().to(computing_device)

        output, hidden = model.forward(current_input, hidden)

    sampled_sequence = torch.cat(final_output_sequence, dim=0).cpu().detach().numpy()

    geneated_seq = ''.join([index_to_char[i] for i in sampled_sequence])

    return geneated_seq


# %%
# train_characters, vocab_size, idx_dict = load_data(opts.data_directory+opts.train_data_file)
train_characters, vocab_size, idx_dict = load_data(opts.data_directory + opts.train_data_file)
val_characters, _, _ = load_data(opts.data_directory + opts.val_data_file,idx_dict)
model = RNN(vocab_size=vocab_size, hidden_size=opts.hidden_size)

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else:  # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

if __name__ == "__main__":
    utils.create_dir_if_not_exists(opts.checkpoints_dir)
    training_model(train_characters, val_characters, vocab_size, idx_dict, model, opts)

    best_model = RNN(vocab_size=vocab_size, hidden_size=opts.hidden_size)
    best_model = utils.restore_checkpoints(best_model, opts)
    best_model = best_model.to(computing_device)
    geneated_seq = generate_sequence('<start>', idx_dict, best_model, opts)
    print(geneated_seq)
