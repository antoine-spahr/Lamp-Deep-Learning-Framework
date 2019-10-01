#
# Note : it takes approximatly 280 seconds to run this script on a 2,8 GHz Intel Core i7
#
#########################################################################################

import torch
import math
import matplotlib.pyplot as plt
from functions import *
import lamp.Modules as mod
import lamp.Sequential as seq
import lamp.Optimizers as optim

# Deactivate the autograd mode
torch.set_grad_enabled(False)

# Train multiple model Model
N_train = 10
N_epochs = 400
eta = 0.05
mini_batch_size = 25
losses = []

# to store outputs of trainings to enables plotting
train_accuracy = torch.empty(N_train)
test_accuracy = torch.empty(N_train)
all_train_input = []
all_train_target = []
all_train_pred = []
all_test_input = []
all_test_target = []
all_test_pred = []

# perform the N_train training
for i in range(N_train):
    # Get Data
    train_input, train_target, train_target_hot = generate_data(1000, one_hot_label=True)
    test_input, test_target, test_target_hot = generate_data(1000, one_hot_label=True)

    # define architecture
    model = seq.Sequential(mod.Linear(2,25), mod.SELU(), \
                           mod.Linear(25,25), mod.SELU(), \
                           mod.Linear(25,25), mod.SELU(), \
                           mod.Linear(25,2))

    # define the loss and optiizer
    criterion = mod.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.SGD(model.parameters(), eta, momentum=True)

    # train the model and get the losses at each epochs
    losses.append(train_model(model, criterion, optimizer, train_input, train_target_hot, N_epochs, mini_batch_size, eta))

    # Compute accuracy
    train_pred = model.forward(train_input).argmax(dim=1)
    test_pred = model.forward(test_input).argmax(dim=1)
    train_accuracy[i] = compute_accuracy(train_pred, train_target)
    test_accuracy[i] = compute_accuracy(test_pred, test_target)

    # stores values
    all_train_input.append(train_input)
    all_train_target.append(train_target)
    all_train_pred.append(train_pred)
    all_test_input.append(test_input)
    all_test_target.append(test_target)
    all_test_pred.append(test_pred)

# %% Print performances over the N_train
print('Train accuracy : {0:.2%} +- {1:.2%}\nTest accuracy : {2:.2%} +- {3:.2%}'.format(train_accuracy.mean(), train_accuracy.std(), test_accuracy.mean(), test_accuracy.std()))

# %% Plots the results
plt.rcParams.update({'font.size': 13})

fig, axs = plt.subplots(1,3,figsize=(22,7))

colors = plt.cm.GnBu([i/N_train for i in range(N_train)])

for idx, l in enumerate(losses):
    axs[0].plot(range(1,N_epochs+1), l.numpy(), linewidth=2.5, color=colors[idx])
axs[0].set_xlabel('epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Loss during the gradient descent on {0} training'.format(N_train))
axs[0].set_xlim([0,N_epochs])

plot_prediction(torch.cat(all_train_input, dim=0), torch.cat(all_train_pred, dim=0), torch.cat(all_train_target, dim=0), axs[1], 'Train predictions (accuracy = {0:.2%} +- {1:.2%})'.format(train_accuracy.mean(), train_accuracy.std()), misclassified=True, legend=True, legend_pos=(0.5, -0.18))
plot_prediction(torch.cat(all_test_input, dim=0), torch.cat(all_test_pred, dim=0), torch.cat(all_test_target, dim=0), axs[2], 'Test predictions (accuracy = {0:.2%} +- {1:.2%})'.format(test_accuracy.mean(), test_accuracy.std()), misclassified=True, legend=False)

plt.show()
