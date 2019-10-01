import torch
import math
import matplotlib.pyplot as plt
import lamp.Modules as mod
import lamp.Sequential as seq
import lamp.Optimizers as optim

def generate_data(N_samples, one_hot_label=False):
    # <*> DOCSTRING
    '''
        Generate N_samples data point uniformly distributed on [0,1]x[0,1], labeled
        as 0 if they are outside the disc of radius 1/sqrt(2pi), else 1.

        INPUT: -> N_samples [integer] number of sample to geenrate
               -> one_hot_label [bool] states whether one_hot_label should be returned

        OUTPUT: -> data [torch.Tensor] of dimension N_samples x 2
                -> target [torch.Tensor] of dimension N_samples x 1
                -> target_hot [torch.Tensor] of dimension N_sample x 2 (if one_hot_label = True)
    '''
    # <*!>
    # Generate a uniformly distributed data in [0,1]x[0,1]
    data = torch.empty(N_samples, 2).uniform_(0,1)
    # define the target as 1 if the point is contained in the circle of radius 1/sqrt(2*pi)
    target = ((data).pow(2).sum(dim=1).sqrt() <= 1/math.sqrt(2*math.pi)) # & ((data-0.5).pow(2).sum(dim=1).sqrt() >= 0.5/math.sqrt(2*math.pi))

    if one_hot_label:
        # build the one_hot_label is required
        target_hot = torch.cat(((target==0).view(-1,1), (target==1).view(-1,1)), dim=1).float()
        return data, target, target_hot
    else:
        return data, target

def train_model(model, criterion, optimizer, input, target, N_epochs=25, mini_batch_size=100, eta=0.1):
    # <*> DOCSTRING
     '''
        Train the given model

        INPUT : -> model [lamp.Sequential] the model to be trained
                -> criterion [lamp.Module] loss to be minimized
                -> optimizer [lamp.Optimizer] object defining the kind of gradient descent to use
                -> input [torch.Tensor] training input data with dimension n_sample x input_dimension
                -> target [torch.Tensor] training target as one_hot_label for all the samples
                -> N_epochs [integer] number of epochs to perform (default is 25)
                -> mini_batch_size [integer] size of mini_batch (default is 100)
                -> eta [float] learning rate to use (default is 0.1)

        OUTPUT : -> losses [torch.Tensor] losses over the epochs
     '''
     #<*!>
     # To store de sum loss at each epoch
     losses = torch.empty(N_epochs)

     for e in range(N_epochs):
         sumloss = 0.0
         for b in range(0,input.size(0),mini_batch_size):
             # forward pass
             output = model.forward(input[b:b+mini_batch_size])
             # get the loss
             loss = criterion.forward(output, target[b:b+mini_batch_size])
             sumloss += loss.item()
             # set gradient to zero
             optimizer.zero_grad()
             # do the backward
             dl = criterion.backward(output, target[b:b+mini_batch_size])
             model.backward(dl)
             # perform the gradient descent
             optimizer.gradient_step()

         losses[e] = sumloss

     return losses

def compute_accuracy(pred, target):
    # <*> DOCSTRING
    '''
        Compute the accuracy of a model with some data.

        INPUT : -> pred [torch.Tensor] tensor of prediction outputed by the model
                   (as class and not hot_label)
                -> target [torch.Tensor] tensor of target class

        OUTPUT : -> accuracy [float] the accuracy associated
    '''
    # <*!>
    # compute number of correctly predicted
    n_correct = (pred == target.long()).sum().item()
    return n_correct/pred.size(0)

def plot_prediction(input, pred, target, ax, title, misclassified=True, legend=True, legend_pos=(0.5,-0.15)):
    #<*> DOCSTRING
    '''
        Plot the model's predictions

        INPUT : -> input [torch.Tensor] with the input data
                -> pred [torch.Tensor] with the class predictions
                -> target [torch.Tensor] with the target classes
                -> ax [pyplot.Axes] axes in which to add the plot
                -> title [string] title of the plot
                -> misclassified [bool] states whether plotting misclassified samples (default is True)

        OUTPUT : none
    '''
    #<*!>
    ax.plot(input[pred==0,0].numpy(), input[pred==0,1].numpy(), \
            linewidth=0, markersize=10, marker='.', \
            markerfacecolor='lightgreen', markeredgewidth=0, \
            label='predicted Outside')
    ax.plot(input[pred==1,0].numpy(), input[pred==1,1].numpy(), \
            linewidth=0, markersize=10, marker='.', \
            markerfacecolor='cornflowerblue', markeredgewidth=0, \
            label='predicted Inside')
    if misclassified:
        ax.plot(input[pred!=target.long(), 0].numpy(), input[pred!=target.long(), 1].numpy(), \
                linewidth=0, markersize=10, marker='.', markeredgecolor='tomato', \
                markeredgewidth=1.8, markerfacecolor='None', label='misclassified')

    ax.add_artist(plt.Circle((0, 0), 1/math.sqrt(2*math.pi), clip_on=True, edgecolor='gray', linewidth=3, fill=False, label='Boundary'))
    if legend:
        ax.legend(loc='lower center', bbox_to_anchor=legend_pos, ncol=3)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_title(title)
    ax.set_ylabel('y')
    ax.set_xlabel('x')
