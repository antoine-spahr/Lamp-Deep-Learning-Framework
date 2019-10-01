import torch
import math
import lamp.Modules

class Sequential:
    ''' Object enabling to define a model by a combination of lamp.Module '''
    def __init__(self, *modules):
        '''
            The sequential model is defined by a list of modules.
            The order of the list matters!

            INPUT : -> *modules [multiple lamp.Modules]

            OUTPUT : None
        '''
        self.modules_list = []

        # add module to self.module_list depending if they are directly given or coming from a submodel
        for mod in modules:
            if type(mod) == Sequential:
                self.modules_list += mod.get_module_list()
            else:
                self.modules_list.append(mod)

    def forward(self, input):
        '''
            Perform the forward pass Sequentially through the modules forming
            the Sequential object

            INPUT : input [torch.Tensor] the input of the model

            OUTPUT : x [torch.Tensor] the output of the model
        '''
        x = input
        for mod in self.modules_list:
            x = mod.forward(x)
        return x

    def backward(self, dl):
        '''
            Perform the backward pass sequentially through the modules of the
            Sequential object (from the last module to the first)

            INPUT : dl [torch.Tensor] the derivative of the loss with respect to the output

            OUTPUT : None
        '''
        for mod in reversed(self.modules_list):
            dl = mod.backward(dl)

    def zero_grad(self):
        '''
            reset the parameters derivative to zero

            INPUT : None

            OUTPUT : None
        '''
        for mod in self.modules_list:
            mod.reset_grad()

    def get_module_list(self):
        '''
            return the list of modules composing the model

            INPUT : None

            OUTPUT : -> self.modules_list [list of lamp.Module]
        '''
        return self.modules_list

    def parameters(self):
        '''
            retrun a list of all the models parameters
            the list retruned contains tuples (Wweight, dWeight) with the parameters
            in position 0 and the gradient in position 1.

            INPUT : None

            OUTPUT : params [list of list of torch.Tensor] the parameters and the associated
                     gradient of the model's parameters
        '''
        params = []
        for mod in self.modules_list:
            params += mod.param()
        return params
