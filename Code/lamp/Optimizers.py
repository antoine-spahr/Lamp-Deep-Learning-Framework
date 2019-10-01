import torch
import math

#<*> MOTHER CLASS
class Optimizer:
    ''' Mother class defining the optimizer performing the gradient descent.
        The method defining the gradient step has to be defined in the daugther
        classes'''
    def __init__(self, params):
        '''
            INPUT : params [list of list of torch.Tensors] list of the model
                    parameters. This list can be obtained from the lamp.Sequential
                    object method parameters()

            OUTPUT : None
        '''
        # The model's parameters
        self.parameters = params
        # to store the previous gradient steps for the momentum computation
        self.previous_update = [0]*len(params)

    def zero_grad(self):
        '''
            set model gradient accumulation to zero

            INPUT : None

            OUTPUT : None
        '''
        for p in self.parameters:
            # set the gradient accumulation to zero
            p[1].fill_(0)

    def gradient_step(self):
        '''
            Perform the gradient update of the model -> TO BE REDEFINE in daughter class

            INPUT : None

            OUTPUT : None
        '''
        raise NotImplementedError
#<*!>

#<*> OPTIMIZER
class SGD(Optimizer):
    ''' Stochastic gradient descent optimizer '''
    def __init__(self, params, lr=0.05, momentum=False, a=0.5):
        '''
            Define the optimizer to perform the Stochastic Gradient Descent.

            INPUT : -> params [list of list of torch.Tensors] list of the model parameters
                    -> lr [float] learning rate (default is 0.05)
                    -> momentum [bool] states whether the momentum must be taken into account or not (default is False)
                    -> a [float] the weight of the momentum in the next update (default is 0.5)

            OUTPUT : None
        '''
        Optimizer.__init__(self, params)
        self.lr = lr
        self.momentum = momentum
        self.a = a

    def gradient_step(self):
        '''
            Perform the gradient descent according to the SGD method (with or without the momentum)

            INPUT : None

            OUTPUT : None
        '''
        for idx, p in enumerate(self.parameters):
            # p[0] -> w and p[1] -> dw
            if self.momentum:
                dw = self.a*self.previous_update[idx] - self.lr*p[1] # dw = a*dw - lr*grad(w)
                self.previous_update[idx] = dw
                p[0] += dw # w = w + dw
            else:
                p[0] -= self.lr*p[1] # w = w - lr*grad(w)

        self.previous_params = self.parameters

class Adam(Optimizer):
    ''' Stochastic Gradient Descent with the Adam algorithm '''
    def __init__(self, params, lr=0.05, beta1=0.9, beta2=0.999):
        '''
            Define the optimizer to perform the Stochastic Gradient Descent with the Adam algorithm.

            INPUT : -> params [list of list of torch.Tensors] list of the model parameters
                    -> lr [float] learning rate (default is 0.05)
                    -> beta1 [float] a parameter of the Adam algorithm(default is 0.9)
                    -> beta2 [float] a parameter of the Adam algorithm (default is 0.999)

            OUTPUT : None
        '''
        Optimizer.__init__(self, params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        # to keep previous moment & velocities
        self.m = [0]*len(params)
        self.v = [0]*len(params)

    def gradient_step(self):
        '''
            Perform the gradient descent according to the Adam algorithm

            INPUT : None

            OUTPUT : None
        '''
        for idx, p in enumerate(self.parameters):
            self.m[idx] = self.beta1*self.m[idx] + (1-self.beta2)*p[1]
            m_avg = self.m[idx] / (1-self.beta1)

            self.v[idx] = self.beta2*self.v[idx] + (1-self.beta2)*p[1].pow(2)
            v_avg = self.v[idx] / (1-self.beta2)

            # do the parameters update
            p[0] -= self.lr/(v_avg.sqrt()+1e-9)*m_avg
#<*!>
