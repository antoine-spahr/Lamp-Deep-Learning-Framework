import torch
import math

# <*> MOTHER CLASS
class Module:
    '''
        Mother class for any module usable to build a sequential neural network.
        It defines the functions that a module must contains (forward, backward,
        and params)
    '''
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
# <*!>

# <*> ACTIVATION FUNCTIONS
class ActivationFunction(Module):
    ''' Define the structure for an activation function, only sigma and dsigma must be define '''
    def __init__(self):
        # weight initialitzation
        Module.__init__(self)
        self.s = None

    def forward(self, s):
        '''
            Define the forward pass through the Activation Function

            INPUT: -> s [torch.Tensor] or [list of torch.Tensor]

            OUTPUT: -> [torch.Tensor] or [list of torch.Tensor] of the ActivationFunction(s)
        '''
        self.s = s
        if type(s) == list:
            # if input is a list of torch.Tensors return a list of tensor with Activation Function applied elementwise
            x = s
            for idx, s_i in enumerate(s):
                x[idx] = self.sigma(s_i)
            return x

        elif type(s) == torch.Tensor:
            # if input is a single torch.Tensor return the a tensor with Activation Function applied elementwise
            return self.sigma(s)

        else:
            # raise error is wrong input
            raise ValueError('Invalid input for the forward pass. Must be torch.Tensor or list of torch.Tensor')

    def backward(self, dx):
        '''
            Define the backward pass through the Activation function

            INPUT: -> dx [torch.Tensor] or [list of torch.Tensor]
                   -> s [torch.Tensor] or [list of torch.Tensor]

            OUTPUT: -> [torch.Tensor] or [list of torch.Tensor] of the dl/ds = dl/dx * dActivationFunction(s)
        '''
        if type(dx) == list:
            # if input is a list of torch.Tensors return a list of tensor with dActivationFunction applied elementwise
            ds = dx
            for idx, dx_i in enumerate(dx):
                ds[idx] = dx_i.t().mul(self.dsigma(self.s[idx]))
            return ds

        elif type(dx) == torch.Tensor:
            # if input is a single torch.Tensor return the a tensor with dActivation Function applied elementwise
            return dx.t().mul(self.dsigma(self.s))

        else:
            # raise error is wrong input
            raise ValueError('Invalid input for the backward pass. Must be torch.Tensor or list of torch.Tensor')

    def sigma(self, x):
        '''
            Define the activation function

            INPUT : -> x [torch.Tensor] the tensor to which Activation function is applied

            OUTPUT : -> y [torch.Tensor] tensor with activation function applied elementwise to x
        '''
        raise NotImplementedError

    def dsigma(self, x):
        '''
            Define the derivative of the activation function

            INPUT : -> x [torch.Tensor] the tensor to which derivative of Activation function is applied

            OUTPUT : -> y [torch.Tensor] tensor with the derivative of the activation function applied elementwise to x
        '''
        raise NotImplementedError

class SELU(ActivationFunction):
    ''' Define the SELU activation function '''
    def __init__(self):
        '''
            Define SELU parameters

            INPUT : None

            OUTPUT : None
        '''
        ActivationFunction.__init__(self)
        self.l = 1.0507
        self.a = 1.67326

    def sigma(self, x):
        '''
            Define the SELU function

            INPUT : -> x [torch.Tensor] the tensor to which SELU is applied

            OUTPUT : -> y [torch.Tensor] tensor with SELU(x) values
        '''
        return self.l*torch.where(x < 0, self.a*(torch.exp(x)-1), x)

    def dsigma(self, x):
        '''
            Define the derivative of SELU function

            INPUT : -> x [torch.Tensor] the tensor to which dSELU is applied

            OUTPUT : -> y [torch.Tensor] tensor with dSELU(x) values
        '''
        return self.l*torch.where(x < 0, self.a*torch.exp(x), torch.empty(x.size()).fill_(1))

class ReLU(ActivationFunction):
    ''' Define the ReLU activation function '''
    def __init__(self):
        '''
            Define ReLU parameters

            INPUT : None

            OUTPUT : None
        '''
        ActivationFunction.__init__(self)

    def sigma(self, x):
        '''
            Define the ReLU function

            INPUT : -> x [torch.Tensor] the tensor to which ReLU is applied

            OUTPUT : -> y [torch.Tensor] tensor with ReLU(x) values
        '''
        return torch.max(torch.zeros(x.size()),x)

    def dsigma(self, x):
        '''
            Define the derivative of SELU function

            INPUT : -> x [torch.Tensor] the tensor to which dReLU is applied

            OUTPUT : -> y [torch.Tensor] tensor with dReLU(x) values
        '''
        return (x > 0).float()

class Tanh(ActivationFunction):
    ''' Define the hyperbolic tangent activation function '''
    def __init__(self):
        '''
            Define tanh parameters

            INPUT : None

            OUTPUT : None
        '''
        ActivationFunction.__init__(self)

    def sigma(self, x):
        '''
            Define the tanh function

            INPUT : -> x [torch.Tensor] the tensor to which tanh is applied

            OUTPUT : -> y [torch.Tensor] tensor with tanh(x) values
        '''
        return x.tanh()

    def dsigma(self, x):
        '''
            Define the derivative of tanh function

            INPUT : -> x [torch.Tensor] the tensor to which dtanh is applied

            OUTPUT : -> y [torch.Tensor] tensor with dtanh(x) values
        '''
        return 1-x.tanh().pow(2)
# <*!>

# <*> LOSS CRITERION
class Loss(Module):
    '''
        Define the loss function.
    '''
    def __init__(self):
        '''
            Define the loss function

            INPUT : None

            OUTPUT : None
        '''
        Module.__init__(self)

    def forward(self, x, t):
        '''
            Define the forward pass through the loss function

            INPUT : -> x [torch.Tensor] or [list of torch.Tensor] the predicted class
                    -> t [torch.Tensor] or [list of torch.Tensor] the target class

            OUTPUT : -> loss [integer] or [list of integer] the loss function evaluated in x,t
        '''
        if (type(x) == list) & (type(x) == list):
            # if input is a list of torch.Tensors return a list of loss
            l = []
            for idx in range(len(x)):
                l.append(self.loss(x[idx], t[idx]))
            return l

        elif (type(x) == torch.Tensor) & (type(x) == torch.Tensor):
            # if input is a single torch.Tensor return the loss
            return self.loss(x,t)

        else:
            # raise error is wrong input
            raise ValueError('Invalid input for the backward pass. Must be torch.Tensor or list of torch.Tensor')

    def backward(self, x, t):
        '''
            Define the backward pass through the loss function

            INPUT : -> x [torch.Tensor] or [list of torch.Tensor] the predicted class
                    -> t [torch.Tensor] or [list of torch.Tensor] the target class

            OUTPUT : -> loss [torch.Tensor] or [list of torch.Tensor] the derivative of the loss function with respect to x
        '''
        if (type(x) == list) & (type(x) == list):
            # if input is a list of torch.Tensors return a list of loss
            l = []
            for idx in range(len(x)):
                l.append(self.dloss(x[idx], t[idx]))
            return l

        elif (type(x) == torch.Tensor) & (type(x) == torch.Tensor):
            # if input is a single torch.Tensor return the loss
            return self.dloss(x, t)

        else:
            # raise error is wrong input
            raise ValueError('Invalid input for the backward pass. Must be torch.Tensor or list of torch.Tensor')

    def loss(self, x, t):
        '''
            The loss function to be applied
            Need to be defined in daugther classes

            INPUT : -> x [torch.Tensor] the predicted class
                    -> t [torch.Tensor] the target class (as label or one_hot_label depending on the loss definition)

            OUTPUT : -> loss [torch.Tensor] the loss function with respect to x
        '''
        raise NotImplementedError

    def dloss(self, x, t):
        '''
            The derivative of loss function with respect to x to be applied
            Need to be defined in daugther classes

            INPUT : -> x [torch.Tensor] the predicted class
                    -> t [torch.Tensor] the target class (as label or one_hot_label depending on the loss definition)

            OUTPUT : -> dloss [torch.Tensor] the derivative of the loss function with respect to x
        '''
        raise NotImplementedError

class LossMSE(Loss):
    '''
        Define the Mean Square Error loss criterion (L2)
    '''
    def __init__(self):
        '''
            Constructor

            INPUT : None

            OUTPUT : None
        '''
        Loss.__init__(self)

    def loss(self, x, t):
        '''
            compute the MSE loss : MSE(x,t) = ∑(xi-ti)^2

            INPUT : -> x [torch.Tensor] the predicted class
                    -> t [torch.Tensor]  the target class

            OUTPUT : -> loss [integer] the MSE
        '''
        return (x-t).pow(2).sum()

    def dloss(self, x, t):
        '''
            compute the derivative of the MSE loss :
            dMSE(x,t)/dx = 2∑(xi-ti)

            INPUT : -> x [torch.Tensor] the predicted class
                    -> t [torch.Tensor] the target class

            OUTPUT : -> loss [torch.Tensor] the derivative of the MSE with respect to x
        '''
        return 2*(x.float()-t.float())

class LossMAE(Loss):
    '''
        Define the Mean Absolute Error loss criterion (L1)
    '''
    def __init__(self):
        '''
            Constructor

            INPUT : None

            OUTPUT : None
        '''
        Loss.__init__(self)

    def loss(self, x, t):
        '''
            compute the MAE loss : MAE(x,t) = ∑|xi-ti|

            INPUT : -> x [torch.Tensor] the predicted class
                    -> t [torch.Tensor] the target class

            OUTPUT : -> loss [integer] the MAE
        '''
        return (x-t).abs().sum()

    def dloss(self, x, t):
        '''
            compute the derivative of the MAE loss :
            dMAE(x,t)/dx = -1 if x<=0 ; 1 if x > 0

            INPUT : -> x [torch.Tensor] the predicted class
                    -> t [torch.Tensor] the target class

            OUTPUT : -> loss [torch.Tensor] the derivative of the MAE with respect to x
        '''
        return torch.where((x-t) > 0, torch.empty(x.size()).fill_(1), torch.empty(x.size()).fill_(-1)) # 1 if x-t > 0, -1 else

class CrossEntropyLoss(Loss):
    '''
        Define the Cross Entropy loss
    '''
    def __init__(self):
        '''
            Constructor

            INPUT : None

            OUTPUT : None
        '''
        Loss.__init__(self)

    def loss(self, x, t):
        '''
            compute the cross-entropy loss : H(x,t) = -∑ti*log(pi) where pi = softmax(xi)

            INPUT : -> x [torch.Tensor] predicrtion of dimension n_sample x n_class
                    -> t [torch.Tensor] target of diemnsion n_sample x n_class (one_hot_encoded)

            OUTPUT : -> loss [float] the cross entropy loss over all samples
        '''
        # check if input are ok
        if len(t.size()) != 2:
             raise ValueError('Wrong dimension of the labels. Expect 2D one_hot encoded target Tensor n_sample x n_class')
        elif len(x.size()) != 2:
             raise ValueError('Wrong dimension of the prediction. Expect 2D one_hot encoded target Tensor n_sample x n_class')
        elif t.size() != x.size():
            raise ValueError('Wrong dimension of the inputs. Prediction and target must have similar 2D dimensions n_sample x n_class')

        p = self.softmax(x)
        return -torch.sum(p.log()*t)

    def dloss(self, x, t):
        '''
            compute the derivative of the cross-entropy loss :
            dH(xi,t)/dxi = pi-ti where pi = softmax(xi)

            INPUT : -> x [torch.Tensor] prediction of dimension n_sample x n_class
                    -> t [torch.Tensor] target of diemnsion n_sample x n_class (one_hot_encoded)

            OUTPUT : -> dloss [torch.Tensor] the derivative of the loss with respect to the output layer.
                        dimension is n_sample x n_class
        '''
        # check if input are ok
        if len(t.size()) != 2:
             raise ValueError('Wrong dimension of the labels. Expect 2D one_hot encoded target Tensor n_sample x n_class')
        elif len(x.size()) != 2:
             raise ValueError('Wrong dimension of the prediction. Expect 2D one_hot encoded target Tensor n_sample x n_class')
        elif t.size() != x.size():
            raise ValueError('Wrong dimension of the labels. Prediction and target must have similar dimensions n_sample x n_class')

        t = t.float()
        return self.softmax(x) - t

    def softmax(self, x):
        '''
            Get the softmax of x along dimension 1 (columns)

            INPUT : -> x [torch.Tensor] the tensor to softmax over columns.
                       it has the dimension : n_sample x n_class

            OUTPUT : -> p [torch.Tensor] same size tensor as input with the softmax
                        applied over columns for each sample (rows)
        '''
        # compute the exponential : exp(x-max(x)) ; the -max(x) enables a stable softmax (not exceeding float values)
        exps = (x - x.max(dim=1, keepdim=True)[0]).exp()
        # return softmax
        return exps / torch.sum(exps, dim=1, keepdim=True)
# <*!>

# <*> LINEAR LAYERS
class Linear(Module):
    '''
        Define a Linear (fully connected) layer.
    '''
    def __init__(self, n_in, n_out):
        '''
            Define a fully connected Linear Module, initialized with Xavier initialization

            INPUT : -> n_in [integer] number of neurons of input
                    -> n_out [integer] number of neurons of output

            OUTPUT : None
        '''
        # weight initialitzation
        Module.__init__(self)
        self.n_in = n_in
        self.n_out = n_out
        # Layer parameters
        std = math.sqrt(2/(n_in + n_out)) # Xavier initatialisation
        self.w = torch.empty(n_out, n_in).normal_(mean=0, std=std)
        self.b = torch.empty(n_out, 1).normal_(mean=0, std=std)
        self.dw = torch.empty(self.w.size()).fill_(0)
        self.db = torch.empty(self.b.size()).fill_(0)

        self.x = None # store the input value for the backward computation

    def forward(self, x0):
        '''
            Define the forward pass through the Linear layer

            INPUT : -> x0 [torch.Tensor or list of torch.Tensor] the input of the
                       layer. The tensor must have the dimension: n_sample x n_in

            OUTPUT : -> x1 [torch.Tensor or list of torch.Tensor] the output of the
                       layer. The tensor have the dimension: n_sample x n_out
        '''
        # store input for backward
        self.x = x0

        if type(x0) == list:
            # if input is a list of torch.Tensors return a list of loss
            x1 = []
            for idx, x0_i in enumerate(x0):
                # check input dimensions
                if x0_i.size(1) != self.n_in:
                    raise ValueError('Invalid input dimension! The input Tensor n°{0} must have dimension n_sample x {1}, but n_sample x {2} has been given'.format(idx, self.n_in, x0_i.size(1)))
                x1.append((self.w @ x0_i.t() + self.b).t())
            return x1

        elif type(x0) == torch.Tensor:
            # check input dimensions
            if x0.size(1) != self.n_in:
                raise ValueError('Invalid input dimension! The input Tensor must have dimension n_sample x {0}, but n_sample x {1} has been given'.format(self.n_in, x0.size(1)))
            # if input is a single torch.Tensor return the loss
            return (self.w @ x0.t() + self.b).t()

        else:
            raise ValueError('Invalid input for the backward pass. Must be torch.Tensor or list of torch.Tensor')

    def backward(self, ds):
        '''
            Define the backward pass through the Linear

            INPUT : -> ds [torch.Tensor] or [list of torch.Tensor] the derivative
                       of the loss with respect to s : dl/ds. The tensor must have
                       the dimension: n_sample x n_out

            OUTPUT : -> dx [torch.Tensor] or [list of torch.Tensor] the derivative
                     of the loss with respect to x_in : dl/dx_in. The tensor have
                     the dimension: n_sample x n_in
        '''
        if type(ds) == list:
            # if input is a list of torch.Tensors return a list of loss
            dx = []
            for idx, ds_i in enumerate(ds):
                if ds_i.size(1) != self.n_out:
                    raise ValueError('Invalid input dimension! The input Tensor n°{0} must have dimension n_sample x {1}, but n_sample x {2} has been given'.format(idx, self.n_out, ds_i.size(1)))
                # compute parameters gradient, accumulate it, and scaled it by the number of sample (ds.size(0))
                self.dw += (ds_i.t() @ self.x[idx])/ds_i.size(0)
                self.db += (ds_i.t().sum(dim=1).view(-1,1))/ds_i.size(0)
                dx.append(self.w.t() @ ds_i.t())
            return dx

        elif type(ds) == torch.Tensor:
            # if input is a single torch.Tensor return the loss
            # check input dimensions
            if ds.size(1) != self.n_out:
                raise ValueError('Invalid input dimension! The input Tensor must have dimension n_sample x {0}, but n_sample x {1} has been given'.format(self.n_out, ds.size(1)))
            # compute parameters gradient, accumulate it, and scaled it by the number of sample (ds.size(0))
            self.dw += (ds.t() @ self.x)/ds.size(0)
            self.db += (ds.t().sum(dim=1).view(-1,1))/ds.size(0)
            return self.w.t() @ ds.t()

        else:
            raise ValueError('Invalid input for the backward pass. Must be torch.Tensor or list of torch.Tensor')

    def reset_grad(self):
        '''
            reset the dl/dw and dl/db to zero

            INPUT : None

            OUTPUT : None
        '''
        self.dw.fill_(0)
        self.db.fill_(0)

    def param(self):
        '''
            Return the parameters and their gradient as a list of list
            first the weight (position 0) and then the bias (position 1)
            The parameter tensor is then on positon 0 of the sublist
            and the gradient is on position 1 of the sublist.

            INPUT : None

            OUTPUT : -> [list of list] parameters of the layer
        '''
        return [[self.w, self.dw],[self.b, self.db]]
# <*!>
