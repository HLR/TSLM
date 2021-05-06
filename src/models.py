import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import string
from os import path

class LSTMFlair(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=1, batch_size=1, bidirectional=False):
        super(LSTMFlair, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.is_cuda = torch.cuda.is_available()
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, bidirectional=bidirectional)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        i = 1
        if self.bidirectional:
            i = 2
        return (torch.zeros(self.num_layers*i, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers*i, self.batch_size, self.hidden_dim))

    def forward(self, input):
        lstm_out, share_hidden = self.lstm(input)
        return lstm_out

class LSTMFlair1(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(LSTMFlair1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
#         self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.is_cuda = torch.cuda.is_available()
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, bidirectional=bidirectional, batch_first=True)

    def init_hidden(self, batch_size):
        # This is what we'll initialise our hidden state as
        i = 1
        if self.bidirectional:
            i = 2
        return (torch.zeros(self.num_layers*i, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers*i, batch_size, self.hidden_dim))

    def forward(self, input):
        h_t, c_t = self.init_hidden(input.size(0))
        lstm_out, share_hidden = self.lstm(input, (h_t.to(next(self.parameters()).device), c_t.to(next(self.parameters()).device)))
        return lstm_out


class FullyConnected(nn.Module):
    def __init__(self, dims, layers = 1):
        super(FullyConnected, self).__init__()
        self.dims = dims
        
        # Define the fully connected layer
        layers_list = []
        for layer in range(layers):
            layers_list.append(nn.Linear(self.dims[layer], self.dims[layer+1]))
        self.layers = nn.ModuleList(layers_list)
            
    def forward(self, _input):
        fc_result = _input
        for layer in range(len(self.layers)):
            fc_result = self.layers[layer](fc_result)
            if layer != len(self.layers) - 1:
                fc_result = F.leaky_relu(fc_result)
        return fc_result
    

class FCDecider(nn.Module):
    def __init__(self, dims, layers = 1):
        super(FCDecider, self).__init__()
        self.dims = dims
        
        # Define the fully connected layer
        layers_list = []
        for layer in range(layers):
            layers_list.append(nn.Linear(self.dims[layer], self.dims[layer+1]))
        self.layers = nn.ModuleList(layers_list)
            
    def forward(self, _input):
        fc_result = _input
        for layer in range(len(self.layers)):
            fc_result = self.layers[layer](fc_result)
            if layer != len(self.layers) - 1:
                fc_result = F.leaky_relu(fc_result)
        fc_result = torch.sigmoid(fc_result)
        return fc_result


class FCGumble(nn.Module):
    def __init__(self, dims, layers = 1):
        super(FCGumble, self).__init__()
        self.dims = dims
        
        # Define the fully connected layer
        layers_list = []
        for layer in range(layers):
            layers_list.append(nn.Linear(self.dims[layer], self.dims[layer+1]))
        self.layers = nn.ModuleList(layers_list)
            
    def forward(self, _input):
        fc_result = _input
        for layer in range(len(self.layers)):
            fc_result = self.layers[layer](fc_result)
            if layer != len(self.layers) - 1:
                fc_result = F.leaky_relu(fc_result)
        fc_result = F.gumbel_softmax(fc_result)
        return fc_result

class DropFullyConnected(nn.Module):
    def __init__(self, dims, layers = 1, dropout = 0.0):
        super(DropFullyConnected, self).__init__()
        self.dims = dims
        
        # Define the fully connected layer
        layers_list = []
        for layer in range(layers):
            layers_list.append(nn.Linear(self.dims[layer], self.dims[layer+1]))
        self.layers = nn.ModuleList(layers_list)
        self.drop_layer = nn.Dropout(p=dropout)
            
    def forward(self, _input, mode="train"):
        fc_result = _input
        for layer in range(len(self.layers)):
            fc_result = self.layers[layer](fc_result)
            if layer != len(self.layers) - 1:
                fc_result = F.leaky_relu(fc_result)
                if mode == "train":
                    fc_result = self.drop_layer(fc_result)                    
        return fc_result    
    
    
class ResidualFullyConnected(nn.Module):
    def __init__(self, dims, layers = 1, p = 0.1):
        super(ResidualFullyConnected, self).__init__()
        self.dims = dims
        
        # Define the fully connected layer
        layers_list = []
        for layer in range(layers):
            layers_list.append(nn.Linear(self.dims[layer], self.dims[layer+1]))
        self.drop_layer = nn.Dropout(p=p)
        self.layers = nn.ModuleList(layers_list)
            
    def forward(self, _input):
        fc_result = _input
        _input = self.drop_layer(_input)
        out = self.layers[0](_input)
        if 0 != len(self.layers) - 1:
            out = F.leaky_relu(out)
        for layer in range(1,len(self.layers)):
            fc_result = out
            if out.shape == _input.shape:
                out = self.layers[layer](out + _input)
            else:
                out = self.layers[layer](out)
            out = F.leaky_relu(out)
            _input = fc_result
        return out


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x
    

class HighwayResidualFC(nn.Module):
    def __init__(self, size, num_layers, f, dims, layers = 1):
        super(HighwayResidualFC, self).__init__()
        self.hgw = Highway(size, num_layers, f)
        self.rfc = ResidualFullyConnected(dims, layers)
            
    def forward(self, _input):
        out = self.hgw(_input)
        out = self.rfc(_input)
        return out
    
    
class HighwayFC(nn.Module):
    def __init__(self, size, num_layers, f, dims, layers = 1):
        super(HighwayFC, self).__init__()
        self.hgw = Highway(size, num_layers, f)
        self.rfc = FullyConnected(dims, layers)
            
    def forward(self, _input):
        out = self.hgw(_input)
        out = self.rfc(_input)
        return out
