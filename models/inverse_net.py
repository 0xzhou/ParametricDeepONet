import torch
import torch.nn as nn
import deepxde as dde

class GradientInitialNet(nn.Module):
    ### In this net, the trainable parameters the system parameters with random initializations
    
    def __init__(self, batch_size, forward_net = nn.Module, case = 'case1'):
        super(GradientInitialNet, self).__init__()
        
        ### Fix forward model
        self.forward_net = forward_net
        for param in self.forward_net.parameters(): 
            param.requires_grad = False # fix the parameters of forward net
        self.forward_net.eval()
        
        ### Define the trainable parameters
        if case == 'case1': # The forced-duffing osicallator
            k = torch.FloatTensor(batch_size, 1).uniform_(10, 100)
            c = torch.FloatTensor(batch_size, 1).uniform_(1, 10)
            params = torch.cat((k, c), 1)
            
        #elif case == 'case2': # wind turbine blade / parameterization (a)            
            # self.mu_1 = torch.FloatTensor(batch_size, 1).uniform_(0, 1)
            # self.mu_2 = torch.FloatTensor(batch_size, 1).uniform_(0, 1)
            # self.mu_3 = torch.FloatTensor(batch_size, 1).uniform_(0, 1)
            # params = torch.cat([self.mu_1, self.mu_2, self.mu_3], dim = 1)
            
        elif case == 'case2': # wind turbine blade / parameterization (b)            
            params = torch.FloatTensor(batch_size, 200).uniform_(0, 1)
        
        self.params = nn.Parameter(params) # set initial params as trainable parameters
        
    def forward(self, x, t):
        return self.forward_net(x, self.params, t)
    
class InitialRefineNet(nn.Module):
    def __init__(self, case = 'case1'):
        super(InitialRefineNet, self).__init__()
        if case == 'case1':
            self.gradient_nn = dde.nn.FNN([(5 + 2) * 6 , 128, 128, (5+1) * 6], 'relu', 'Glorot normal')
        elif case == 'case2':
            self.gradient_nn = dde.nn.FNN([(5 + 2) * 6 , 128, 128, (5+1) * 6], 'relu', 'Glorot normal')
        
    def estimate_learned_gradient(self, memory_dydmu, dydmu, previous_mu_hat):
        
        if type(memory_dydmu) == torch.nn.parameter.Parameter or type(memory_dydmu) == torch.Tensor: # with ierative scheme
            inputs = torch.cat([memory_dydmu, dydmu, previous_mu_hat], dim = 1)
            outputs = self.gradient_nn(inputs)
            dydmu = outputs[:, memory_dydmu.shape[1]:]
            memory_dydmu = outputs[:, :memory_dydmu.shape[1]]
            memory_dydmu = nn.functional.relu(memory_dydmu)
            return dydmu, memory_dydmu
        
        else: # Directly estimate the gradient w/o iterative scheme
            inputs = torch.cat([dydmu, previous_mu_hat], dim = 1)
            outputs = self.gradient_nn(inputs)
            dydmu = outputs[:]
            return dydmu, memory_dydmu
    
