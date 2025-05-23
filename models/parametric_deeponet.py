import torch
import torch.nn as nn
import deepxde as dde
import sys
import models
from models import cnn_config
import numpy as np


def fourier_features(x, k,  L):
    omega = 2 * torch.pi / L
    a = torch.arange(1, k + 1) * omega
    return torch.cat([torch.sin(a * x), torch.cos(a * x)], dim=-1)

def random_fourier_features(x, k, d = 1):
    B = torch.randn(k, d) # (k, d)
    x = 2 * torch.pi * x @ B.T # (,d) @ (d, k) -> (,k)
    return torch.cat([torch.cos(x),torch.sin(x)], dim=-1)


class Params_CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.cnn_encoder(config=cnn_config.cnn_encoder_config["params_cnn_1"])
    
    def forward(self, mu):
        mu = mu.unsqueeze(1) 
        mu_latent = self.model(mu) 
        return mu_latent.squeeze(1) 

    
class ParametricDeepONet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.branch_net = dde.nn.FNN([self.args.input_dim] + [self.args.width] * (self.args.depth - 1), 'relu', 'Glorot normal')
        if self.args.params_dim == 200:
             self.params_net = Params_CNN_Net()
        else:
            self.params_net = dde.nn.FNN([self.args.params_dim] + [self.args.width] * (self.args.depth - 1), 'relu', 'Glorot normal')
        
        if self.args.data_case == 'duffing':
            self.trunk_net = dde.nn.FNN([self.args.pebasis*2] + [self.args.width] * (self.args.depth - 1), 'relu', 'Glorot normal')
        elif self.args.data_case == 'blade':
            self.trunk_net = dde.nn.FNN([self.args.pebasis*2] + [self.args.width] * (self.args.depth - 1), 'relu', 'Glorot normal')
            
            
        if self.args.decode_mode == 'linear':
            self.model_name = 'ParametricLinearDeepONet'
        elif self.args.decode_mode == 'nonlinear':
            self.model_name = 'ParametricNonlinearDeepONet'
            if self.args.data_case == 'duffing':
                self.decoder = dde.nn.FNN([200, self.args.width, 200], 'relu', 'Glorot normal')
                #self.decoder = models.cnn_encoder(config=cnn_config.cnn_encoder_config["decoder_cnn_case1"])
            elif self.args.data_case == 'blade':
                self.decoder = models.cnn_encoder(config=cnn_config.cnn_encoder_config["decoder_cnn_1"])
    
    def forward(self, x_t, mu, y_loc, resolution = 200):
        if self.args.data_case == 'duffing':
            y_loc = y_loc[0] # (bs, r) -> (r)
            y_loc = y_loc.unsqueeze(1) # (r) -> (r, 1)
            
            input_latent = self.branch_net(x_t) # (bs, m) -> (bs, p)
            params_latent = self.params_net(mu) # (bs, n) -> (bs, p)
            
            y_loc = fourier_features(y_loc, k = self.args.pebasis, L = 2) # (r, 1) -> (r, 2k)
            eval_latent = self.trunk_net(y_loc) # (r, 1) -> (r, p)
            
            latent_output = torch.einsum('bp,bp,rp->br', input_latent, params_latent, eval_latent)

            if self.args.decode_mode == 'linear':
                output = latent_output
                output = output.unsqueeze(2)
            elif self.args.decode_mode == 'nonlinear':
                ## MLP 
                output = self.decoder(latent_output)
                output = output.unsqueeze(2)
            return output
            
        elif self.args.data_case == 'blade':
            x_t = x_t.squeeze(1)
            input_latent = self.branch_net(x_t) # (bs, n_i) -> (bs, p)
            params_latent = self.params_net(mu) # (bs, n_e) -> (bs, p)
            
            if resolution == 200:
                y_loc = torch.arange(0, 4 * 200 * 1.0/1666, 1.0/1666).unsqueeze(1) 
            else:
                y_loc = torch.arange(0, 4 * 200 * 1.0/1666, 1.0/1666 * 200/resolution ).unsqueeze(1) # for super-resolution
            y_loc = fourier_features(y_loc, k = self.args.pebasis, L = 800 * 1.0 / 1666) # (r) -> (r, 2k) # position encoding
            y_t_latent = self.trunk_net(y_loc) # (r, 2k) -> (r, p)
            
            latent_output = torch.einsum('bp,bp,rp->br', input_latent, params_latent, y_t_latent)
            if self.args.decode_mode == 'linear':
                output = latent_output
                output = output.reshape(latent_output.shape[0], 4, -1)
            elif self.args.decode_mode == 'nonlinear':
                latent_output = latent_output.reshape(latent_output.shape[0], 4, -1)
                output = self.decoder(latent_output)

            return output
        
    def forward_resolution(self, x_t, mu, resolution):
        if self.args.data_case == 'duffing':
            y_loc = torch.arange(0, 4 * resolution * 1.0/1666, 1.0/1666).unsqueeze(1)
            y_loc = fourier_features(y_loc, k = 20, L = 2)
            weights = self.branch_net(x_t)
            basis_1 = self.params_net(mu)
            basis_2 = self.trunk_net(y_loc)
            latent = torch.einsum('bp,bp,np->bn', weights, basis_1, basis_2)
            return latent
    


        

        
        
