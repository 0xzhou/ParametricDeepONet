import torch
import torch.nn as nn
import deepxde as dde
import sys
import models
from models import cnn_config
import numpy as np


def fourier_features(x, k,  L):
    omega = 2 * torch.pi / L
    #omega = 2 * torch.pi
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
        mu = mu.unsqueeze(1) # (bs, 1, n_e)
        mu_latent = self.model(mu) # (bs, 1, n_e) -> (bs, 1, p)
        return mu_latent.squeeze(1) # (bs, 1, p) -> (bs, p)
    
class Params_CNN_Refine_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.cnn_encoder(config=cnn_config.cnn_encoder_config["params_refine_cnn_1"])
        self.fnn = dde.nn.FNN([200, 512, 512, 200], 'relu', 'Glorot normal')
    
    def forward(self, mu):
        mu = mu.unsqueeze(1) # (bs, 1, n_e)
        mu_latent = self.model(mu) # (bs, 1, n_e) -> (bs, 1, p)
        mu_latent = mu_latent.squeeze(1) # (bs, 1, p) -> (bs, p)
        output = self.fnn(mu_latent) # (bs, p) -> (bs, 200)
        return output # (bs, 1, p) -> (bs, p)


class ParametricDeepONet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.branch_net = dde.nn.FNN([self.args.input_dim] + [self.args.width] * (self.args.depth - 1), 'relu', 'Glorot normal')
        self.params_net = dde.nn.FNN([self.args.params_dim] + [self.args.width] * (self.args.depth - 1), 'relu', 'Glorot normal')
        #self.params_net = Params_CNN_Net()
        
        if self.args.data_case == 'duffing':
            self.trunk_net = dde.nn.FNN([self.args.pebasis*2] + [self.args.width] * (self.args.depth - 1), 'relu', 'Glorot normal')
        elif self.args.data_case == 'blade':
            self.trunk_net = dde.nn.FNN([self.args.pebasis*2] + [self.args.width] * (self.args.depth - 1), 'relu', 'Glorot normal')
            
            
        if self.args.decode_mode == 'linear':
            self.model_name = 'ParametricLinearDeepONet'
        elif self.args.decode_mode == 'nonlinear':
            self.model_name = 'ParametricNonlinearDeepONet'
            if self.args.data_case == 'duffing':
                #self.decoder = dde.nn.FNN([200, self.args.width, 200], 'relu', 'Glorot normal')
                self.decoder = models.cnn_encoder(config=cnn_config.cnn_encoder_config["decoder_cnn_case1"])
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
                # ### CNN
                latent_output = latent_output.unsqueeze(1)
                output = self.decoder(latent_output)
                output = output.swapaxes(1,2)
                
                ## MLP 
                # output = self.decoder(latent_output)
                # output = output.unsqueeze(2)
                #input()
            return output
            
        elif self.args.data_case == 'blade':
            x_t = x_t.squeeze(1)
            input_latent = self.branch_net(x_t) # (bs, n_i) -> (bs, p)
            params_latent = self.params_net(mu) # (bs, n_e) -> (bs, p)
            
            if resolution == 200:
                y_loc = torch.arange(0, 4 * 200 * 1.0/1666, 1.0/1666).unsqueeze(1) # for super-resolution
                #y_loc = torch.rand(200*4, 1) * 4 * 200 * 1.0/1666
                #y_loc = torch.sort(y_loc, dim = 0)[0]
                #print("In parametric_deeponet.py, y_loc.shape: ", y_loc.shape)
            else:
                y_loc = torch.arange(0, 4 * 200 * 1.0/1666, 1.0/1666 * 200/resolution ).unsqueeze(1) 
            #print("In parametric_deeponet.py, latent_output.shape: ", y_loc.shape)
            y_loc = fourier_features(y_loc, k = self.args.pebasis, L = 800 * 1.0 / 1666) # (r) -> (r, 2k) # position encoding
            y_t_latent = self.trunk_net(y_loc) # (r, 2k) -> (r, p)
            
            latent_output = torch.einsum('bp,bp,rp->br', input_latent, params_latent, y_t_latent)
            #print("In parametric_deeponet.py, latent_output.shape: ", latent_output.shape)
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
    
    def generate_latent(self, x_t, mu, y_loc):
        if self.args.data_case == 'duffing':
            y_loc = y_loc[0]
            y_loc = y_loc.unsqueeze(1) 
            weights = self.branch_net(x_t)
            basis_1 = self.params_net(mu)
            basis_2 = self.trunk_net(y_loc)
            latent = torch.einsum('bp,bp,np->bn', weights, basis_1, basis_2)
            return latent






class ParametricDeepONet2(ParametricDeepONet):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'ParametricDeepONet2'
        
        if self.args.decode_mode == 'nonlinear':
            if self.args.data_case == 'duffing':
                self.decoder = dde.nn.FNN([1, self.args.width, self.args.width, 1], 'relu', 'Glorot normal')
            # elif self.args.data_case == 'blade':
            #     self.decoder = models.cnn_encoder(config=cnn_config.cnn_encoder_config["decoder_cnn_1"])
        
    def forward(self, x_t, mu, y_loc, resolution = 200):
        y_loc = y_loc[0] # (bs, r) -> (r)
        y_loc = y_loc.unsqueeze(1) # (r) -> (r, 1)

        if self.args.data_case == 'duffing':
            input_latent = self.branch_net(x_t) # (bs, m) -> (bs, p)
            params_latent = self.params_net(mu) # (bs, n) -> (bs, p)
            
            y_loc = fourier_features(y_loc, k = self.args.pebasis, L = 2) # (r, 1) -> (r, 2k)
            eval_latent = self.trunk_net(y_loc) # (r, 1) -> (r, p)
            
            latent_output = torch.einsum('bp,bp,rp->br', input_latent, params_latent, eval_latent)
            latent_output = latent_output.unsqueeze(2)
            output = self.decoder(latent_output)
            return output
            
        elif self.args.data_case == 'blade':
            x_t = x_t.squeeze(1)
            input_latent = self.branch_net(x_t) # (bs, n_i) -> (bs, p)
            params_latent = self.params_net(mu) # (bs, n_e) -> (bs, p)
            
            if resolution == 200:
                y_loc = torch.arange(0, 4 * 200 * 1.0/1666, 1.0/1666).unsqueeze(1) # for super-resolution
            else:
                y_loc = torch.arange(0, 4 * 200 * 1.0/1666 *  200/resolution, 1.0/1666 * 200/resolution ).unsqueeze(1) 
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
        
class ParametricDeepONet3(ParametricDeepONet):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'ParametricDeepONet3'
        
        self.branch_net = dde.nn.FNN([self.args.input_dim] + [self.args.width] * (self.args.depth - 1), 'relu', 'Glorot normal')
        
        
    def forward(self, x_t, mu, y_loc, resolution = 200):
        y_loc = y_loc[0] # (bs, r) -> (r)
        y_loc = y_loc.unsqueeze(1) # (r) -> (r, 1)
         
        if self.args.data_case == 'blade':
            x_t = x_t.squeeze(1)
            input_latent = self.branch_net(x_t) # (bs, n_i) -> (bs, p)
            params_latent = self.params_net(mu) # (bs, n_e) -> (bs, p)
            
            if resolution == 200:
                y_loc = torch.arange(0, 4 * 200 * 1.0/1666, 1.0/1666).unsqueeze(1) # for super-resolution
            else:
                y_loc = torch.arange(0, 4 * 200 * 1.0/1666 *  200/resolution, 1.0/1666 * 200/resolution ).unsqueeze(1) 
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
        

        
        
