import torch
import torch.nn as nn
import deepxde as dde
import sys
sys.path.append('../')
from models import cnn_config


class cnn_encoder(nn.Module):
    def __init__(self, config = cnn_config.cnn_encoder_config['branch_cnn_1'], pooling = False, vis_shape = False):
        super(cnn_encoder, self).__init__()
        
        self.num_layers = config['num_layers']
        self.input_size_list = config['input_size_list']
        self.hidden_size_list = config['hidden_size_list']
        self.conv_kernel_size_list = config['conv_kernel_size_list']
        self.conv_stride_list = config['conv_stride_list']
        self.padding_list = config['padding_list']
        self.vis_shape = vis_shape
        
        self.filter_convs = nn.ModuleList()
        
        for i in range(self.num_layers):
            if i != self.num_layers - 1: 
                self.filter_convs.append(nn.Conv1d(in_channels=self.input_size_list[i],
                                                out_channels=self.hidden_size_list[i],
                                                kernel_size=self.conv_kernel_size_list[i],
                                                stride=self.conv_stride_list[i],padding=self.padding_list[i]))
                
                self.filter_convs.append(nn.BatchNorm1d(self.hidden_size_list[i]))
                self.filter_convs.append(nn.LeakyReLU(0.1))
            
            else:
                self.filter_convs.append(nn.Conv1d(in_channels=self.input_size_list[i],
                                                   out_channels=self.hidden_size_list[i],
                                                   kernel_size=self.conv_kernel_size_list[i],
                                                   stride=self.conv_stride_list[i],padding=self.padding_list[i]))
    
            if pooling:
                self.filter_convs.append(nn.MaxPool1d(kernel_size=config['pool_kernel_size_list'][i], stride=config['pool_stride_list'][i]))
                
                
    def forward(self, x):
        for layer in self.filter_convs:
            x = layer(x)
            if self.vis_shape:
                print(x.shape)
        return x
    
class cnn_decoder(nn.Module):
    def __init__(self, config = cnn_config.cnn_decoder_config['trunk_decnn_1'], vis_shape = False):
        super(cnn_decoder, self).__init__()
        
        self.num_layers = config['num_layers']
        self.input_size_list = config['input_size_list']
        self.hidden_size_list = config['hidden_size_list']
        self.conv_kernel_size_list = config['conv_kernel_size_list']
        self.conv_stride_list = config['conv_stride_list']
        self.padding_list = config['padding_list']
        self.output_padding_list = config['output_padding_list']
        self.vis_shape = vis_shape
        
        self.filter_convs = nn.ModuleList()
        
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                self.filter_convs.append(nn.ConvTranspose1d(in_channels=self.input_size_list[i],
                                                            out_channels=self.hidden_size_list[i],
                                                            kernel_size=self.conv_kernel_size_list[i],
                                                            stride=self.conv_stride_list[i],
                                                            padding=self.padding_list[i],
                                                            output_padding=self.output_padding_list[i]))
                self.filter_convs.append(nn.BatchNorm1d(self.hidden_size_list[i]))
                self.filter_convs.append(nn.LeakyReLU(0.1))
             
            else:
                self.filter_convs.append(nn.ConvTranspose1d(in_channels=self.input_size_list[i],
                                                            out_channels=self.hidden_size_list[i],
                                                            kernel_size=self.conv_kernel_size_list[i],
                                                            stride=self.conv_stride_list[i],
                                                            padding=self.padding_list[i],
                                                            output_padding=self.output_padding_list[i]))
                
    def forward(self, x):
        for layer in self.filter_convs:
            x = layer(x)
            if self.vis_shape:
                print(x.shape)
        return x

class ParamsCNN(nn.Module):
    def __init__(self, x_t_shape, params_dim, case = 'blade'):
        super().__init__()
        self.case = case
        self.model_name = 'ParamsCNN'
    
        if case == 'blade':
            self.encoder = cnn_encoder(config=cnn_config.cnn_encoder_config['branch_cnn_8'])
            self.decoder = cnn_decoder(config=cnn_config.cnn_decoder_config['trunk_decnn_7'])
        elif case == 'duffing':
            self.encoder = cnn_encoder(config=cnn_config.cnn_encoder_config['case1_cnn_1'])
            self.decoder = cnn_decoder(config=cnn_config.cnn_decoder_config['case1_decnn_1'])
        
    def forward(self, x_t, mu, y_loc):
        if self.case == 'blade': 
            mu = mu.unsqueeze(1) 
            inputs = torch.cat((x_t, mu), dim = 2) # output: [batch_size, time_step, input_shape + params_dim]
            latent = self.encoder(inputs) 
            output = self.decoder(latent) 
            return output
        
        elif self.case == 'duffing':
            mu = mu.unsqueeze(1)
            x_t = x_t.unsqueeze(1)
            inputs = torch.cat((x_t, mu), dim = 2)
            latent = self.encoder(inputs) 
            output = self.decoder(latent)
            output = output.swapaxes(1,2)
            return output