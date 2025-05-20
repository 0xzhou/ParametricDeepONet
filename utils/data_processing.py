import numpy as np
import random
import torch

duffing_normal_config = {'k': {'min': 0, 'max': 25}, 
                            'c': {'min': 0, 'max': 0.5},
                         }

def prepare_duffing_data(data, mode = 'joint', device = 'cpu', batch_size = 32, params_label = ['k', 'c'], normalization = False):
    
    para_data = data['para_data']
    resp_data = data['resp_data']
    input_data = data['force_data']
    ts = data['time_step']
    
    input = torch.from_numpy(input_data).float().to(device)
    para = torch.from_numpy(para_data).float().to(device)
    output = torch.from_numpy(resp_data).float().to(device)
    
    output = output[:,:,2]
    output = output.unsqueeze(2)
    
    ts = torch.from_numpy(ts).float().to(device)
    
    ### Varying k, c
    if params_label == ['k', 'c'] or params_label == ['k', 'c', 'A']:
        if normalization:
            print('Normalizing k, c')
            k = normalize(para[:,0], duffing_normal_config['k']['min'], duffing_normal_config['k']['max'])
            c = normalize(para[:,2], duffing_normal_config['c']['min'], duffing_normal_config['c']['max'])
            para = torch.cat((k.unsqueeze(1), c.unsqueeze(1)), dim = 1)
        else:
            para = torch.cat((para[:,0].unsqueeze(1),para[:,2].unsqueeze(1)), dim = 1)
    
    elif params_label == ['k', 'kn', 'c', 'A']:
        para = torch.cat((para[:,0].unsqueeze(1),para[:,1].unsqueeze(1), para[:,2].unsqueeze(1)), dim = 1)
    elif params_label == ['k', 'kn', 'c', 'omega', 'A']:
        para = torch.cat((para[:,0].unsqueeze(1),para[:,1].unsqueeze(1), para[:,2].unsqueeze(1)), dim = 1)
    
    if mode == 'train':
        tuple_train_data = (input, para, output, ts)
        train_data = torch.utils.data.TensorDataset(input, para, output, ts)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True, generator=torch.Generator(device = device)) 
        return tuple_train_data, train_loader
    
    elif mode == 'test':
        tuple_test_data = (input, para, output, ts)
        test_data = torch.utils.data.TensorDataset(input, para, output, ts)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle=False, generator=torch.Generator(device = device))
        return tuple_test_data, test_loader
    
    elif mode == 'joint':
        n_train = int(0.8 * para_data.shape[0])
        
        input_train = input_data[:n_train,:,0]
        input_test  = input_data[n_train:,:,0]
        
        para_train = para_data[:n_train,:]
        para_test = para_data[n_train:,:]
        
        output_train = resp_data[:n_train,:,0]
        output_test = resp_data[n_train:,:,0]
        
        ts_train = ts[:n_train]
        ts_test = ts[n_train:]
        
        train_data = torch.utils.data.TensorDataset(input_train, para_train, output_train, ts_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, shuffle=True, generator=torch.Generator(device = device)) 
        
        test_data = (input_test, para_test, output_test, ts_test)

        return train_loader, test_data
    
def merge_id_ood_test_data(id_test_data, ood_test_data):
    
    input_id, para_id, output_id, ts_id = id_test_data
    input_ood, para_ood, output_ood, ts_ood = ood_test_data
    
    input_data = torch.cat((input_id, input_ood), dim = 0)
    para_data = torch.cat((para_id, para_ood), dim = 0)
    output_data = torch.cat((output_id, output_ood), dim = 0)
    ts_data = torch.cat((ts_id, ts_ood), dim = 0)
    
    return input_data, para_data, output_data, ts_data


def denormalize_params(param, params_label = ['k', 'c']):
    if params_label == ['k', 'c']:
        k = denormalize(param[:,0], duffing_normal_config['k']['min'], duffing_normal_config['k']['max'])
        c = denormalize(param[:,1], duffing_normal_config['c']['min'], duffing_normal_config['c']['max'])
        param = torch.cat((k.unsqueeze(1), c.unsqueeze(1)), dim = 1)
    return param

def denormalize(v, v_min, v_max):
    return v * (v_max - v_min) + v_min
    
def normalize(v, v_min, v_max):
    return (v - v_min) / (v_max - v_min)


def prepare_cell_data(pkl_data_dir):
    pass

