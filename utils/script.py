import sys
sys.path.append('/home/zmy2022/ShareFolder/Github/neural-PGD/blade_data')
import torch
from utils.blade_preprocess import ProcessedSampleSet
import numpy as np
import pickle

y_length = 0.2
damage_label_dic = {
    "R": [0.17, 0.3, 0.5, 0, 0, 0],  # position of damage, three params; length of damage, three params
    "D": [0.17, 0.3, 0.5, 0.05 / y_length, 0, 0],
    "E": [0.17, 0.3, 0.5, 0.05 / y_length, 0.05 / y_length, 0],
    "F": [0.17, 0.3, 0.5, 0.05 / y_length, 0.05 / y_length, 0.05 / y_length],
    "G": [0.17, 0.3, 0.5, 0.1 / y_length, 0.05 / y_length, 0.05 / y_length],
    "H": [0.17, 0.3, 0.5, 0.1 / y_length, 0.1 / y_length, 0.05 / y_length],
    "I": [0.17, 0.3, 0.5, 0.1 / y_length, 0.1 / y_length, 0.1 / y_length],
    "J": [0.17, 0.3, 0.5, 0.15 / y_length, 0.1 / y_length, 0.1 / y_length],
    "K": [0.17, 0.3, 0.5, 0.15 / y_length, 0.15 / y_length, 0.1 / y_length],
    "L": [0.17, 0.3, 0.5, 0.15 / y_length, 0.15 / y_length, 0.15 / y_length],
}

normalization = {
    'params': { 'location': {'min': 0, 'max': 0.5},
                'length': {'min': 0, 'max': 15},
        }
}

def normalize(data, min_data, max_data):
    return (data - min_data) / (max_data - min_data)

def get_params(sample_list):
    params = []
    for i, sample in enumerate(sample_list):
        params.append(damage_label_dic[sample['w_label']])
    params = np.array(params)
    
    ### damage length parameterization (a)
    #params = params[:, 3:] 
    
    ### damage shape parameterization (b)
    params = vectorize_params(params)
    
    return params


def generate_param_vector(mu, sigma, length):
    x = np.linspace(0, 1, length)
    y = np.exp(-(x-mu)**2/(2*sigma**2))
    return x, y

def vectorize_params(params):
    vectorized_params = []
    
    for i in range(params.shape[0]):
        y_sum = 0
        for j in range(3):
            mu = params[i, j]
            y_scale = params[i, j+3]
            
            x, y = generate_param_vector(mu, 0.01, 200)
            y *= y_scale
            y_sum += y
        vectorized_params.append(y_sum)
    return np.array(vectorized_params)

def get_response(sample_list):
    response = []
    for sample in sample_list:
        response.append(np.concatenate((sample['ts'][:,0, np.newaxis], sample['ts'][:,1, np.newaxis], sample['ts'][:,2, np.newaxis], sample['ts'][:,3, np.newaxis]), axis = 1))
    return np.array(response)

def get_source(sample_list):
    source = []
    for sample in sample_list:
        source.append(sample['ts'][:,-1])
    return np.array(source)

def normalize_signal(data, normalize_config = {}, mode = 'self'):
    # Normalize the input force signal & output response signal
    print("Normalizing the data mode is {}...".format(mode))
    if mode == 'self_min_max': # normalize the data to [-1, 1]
        if normalize_config:
            min_data = normalize_config['min']
            max_data = normalize_config['max']
        else:
            min_data = torch.min(data)
            max_data = torch.max(data)
            
        data_range = max_data - min_data
        data = -1 + 2 * (data - min_data) / data_range
        
    if mode == 'min_max':
        mean_per_channel = torch.mean(data, axis=2, keepdims=True)
        data_zero_meaned = data - mean_per_channel
        
        min_per_channel = torch.min(data_zero_meaned, axis=2, keepdims=True)[0]
        max_per_channel = torch.max(data_zero_meaned, axis=2, keepdims=True)[0]
        data = (data_zero_meaned - min_per_channel) / (max_per_channel - min_per_channel)
        
    elif mode == 'z_score':
        mean_data = np.mean(data)
        std_data = np.std(data)
        data = (data - mean_data) / std_data
        
    return data

def get_range(data):
    return np.min(data), np.max(data)

def prepare_blade_dataset(pkl_data_dir = None, key='train', batch_size = 64, device='cpu', normalize_config = {}, noised = False):
    if not noised:
        # Load the original .pkl data, without noise data augmentation
        sample_set = ProcessedSampleSet().load_from(pkl_data_dir) # {}: data['train'], data['test'], data['val']
        data = sample_set.data[key]
        
        response_data = get_response(data)
        params_data = get_params(data)
        input_data = get_source(data)
        
        number_of_samples = input_data.shape[0]
        ts_step = 1/1666 # sampling frequency
        ts = np.arange(0, 200*ts_step, ts_step)
        ts = np.tile(ts, (number_of_samples, 1))
        
        source_signal = torch.from_numpy(input_data).float().to(device)
        source_signal = source_signal.unsqueeze(1) # adapt to the input shape of CNN
        params_data = torch.from_numpy(params_data).float().to(device)
        response_data = torch.from_numpy(response_data).float().to(device)
        response_data = response_data.permute(0, 2, 1)
        ts = torch.from_numpy(ts).float().to(device)
        
        output_normalize_config = {}
        if key == 'train':
            output_normalize_config = {'input': {'min': torch.min(response_data), 'max': torch.max(response_data)},
                                        'output': {'min': torch.min(response_data), 'max': torch.max(response_data)}}
            response_data = normalize_signal(response_data, {}, mode = 'self_min_max')

            print("Train data is normalized by the train data...")
        else:
            print("Test data is normalized by the test data...")
            response_data = normalize_signal(response_data, {}, mode = 'self_min_max')
            
    else:
        with open(pkl_data_dir, 'rb') as f:
            data = pickle.load(f)
            source_signal, params_data, response_data, ts = data
            output_normalize_config = {}
    
    print("The shape of {} dataset: source_signal, params_data, response_data, ts...".format(key), source_signal.shape, params_data.shape, response_data.shape, ts.shape)
    
    tuple_data = (source_signal, params_data, response_data, ts)
    data = torch.utils.data.TensorDataset(source_signal, params_data, response_data, ts)
    
    if key == 'train':
        data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle=True, generator = torch.Generator(device))
    else:
        data_loader = torch.utils.data.DataLoader(data, batch_size = 1, shuffle=False, generator = torch.Generator(device))
    
    if key == 'train':
        return tuple_data, data_loader, output_normalize_config
    else:
        return tuple_data, data_loader
