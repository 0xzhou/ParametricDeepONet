from scipy.signal import butter, filtfilt
import lvm_read, random, pickle
import numpy as np

SENSITIVITY_LIST = [51.4, 51.7, 51.8, 52.0, 52.2, 52.6, 51.4, 52.6, 23.7]  # ratios from the dataset paper, Unit: mV / N

def parse_data_config(data_config={}):
    lvm_list = data_config['lvm_list']
    h_list = data_config['h_list']
    t_list = data_config['t_list']
    w_list = data_config['w_list']

    return lvm_list, h_list, t_list, w_list


def generate_sample_data(lvm, start_index=int, sample_length=int, sensor_idx=int, num_sample=int, filter=True):
    data = lvm[0]['data']  # numpy array

    if sensor_idx: # a_sensor_idx: 1~8
        data = data[start_index:100000, sensor_idx]
        data *= 1000.0 /SENSITIVITY_LIST[sensor_idx-1] # sensor reading to acceleration (volt -> m/s^2)
    else:
        end_index = start_index + 20000
        acc_data = data[start_index:end_index, 1:9]
        force_data = data[start_index:end_index, 13]
        force_data = np.expand_dims(force_data, axis=1)
        data = np.concatenate((acc_data, force_data), axis=1)
        print("The shape of data...", data.shape)
        
        for i in range(9):
            data[:, i] *= 1000.0 / SENSITIVITY_LIST[i]  # sensor reading to acceleration (volt -> m/s^2)

    # Bandpass filter
    if filter:
        order = 6
        cutoff = 380
        fs = 1666
        normal_cutoff = cutoff / (0.5 * fs)
        x, y = butter(order, normal_cutoff, btype='low', analog=False)
        if sensor_idx:
            data = filtfilt(x, y, data)
        else:
            for i in range(9):
                data[:, i] = filtfilt(x, y, data[:, i])
        

    # Generate Sample
    samples = []
    sample_start = 0
    #max_start_idx = 70000 - start_index - sample_length + 1
    for _ in range(num_sample):
        if sensor_idx:
            samples.append(data[sample_start: sample_start + sample_length])
        else:
            samples.append(data[sample_start: sample_start + sample_length, :])
            
        sample_start += 200

    samples = np.stack(samples, axis=0)
    print("The shape of generated sample data...", samples.shape)
    return samples


def create_sample(ts, h_label, t_label, w_label):
    return {
        'ts': ts,
        'h_label': h_label,
        't_label': t_label,
        'w_label': w_label,
    }
    

class ProcessedSampleSet:
    def __init__(self, dataset_config={}, num_sample_per_class = 700, type='', start_idx = 50000, filter=True):
        super(ProcessedSampleSet, self).__init__()
        
        self.start_idx = start_idx
        self.filter = filter
        if dataset_config and num_sample_per_class:
            self.lvm_list, self.h_list, self.t_list, self.w_list = parse_data_config(dataset_config)
            self.data = {}
            self.num_sample_per_class = num_sample_per_class
            self.dataset_config = dataset_config

        if type:
            if type == "full":
                self.create_train_val_test()
            else:
                self.create_single_type(key = type)
                
        
                
    def create_single_type(self, key = None):
        self.data[key] = []

        for i, lvm_file in enumerate(self.lvm_list):
            lvm = lvm_read.read(lvm_file)
            segments = generate_sample_data(lvm, 
                                            start_index=self.start_idx, 
                                            sample_length=200,
                                            sensor_idx= None,
                                            num_sample=self.num_sample_per_class,
                                            filter=self.filter)
            
            for j in range(self.num_sample_per_class):
                self.data[key].append(
                    create_sample(segments[j], self.h_list[i], self.t_list[i], self.w_list[i]))

    def create_train_val_test(self):

        self.data['train'] = []
        self.data['test'] = []
        self.data['val'] = []

        for i, lvm_file in enumerate(self.lvm_list):
            lvm = lvm_read.read(lvm_file)
            segments = generate_sample_data(lvm,
                                            start_index=50000,
                                            sample_length=200,
                                            sensor_idx = None,
                                            num_sample=self.num_sample_per_class,
                                            filter=self.filter)

            num_train = 500
            num_test = 100
            num_val = 100
            
            print("num_train", num_train)
            print("num_test", num_test)
            print("num_val", num_val)

            for j in range(num_train+num_test+num_val):
                if j < num_train:
                    self.data['train'].append(
                        create_sample(segments[j], self.h_list[i], self.t_list[i], self.w_list[i]))
                elif j < num_train + num_test:
                    self.data['test'].append(
                        create_sample(segments[j], self.h_list[i], self.t_list[i], self.w_list[i]))
                else:
                    self.data['val'].append(
                        create_sample(segments[j], self.h_list[i], self.t_list[i], self.w_list[i]))

    def save_as(self, output_dir):
        with open(output_dir, "wb") as f:
            pickle.dump(self, f)
            f.close()

    def load_from(self, input_dir):
        with open(input_dir, "rb") as f:
            data = pickle.load(f)
            f.close()
        return data