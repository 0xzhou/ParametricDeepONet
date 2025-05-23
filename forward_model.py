import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchinfo import summary
from tqdm import tqdm
from datetime import datetime
import os, sys, math, yaml, random
import models
from utils import args_parser, train_helper, data_processing
from utils.script import prepare_blade_dataset
from sklearn.decomposition import PCA


class Surrogate(object):
    """ neural network-based surrogate model for approximating forward dynamics/physics
    Args:
        args: from config file
    """
    def __init__(self, args) -> None:
        super(Surrogate).__init__()

        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        assert self.args.save_in, "Please specify the save directory in the config file."
        
        print("The model is set to be: ", self.args.model_type)
        self.model_register()

        self.load_data()
        self.training_record = {}

        self.set_all_seeds(seed=self.args.seed)

    def train(self):
        self.save_config()
        self.model.to(self.device)

        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr= self.args.lr)
        elif self.args.optimizer == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr= self.args.lr, momentum=0.9)

        Loss_train = []
        Loss_test = []
        Loss_id_test  = []
        Loss_ood_test = []

        best_train_loss, best_test_loss = math.inf, math.inf
        best_train_loss_epoch, best_test_loss_epoch = 0, 0
        
        for key, value in vars(self.args).items():
            print(key, ":", value)

        summary(self.model)
    
        for epoch in range(self.args.epoch):
            epoch_loss = []

            # Training
            self.model.train()       
            for (input_data, para_data, output_data, ts) in tqdm(self.train_loader, disable=True):
                #print(input_data.shape, para_data.shape, output_data.shape, ts.shape)
                # input()
                pred_output = self.model(input_data, mu = para_data, y_loc = ts) # Predicted output
                loss = self.calculate_metrics(pred_output, output_data, mode = self.args.loss_mode)
                epoch_loss.append(loss.item()) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if np.mean(epoch_loss) < best_train_loss:
                best_train_loss = np.mean(epoch_loss)
                best_train_loss_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_train_loss_model.pt'))

            # Test
            self.model.eval()
            with torch.no_grad():
                if self.args.data_case == 'blade':
                    id_loss = self.evaluate_y_loss(model_path = None, data = self.id_test_data) # in-distribution test data
                    ood_loss = self.evaluate_y_loss(model_path = None, data = self.ood_test_data) # out-of-distribution test data

                elif self.args.data_case == 'duffing':
                    id_loss = torch.tensor(.0)
                    ood_loss = torch.tensor(.0)

                test_loss = self.evaluate_y_loss(model_path = None, data = self.test_data)

            if test_loss < best_test_loss:
                best_test_loss = test_loss.item()
                best_test_loss_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_test_loss_model.pt'))

            Loss_train.append(np.mean(epoch_loss))
            Loss_id_test.append(id_loss.item())
            Loss_test.append(test_loss.item())
            Loss_ood_test.append(ood_loss.item())

            if epoch % self.args.print_every == 0:
                print('epoch: {}, loss: {:.4e}'.format(epoch, np.mean(epoch_loss)))
                print('epoch: {}, test_loss: {:.4e}'.format(epoch, test_loss))
                if self.args.data_case == 'blade':
                    print('epoch: {}, id_error: {:.4e}, ood_error: {:.4e}'.format(epoch, torch.mean(id_loss).item(), torch.mean(ood_loss).item()))

        self.plot_training_loss(Loss_train[1:], Loss_test[1:], Loss_id_test[1:], Loss_ood_test[1:] , save_as = os.path.join(self.save_dir, 'loss.png'))

        self.training_record['Loss_train'] = Loss_train
        self.training_record['Loss_test'] = Loss_test
        self.training_record['Loss_id_test'] = Loss_id_test
        self.training_record['Loss_ood_test'] = Loss_ood_test

        print("-"*50)
        print("The best train loss epoch is: {}".format(best_train_loss_epoch))
        print("The best train loss is: {:.4e}".format(best_train_loss))
        print("The best corresponding test is: {:.4e}".format(Loss_test[best_train_loss_epoch]))
        print("The best corresponding id test is: {:.4e}".format(Loss_id_test[best_train_loss_epoch]))
        print("The best corresponding ood test is: {:.4e}".format(Loss_ood_test[best_train_loss_epoch]))
        
        print("-"*50)
        print("The best test loss epoch is: {}".format(best_test_loss_epoch))
        print("The best test loss is: {}".format(best_test_loss))
        print("The best iid test loss is: {:.4e}".format(np.min(Loss_id_test)))
        print("The best ood test loss is: {:.4e}".format(np.min(Loss_ood_test)))

        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, 'best_train_loss_model.pt'),weights_only=True))
        self.model.eval()
        np.save(os.path.join(self.save_dir, 'training_record.npy'), self.training_record)

        return 0
    
    def evaluate_y_loss(self, model_path = None, data = None):
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        with torch.no_grad():
            input_test, para_test, output_test, ts_test = data
            pred_output_test = self.model(input_test, para_test, y_loc = ts_test)

        loss = self.calculate_metrics(pred_output_test, output_test, mode = self.args.loss_mode)
        return loss
    
    def evaluate_y(self):
        test_x, test_params, test_y, test_ts = self.test_data
        with torch.no_grad():
            test_data_pred = self.model(test_x, test_params, test_ts)
        return test_data_pred
    
    def evaluate_y_resolution(self, resolution = 100, data_case = 'duffing'):
        test_x, test_params, test_y, test_ts = self.test_data

        if data_case == 'blade':
            varying_test_ts = torch.linspace(0, 4 * 200 * 1.0/ 1666, resolution * 4).to(self.device)
            if self.args.decode_mode == 'nonlinear':
                varying_test_ts = varying_test_ts.reshape(-1, resolution)
                varying_test_ts = varying_test_ts.unsqueeze(0).repeat(test_x.shape[0], 1, 1)
        else:
            varying_test_ts = torch.linspace(0, 2, resolution).to(self.device)
            varying_test_ts = varying_test_ts.unsqueeze(0).repeat(test_x.shape[0], 1)
        print("The shape of varying_test_ts is: ", varying_test_ts.shape)
        
        with torch.no_grad():
            if data_case == 'duffing':
                if self.args.model_type == 'ParamsMLP':
                    varying_test_ts = varying_test_ts.unsqueeze(2)
                    varying_test_x = 10 * torch.sin(2 * torch.pi * 1 * varying_test_ts + 2 * np.pi * (10 - 1) / 2 * varying_test_ts **2 / 2)
                    outputs = []
                    reshaped_test_x = varying_test_x.reshape(test_x.shape[0], -1, 200)
                    for x in torch.split(reshaped_test_x, 1, dim = 1):
                        x = x.squeeze(1)
                        output = self.model(x, test_params, test_ts)
                        outputs.append(output)
                    test_data_pred = torch.cat(outputs, dim = 1) 
                else:
                    test_data_pred = self.model(test_x, test_params, varying_test_ts)   
                    
            elif data_case == 'blade':
                if self.args.decode_mode == 'nonlinear':
                    outputs = []
                    #reshaped_test_ts = varying_test_ts.reshape(test_ts.shape[0], -1, 800)
                    for ts in torch.split(varying_test_ts, 1, dim = 1):
                        ts = ts.squeeze(1)
                        print("The shape of ts is: ", ts.shape)
                        output = self.model(test_x, test_params, ts, resolution)
                        outputs.append(output)
                        test_data_pred = torch.cat(outputs, dim = 1)
                        test_data_pred = test_data_pred.reshape(test_x.shape[0], -1, resolution)
                    print("The shape of test_data_pred is: ", test_data_pred.shape)
                
                elif self.args.decode_mode == 'linear' :
                    print("The shape of varying_test_ts is: ", varying_test_ts.shape)
                    test_data_pred = self.model(test_x, test_params, varying_test_ts, resolution)
                    print("The shape of test_data_pred is: ", test_data_pred.shape)
                    #test_data_pred = torch.cat(outputs, dim = 1)
        return test_data_pred
        
    
    def save_config(self):
        # Create a new folder to save results
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        postfix = "{}_D{}W{}K{}_Epoch{}_{}_Batch{}_lr{}_Seed{}".format(
            self.model.model_name,
            self.args.depth,
            self.args.width,
            self.args.pebasis,
            self.args.epoch,
            self.args.optimizer,
            self.args.batch_size,
            self.args.lr,
            self.seed,
        )
        save_name = time_stamp + "_" + postfix
        self.save_dir = os.path.join(self.args.save_in, save_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.terminal_log = os.path.join(self.save_dir, "terminal_log.txt")
        sys.stdout = train_helper.TerminalLogger(sys.stdout, self.terminal_log)

    def create_eval_dir(self, save_root_dir = ''):
        if save_root_dir:
            save_root_dir = save_root_dir
        else:
            save_root_dir = self.args.project_dir

        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.args.eval_type == 'forward_eval':
            self.save_dir = os.path.join(save_root_dir, time_stamp+'{}_Seed_'.format(self.args.eval_type
                                                                                    )+str(self.seed)
                                        )
            
        elif self.args.eval_type == 'inverse_eval':
            self.save_dir = os.path.join(save_root_dir, time_stamp+'{}_BS-{}_Run-{}_Epoch-{}_Reg-{}_Seed_'.format(self.args.eval_type,
                                                                                        self.args.batch_size,
                                                                                        self.args.run,
                                                                                        self.args.epoch,
                                                                                        self.args.regl_case,
                                                                                        )+str(self.seed)
                                        )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        

    def calculate_metrics(self, pred_output, output, mode="MSE"):
        if mode == "MSE":
            return torch.mean((pred_output - output) ** 2)
        elif mode == "RMSE":
            return torch.sqrt(torch.mean((pred_output - output) ** 2))
        elif mode == "NRMSE":
            return torch.sqrt(torch.mean((pred_output - output) ** 2) / torch.mean(output**2))
        elif mode == "NRMSE_y_max_min":
            return torch.sqrt(torch.mean((pred_output - output) ** 2)) / (torch.max(output) - torch.min(output))
        
        
    def model_register(self):
        if self.args.model_type == "ParametricDeepONet":
            self.model = models.ParametricDeepONet(self.args)
            print("The decoder mode is: ", self.args.decode_mode)
            print("The depth of the model is: ", self.args.depth)
            print("The width of the model is: ", self.args.width)
        elif self.args.model_type == "ParamsMLP":
            self.model = models.ParamsMLP(self.args)
        elif self.args.model_type == "DeepONet":
            self.model = models.DeepONet(self.args)
        elif self.args.model_type == "ParamsCNN":
            self.model = models.ParamsCNN(x_t_shape=0, params_dim=0, case=self.args.data_case)
            

    def plot_training_loss(self, train_loss_list, test_loss_list, id_test_loss_list, ood_test_loss_list, save_as = None):
        plt.figure(figsize=(10, 10))
        plt.plot(train_loss_list,linewidth=1, label='train', color = 'b')
        plt.plot(test_loss_list,linewidth=1, label='test', color = 'r')
        if self.args.data_case == 'blade':
            plt.plot(ood_test_loss_list,linewidth=1, label='ood_test', color = 'g')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Loss')
        plt.legend()
        plt.savefig(save_as, dpi = 600)
        plt.close()

    
    def load_data(self):
        if self.args.data_case == "duffing":
            train_data = np.load(self.args.train_dataset, allow_pickle=True).item()
            self.train_data, self.train_loader = data_processing.prepare_duffing_data(
                train_data,
                mode="train",
                batch_size=self.args.batch_size,
                device=self.device,
                params_label=self.args.params_label,
                normalization=self.args.params_normalize,
            )
            
            test_data = np.load(self.args.test_dataset, allow_pickle=True).item()
            self.test_data, self.test_loader = data_processing.prepare_duffing_data(
                test_data,
                mode="test",
                device=self.device,
                params_label=self.args.params_label,
                normalization=self.args.params_normalize,
            )

        elif self.args.data_case == "blade":
            self.train_data, self.train_loader, normalize_config = prepare_blade_dataset(
                self.args.train_dataset, key="train", device=self.device, noised=self.args.noised_data
            )
            if self.args.test_dataset:
                test_data, test_loader = prepare_blade_dataset(
                    self.args.test_dataset,
                    key="test",
                    device=self.device,
                    normalize_config=normalize_config,
                    noised=self.args.noised_data,
                )
                self.test_data = test_data
            elif self.args.id_test_dataset:
                self.id_test_data, self.id_test_loader = prepare_blade_dataset(
                    self.args.id_test_dataset,
                    key="test",
                    device=self.device,
                    normalize_config=normalize_config,
                    noised=self.args.noised_data,
                )
                self.ood_test_data, self.ood_test_loader = prepare_blade_dataset(
                    self.args.ood_test_dataset,
                    key="test",
                    device=self.device,
                    normalize_config=normalize_config,
                    noised=self.args.noised_data,
                )
                self.test_data = data_processing.merge_id_ood_test_data(self.id_test_data, self.ood_test_data)
                
                
    def load_weights(self, weight_path = ''):
        # Load the weights of the model
        if weight_path:
            model_path = weight_path
        else:
            if self.args.eval_type == 'forward_eval':
                model_path = os.path.join(self.args.project_dir, 'best_train_loss_model.pt')
            else:
                model_path = os.path.join(self.args.project_dir, 'best_train_loss_model.pt')
            
        self.model.load_state_dict(torch.load(model_path,weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def set_all_seeds(self, seed):
        if type(seed) == int:
            self.seed = seed
            print("Seed is set to be: ", self.seed)
        else:
            self.seed = random.randint(1, 10000)
            print("Seed is randomly set to be: ", self.seed)
            
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        
    def generate_meshgrid(self, x_range, y_range, latent_net, model_type = 'ParamsCNN'):
        X, Y = np.meshgrid(x_range, y_range)

        params = np.vstack([X.ravel(), Y.ravel()]).T
        print(params.shape)
        params = torch.tensor(params).float().to(self.device)
        params_num = params.shape[0]

        with torch.no_grad():
            if model_type == 'ParametricDeepONet':
                one_x_t = self.train_data[0][0].unsqueeze(0)
                one_y_coord = self.train_data[3][0].unsqueeze(1)
                
                mesh_params_latent = latent_net(one_x_t, params, one_y_coord)
                train_params_latent = latent_net(one_x_t, self.train_data[1], one_y_coord)
                test_params_latent = latent_net(one_x_t, self.test_data[1], one_y_coord)
                print("The shape of mesh_params_latent is: ", mesh_params_latent.shape)
                 
        latent = torch.cat([mesh_params_latent, train_params_latent], dim = 0)
        print("The shape of latent is: ", latent.shape)
        pca = PCA(n_components=1)
        pca.fit_transform(latent.cpu().detach().numpy())
        Z = pca.transform(latent.cpu().detach().numpy())

        test_params_pca = pca.transform(test_params_latent.cpu().detach().numpy())
        train_params_pca = Z[params_num: params_num + len(self.train_data[1])]

        Z = Z[:params_num]
        Z = Z.reshape(X.shape)
        print(Z.shape)
        return X, Y, Z, train_params_pca, test_params_pca


if __name__ == '__main__':
    args = args_parser.parse_arguments(sys.argv[1:])
    
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)
    
    for run in range(args.run):
        trainer = Surrogate(args)
        trainer.train()
        del trainer
    

