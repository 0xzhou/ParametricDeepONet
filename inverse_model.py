import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, math, yaml, random
sys.path.append('./utils')
from forward_model import Surrogate
import models.inverse_net as inverse_net
from forward_model import Surrogate

class InverseModel(Surrogate):
    def __init__(self, args) -> None:
        super(InverseModel, self).__init__(args)
        
        self.evaluate_record = {}
        self.evaluate_record['gi_process_params'] = []
        self.evaluate_record['nr_process_params'] = []
        self.evaluate_record['test_params'] = self.test_data[1].detach().cpu().numpy()

    def gradient_init(self, data, batch_size, epoch, forward_net, lr, regl_case = '', data_case = 'case1', check_int = 500, data_source = 'train'):
        """
        Inverse modeling: Gradient initialization step
        Args:
            data: the tuple of data, (x, params, y, t)
            forward_net: the pre-trained forward model
            regl_case: the regularization choice when updating the parameters.

        Returns:
            pred_params: the initializated parameters
        """
        dataset = torch.utils.data.TensorDataset(data[0], data[1], data[2], data[3])
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False) 

        pred_params = []
        
        for idx, s in enumerate(data_loader):
            x, _, y, t = s # params is not used in gradient-based initialization
            
            gi_net = inverse_net.GradientInitialNet(batch_size = x.shape[0], forward_net = forward_net, case = data_case)
            optimizer = optim.Adam(gi_net.parameters(), lr=lr)
            gi_net.to(self.device)
            gi_net.train()
            
            best_test_loss = math.inf
            best_test_loss_epoch = 0
            epoch_params_init = []
            
            for i in range(epoch):
                pred = gi_net(x, t)
                loss = self.calculate_metrics(pred, y, mode = self.args.loss_mode)
                
                if data_case == 'case2' and data_source == 'train':
                    mu_init = gi_net.params.data.cpu().detach().numpy()
                    epoch_params_init.append(mu_init)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    gi_net.params.data[:] = self.proper_update_params(gi_net.params.data, regl_case)

                if loss < best_test_loss:
                    best_test_loss = loss
                    best_test_loss_epoch = i
                    torch.save(gi_net.state_dict(), './best_model.pth')

                if i % check_int == 0:
                    print(f'Epoch: %d, Loss: %.3f ' % (i, loss.item()))
            
            print(f'Best test loss: %.3f at epoch %d' % (best_test_loss, best_test_loss_epoch)) 
            gi_net.load_state_dict(torch.load('best_model.pth', weights_only=True))
            self.evaluate_record['gi_process_params'].append(epoch_params_init)
            pred_params.append(gi_net.params)

        pred_params = torch.cat(pred_params, dim = 0)
        if data_source == 'train':
            np.save(os.path.join(self.save_dir, 'pred_params_train.npy'), epoch_params_init)
        else:
            np.save(os.path.join(self.save_dir, 'pred_params_test.npy'), epoch_params_init)
        
        return pred_params
    
    def iterative_refine(self, ir_net, optimizer, forward_model, data_loader, check_int = 100, epochs = 1000, iter_steps = 10, memory_iter = 0, 
                         regl_case = '', data_case = 'case2', data_source = 'train', use_evaluate_mode = True):

        best_loss = math.inf
        best_loss_epoch = 0
        self.evaluate_record['forward_gradients_list'] = []
        self.evaluate_record['train_refine_loss'] = []
        self.evaluate_record['test_refine_loss'] = []

        for i in range(epochs):
            epoch_forward_loss, epoch_inverse_loss, epoch_loss = [], [], []
            for (mu_0, x_t, mu, y_t, ts) in data_loader:
                mu_hat, forward_loss = self.iteratively_update_mu(forward_model = forward_model, ir_net = ir_net, 
                                                            x_t = x_t, mu_hat_0 = mu_0, y_t = y_t, ts = ts, iter_steps = iter_steps, memory_iter = memory_iter,
                                                            regl_case=regl_case)

                inverse_loss = self.calculate_metrics(mu_hat, mu, mode = 'RMSE')
                loss = forward_loss + inverse_loss
                #loss = inverse_loss

                epoch_loss.append(loss.item())
                epoch_forward_loss.append(forward_loss.item())
                epoch_inverse_loss.append(inverse_loss.item())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            
            self.evaluate_record['train_refine_loss'].append(np.mean(epoch_inverse_loss))

            if loss < best_loss:
                best_loss = loss
                best_loss_epoch = i
                torch.save(ir_net.state_dict(), os.path.join(self.save_dir,'best_inverse_model.pt'))

            if i % check_int == 0:
                print("Epoch:{}, Loss:{:.5f}, Forward Loss:{:.5f}, Inverse Loss:{:.5f}".format(i, np.mean(epoch_loss), np.mean(epoch_forward_loss), np.mean(epoch_inverse_loss)))
                
            if data_case == 'case2' and data_source == 'train':
                        _mu_0, _x_t, _mu, _y_t, _ts = self.post_pi_train_data[0], self.post_pi_train_data[1], self.post_pi_train_data[2], self.post_pi_train_data[3], self.post_pi_train_data[4]
                        _mu_hat, _ = self.iteratively_update_mu(forward_model = forward_model, ir_net = ir_net,
                                                                x_t = _x_t, mu_hat_0 = _mu_0, y_t = _y_t, ts = _ts, iter_steps = iter_steps, memory_iter = memory_iter,
                                                                regl_case=regl_case)
                        _mu_refine = _mu_hat.cpu().detach().numpy()
                        self.evaluate_record['nr_process_params'].append(_mu_refine)
            
            if use_evaluate_mode:
                mu_test_hat, test_forward_loss = self.iteratively_update_mu(forward_model = forward_model, ir_net = ir_net,
                                                            x_t = self.post_pi_test_data[1], mu_hat_0 = self.post_pi_test_data[0], y_t = self.post_pi_test_data[3], ts = self.post_pi_test_data[4], 
                                                            iter_steps = iter_steps, memory_iter = memory_iter, regl_case=regl_case)
                test_inverse_loss = self.calculate_metrics(mu_test_hat, self.post_pi_test_data[2], mode = self.args.loss_mode)
                test_loss = test_forward_loss + test_inverse_loss
                self.evaluate_record['test_refine_loss'].append(test_inverse_loss.item())

        print("Best loss: {:.5f} at epoch {}".format(best_loss, best_loss_epoch))

        ir_net.load_state_dict(torch.load(os.path.join(self.save_dir,'best_inverse_model.pt'), weights_only=True))
        self.generate_post_ir_data(ir_net, iter_steps, memory_iter, regl_case)
        
        np.save(os.path.join(self.save_dir, 'evaluate_record.npy'), self.evaluate_record)
        
    def iteratively_update_mu(self, forward_model, ir_net, x_t, mu_hat_0, y_t, ts, iter_steps, memory_iter, regl_case):

        mu_hat = mu_hat_0 # the iteration starts

        if memory_iter == 0:
            memory_dydmu = 0
        else:
            memory_dydmu = nn.Parameter(torch.zeros(mu_hat.shape[0], mu_hat.shape[1] * memory_iter)).float().to(self.device)

        for iter in range(iter_steps):
            mu_hat = mu_hat.requires_grad_(True)
            mu_hat_foward_loss = self.calculate_metrics(forward_model(x_t, mu_hat, y_loc = ts), y_t, mode = self.args.loss_mode)
            dydmu = torch.autograd.grad(mu_hat_foward_loss, mu_hat)[0]
            learned_dydmu, memory_dydmu = ir_net.estimate_learned_gradient(memory_dydmu, dydmu, mu_hat)
            
            mu_hat = mu_hat + learned_dydmu
            with torch.no_grad():
                mu_hat[:] = self.proper_update_params(mu_hat, regl_case)

        mu_hat_foward_loss = self.calculate_metrics(forward_model(x_t, mu_hat, y_loc = ts), y_t, mode = self.args.loss_mode)

        return mu_hat, mu_hat_foward_loss


    def proper_update_params(self, params, regl_case):
        if regl_case == 'no_regularization':
            pass
        elif regl_case == 'case1':
            params.data[:,0] = params.data[:,0].clamp(10, 100)
            params.data[:,1] = params.data[:,1].clamp(1, 10)
        elif regl_case == 'blade_case2_free':
            params.data[:,:] = params.data[:,:].clamp(0, 1)
        return params


    def generate_post_pi_data(self, pi_pred_params, mode = 'train'):
        if mode == 'train':
            post_pi_data = pi_pred_params, self.train_data[0], self.train_data[1], self.train_data[2], self.train_data[3]   
            self.post_pi_train_data = post_pi_data 
            self.post_pi_train_dataset = torch.utils.data.TensorDataset(post_pi_data[0], post_pi_data[1], post_pi_data[2], post_pi_data[3], post_pi_data[4])
        else:
            post_pi_data = pi_pred_params, self.test_data[0], self.test_data[1], self.test_data[2], self.test_data[3]
            self.post_pi_test_data = post_pi_data

    def generate_post_ir_data(self, ir_net, iter_steps, iter_memory, regl_case = ''):

        self.ir_train_params, _ = self.iteratively_update_mu(forward_model = self.model, ir_net = ir_net, 
                                                            x_t = self.post_pi_train_data[1], 
                                                            mu_hat_0 = self.post_pi_train_data[0],
                                                            y_t = self.post_pi_train_data[3], 
                                                            ts = self.post_pi_train_data[4], 
                                                            iter_steps = iter_steps, memory_iter = iter_memory,
                                                            regl_case=regl_case)

        self.ir_test_params, _ = self.iteratively_update_mu(forward_model = self.model, ir_net = ir_net,
                                                               x_t = self.post_pi_test_data[1],
                                                               mu_hat_0 = self.post_pi_test_data[0],
                                                               y_t = self.post_pi_test_data[3],
                                                               ts = self.post_pi_test_data[4],
                                                               iter_steps = iter_steps, memory_iter = iter_memory,
                                                               regl_case=regl_case)

    
    def evaluate_metrics(self, case = 'duffing', mode = 'GI', data = 'train'):
        if data == 'train':
            if mode == 'GI':
                post_GI_data = self.post_pi_train_data
            elif mode == 'NR':
                post_IR_data = self.ir_train_params
            original_data = self.train_data
        elif data == 'test':
            if mode == 'GI':
                post_GI_data = self.post_pi_test_data
            elif mode == 'NR':
                post_IR_data = self.ir_test_params
            original_data = self.test_data
            
        if case == 'duffing':
            if mode == 'GI':
                normalized_rmse = torch.sqrt(torch.mean((post_GI_data[0][:,:] - original_data[1][:,:])**2, dim=0) / torch.mean(original_data[1][:,:]**2, dim=0))
                R2 = 1 - torch.sum((post_GI_data[0][:,:] - original_data[1][:,:])**2, dim=0) / torch.sum((original_data[1][:,:] - torch.mean(original_data[1][:,:], dim=0))**2, dim=0)
            elif mode == 'NR':
                normalized_rmse = torch.sqrt(torch.mean((post_IR_data[:,:] - original_data[1][:,:])**2, dim=0) / torch.mean(original_data[1][:,:]**2, dim=0))
                R2 = 1 - torch.sum((post_IR_data[:,:] - original_data[1][:,:])**2, dim=0) / torch.sum((original_data[1][:,:] - torch.mean(original_data[1][:,:], dim=0))**2, dim=0)

            print("Normalized RMSE of {} mu: ".format(data), normalized_rmse)
            return normalized_rmse, R2
                
        elif case == 'blade':
            if mode == 'GI':
                mu_nrmse = self.calculate_metrics(post_GI_data[0][:,:], original_data[1][:,:], mode = self.args.loss_mode)
            
            elif mode == 'NR':
                mu_nrmse = self.calculate_metrics(post_IR_data[:,:], original_data[1][:,:], mode = self.args.loss_mode)
                
            print("Normalized RMSE of {} mu: ".format(data), mu_nrmse)
            return mu_nrmse, 0
