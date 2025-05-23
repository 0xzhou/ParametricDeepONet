

import matplotlib.pyplot as plt
import torch
import numpy as np
import os, pylab
from utils import GLOBAL
plt.rcParams["font.family"] = "Times New Roman"


def plot_case1_data_samples(pred_output, output, para, ts, save_as, legend = False, alpha = 0.9, resolution = 200):
        #print("mu1 is {}, mu2 is {}".format(para[0], para[1]))
        fig = plt.figure(figsize=(5, 3))
        ft = 18
        ticks_fz = 14
        x = np.linspace(0, 2, 200)
        
        if resolution != 200:
            varying_x = np.linspace(0, 2, resolution)
            plt.plot(x ,output[:,0].cpu().numpy(), '-', color = GLOBAL.red, lw = 5, label = "Ground truth", alpha = 0.9, zorder = -1)
            plt.scatter(varying_x, pred_output[:,0].cpu().numpy(), color = 'k', s = 5, label = "Evaluation points", alpha = 0.9, marker='o', zorder = 1)
        else:
            plt.plot(x ,output[:,0].cpu().numpy(), '-', color = GLOBAL.red, lw = 3, label = "Ground truth", alpha = alpha)
            plt.plot(x, pred_output[:,0].cpu().numpy(), "--", color = GLOBAL.blue, lw = 2, label = "Prediction", alpha = alpha)

        plt.xticks(np.arange(0,2.1,0.5),fontsize=ticks_fz)
        plt.yticks(fontsize=ft)
        plt.ylim(-15, 20)
        plt.ylabel("$\ddot{x}$", rotation = 0, fontsize = ft)
        plt.xlabel("$t$", fontsize = ft)
        
        plt.title(r"$\mu_1 = {:.2f}, \mu_2 = {:.2f}$".format(para[0], para[1]), fontsize = ft)

        ax=plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        if save_as:
            plt.savefig(save_as, dpi = 600)
        
        leg_fig = plt.figure(figsize=(6, 0.5))
        pylab.figlegend(*fig.gca().get_legend_handles_labels(), loc = 'center', ncol = 2, fontsize = 15, frameon = False)
        leg_fig.savefig(os.path.join('./legend_case1.png'), dpi = 600)
        
        plt.show() 
        plt.close()
        
def plot_case1_params_scatter_v0(params_group_1, params_group_pred_1, std_params_group_pred_1,
                        params_group_2, params_group_pred_2, std_params_group_pred_2,
                        label, title, save_as = None, errorbar = True, metrics = None, color1 = None, color2 = None, alpha = 0.5):
    
        with torch.no_grad():
            plt.figure(figsize=(5, 5)) 
            size = 60
            ft = 18

            if errorbar:
                plt.errorbar(np.squeeze(params_group_1.cpu().numpy()) , np.squeeze(params_group_pred_1.cpu().numpy()), color = color1, yerr = std_params_group_pred_1.cpu().numpy(), fmt='None', alpha = alpha)
                plt.errorbar(np.squeeze(params_group_2.cpu().numpy()) , np.squeeze(params_group_pred_2.cpu().numpy()), color = color2, yerr = std_params_group_pred_2.cpu().numpy(), fmt='None', alpha = alpha)
                
            plt.scatter(params_group_1.cpu().numpy(), params_group_pred_1.cpu().numpy(), label = "in-distribution test", color = color1, s = size, marker = 'o', alpha = alpha)
            plt.scatter(params_group_2.cpu().numpy(), params_group_pred_2.cpu().numpy(), color = color2, label = "out-of-distribution test", s = size, alpha = alpha)
            plt.xticks(fontsize = ft)
            plt.yticks(fontsize = ft)
            
            if label == 'Stiffness':
                if metrics:
                    nrmse, R2 = metrics[0].cpu().numpy()[0], metrics[1].cpu().numpy()[0]
                plt.xlim(0, 110)
                plt.ylim(0, 110)
                plt.xticks(np.arange(0, 101, 20))
                plt.yticks(np.arange(0, 101, 20))
            elif label == 'Damping':
                if metrics:
                    nrmse, R2 = metrics[0].cpu().numpy()[1], metrics[1].cpu().numpy()[1]
                    print("NRMSE: {:.3e} \n $R^2$ = {:.3e}".format(nrmse, R2))
                plt.xlim(0, 11)
                plt.ylim(0, 11)
                plt.xticks(np.arange(0, 11, 2))
                plt.yticks(np.arange(0, 11, 2))


            plt.xlabel(r"True value", fontsize = ft)
            plt.ylabel(r"Estimation", fontsize = ft)
            #plt.legend(loc = 'upper left', fontsize = 14) 

            ax=plt.gca()
            plt.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="k", lw =3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        plt.title(title, fontsize = ft)
    
        plt.tight_layout()
        plt.savefig(save_as, dpi = 600)
        plt.show()
        plt.close()
        
def plot_case1_params_scatter(params_group_1, params_group_pred_1, std_params_group_pred_1,
                        params_group_2, params_group_pred_2, std_params_group_pred_2,
                        label, title, save_as = None, errorbar = True, metrics = None, color1 = '#007afe', color2 = None, alpha = 0.8):
    
        with torch.no_grad():
            plt.figure(figsize=(5, 5)) 
            size = 100
            ft = 30
            ticksize = 20

            if errorbar:
                plt.errorbar(np.squeeze(params_group_1.cpu().numpy()) , np.squeeze(params_group_pred_1.cpu().numpy()), color = color1, yerr = std_params_group_pred_1.cpu().numpy(), fmt='None', alpha = alpha)
                plt.errorbar(np.squeeze(params_group_2.cpu().numpy()) , np.squeeze(params_group_pred_2.cpu().numpy()), color = color2, yerr = std_params_group_pred_2.cpu().numpy(), fmt='None', alpha = alpha)
                
            plt.scatter(params_group_1.cpu().numpy(), params_group_pred_1.cpu().numpy(), label = "in-distribution test", color = color1, s = size, marker = 'x', alpha = alpha)
            plt.scatter(params_group_2.cpu().numpy(), params_group_pred_2.cpu().numpy(), color = color2, label = "out-of-distribution test", s = size, alpha = alpha)
            plt.xticks(fontsize = ft)
            plt.yticks(fontsize = ft)
            
            if label == 'Stiffness':
                if metrics:
                    nrmse, R2 = metrics[0].cpu().numpy()[0], metrics[1].cpu().numpy()[0]
                    print("NRMSE: {:.3e}".format(nrmse))
                plt.xlim(0, 110)
                plt.ylim(0, 110)
                plt.xticks(np.arange(0, 101, 20), fontsize = ticksize)
                plt.yticks(np.arange(0, 101, 20), fontsize = ticksize)
            elif label == 'Damping':
                if metrics:
                    nrmse, R2 = metrics[0].cpu().numpy()[1], metrics[1].cpu().numpy()[1]
                    print("NRMSE: {:.3e}".format(nrmse))
                plt.xlim(0, 11)
                plt.ylim(0, 11)
                plt.xticks(np.arange(0, 11, 2), fontsize = ticksize)
                plt.yticks(np.arange(0, 11, 2), fontsize = ticksize)

            plt.xlabel(r"Ground truth", fontsize = ft)
            plt.ylabel(r"Estimation", fontsize = ft)

            ax=plt.gca()
            plt.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="k", lw =3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        plt.title(title, fontsize = 30)
    
        plt.tight_layout()
        plt.savefig(save_as, dpi = 600)
        plt.show()
        plt.close()
        
def plot_params_range(train_data, test_data, sub_case, save_as):
    plt.figure(figsize=(5, 5))
    alpha = 1
    font_size = 15
    ticks_fz = 15
    
    color1 = GLOBAL.red
    color2 = GLOBAL.blue
    
    plt.scatter(train_data[1][:,0].cpu().detach().numpy(), train_data[1][:,1].cpu().detach().numpy(), color = color1, s = 10, label = 'training data', marker= 'o' , alpha = alpha)
    plt.scatter(test_data[1][:200,0].cpu().detach().numpy(), test_data[1][:200,1].cpu().detach().numpy(),color = color2, s = 20, label = 'test data', marker= 'x' , alpha = alpha)
    plt.xlabel(r'$\mu_1$', fontsize = font_size)
    plt.ylabel(r'$\mu_2$', fontsize = font_size, rotation = 0)
    plt.xlim([10, 100])
    plt.ylim([1, 10])
    plt.xticks(fontsize = ticks_fz)
    plt.yticks(fontsize = ticks_fz)
    if sub_case == 'case1a':
        plt.title("Case 1a", fontsize = 20)
    elif sub_case == 'case1b':
        plt.title("Case 1b", fontsize = 20)
        x = [[40, 40], [10, 40], [10,40], [40,40], [70,70],[70,100], [70,100], [70,70]]
        y = [[1, 4],   [4, 4],    [7,7],    [7,10], [1,4], [4,4],    [7,7],     [7,10]]
        for i in range(len(x)):
            plt.plot(x[i], y[i], ls = '--', color = color1, alpha = alpha)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi = 600)
    plt.show()
    plt.close()
    
    
def calculate_arg_metrics(model_experiments_dir):
    train_list = os.listdir(model_experiments_dir)
    npy_file_list = []
    for train_run in train_list:
        if 'training_record.npy' in os.listdir(os.path.join(model_experiments_dir, train_run)):
            npy_file_list.append(os.path.join(model_experiments_dir, train_run, 'training_record.npy'))
        
    min_train_loss_list = []
    min_test_loss_list = []
    print(npy_file_list)
    for npy_file in npy_file_list:
        data = np.load(npy_file, allow_pickle=True)
        min_train_loss_list.append(min(data.item()['Loss_train']))
        min_test_loss_list.append(min(data.item()['Loss_test']))
        
    min_train_loss = np.mean(min_train_loss_list)
    min_test_loss = np.mean(min_test_loss_list)
    std_train_loss = np.std(min_train_loss_list)
    std_test_loss = np.std(min_test_loss_list)
    print('min_train_loss: ', min_train_loss)
    print('min_test_loss: ', min_test_loss)
    print('std_train_loss: ', std_train_loss)
    print('std_test_loss: ', std_test_loss)
    
    return min_train_loss, min_test_loss, std_train_loss, std_test_loss


### Visualization script for Case 2

def plot_output_samples(case, gt, pred, save_as=None, num_sensors=4):
    ft = 25
    tick_size = 20
    lw = 2
    fig = plt.figure(figsize=(5 * num_sensors, 3.5))
    for i in range(num_sensors):
        x_1 = np.linspace(0, 1/1666 * 200, 200)
        x_2 = np.linspace(0, 200, 400)
        plt.subplot(1, num_sensors, i + 1)
        plt.plot(x_1,gt[i].cpu().detach().numpy(), color="#D4352D", lw=3, label="Ground truth", alpha=0.9)
        plt.plot(x_1,pred[i].cpu().detach().numpy(), color="#007afe", linestyle="--", lw=lw, label="Prediction")
        plt.xlabel("$t$", fontsize=tick_size)
        plt.xticks(fontsize=tick_size)
        plt.xticks(np.arange(0,0.13,0.03))
        plt.ylabel(r"$a_{}$".format(i + 1), fontsize=tick_size, rotation=0)
        plt.yticks(fontsize=tick_size)
        plt.yticks(np.arange(-0.4, 0.41, 0.2))
        plt.title("State {}".format(case), fontsize=ft)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_as, dpi=600)
    plt.show()
    plt.close()

def generate_param_vector(mu, sigma, length):
    x = torch.linspace(0, 1, length)
    y = torch.exp(-(x-mu)**2/(2*sigma**2))
    return x, y

def params_to_vector(params, length):
    batchsize = params.shape[0]
    y_sum = torch.zeros(batchsize, length)
    
    data_mu = [0.17, 0.3, 0.5]
    data_mu = torch.tensor(data_mu).to("cuda:0")
    data_mu = data_mu.repeat(batchsize, 1)

    for i in range(3):
        mu =  data_mu[:, i].unsqueeze(1) # Expand dims to broadcast
        high =  params[:, i].unsqueeze(1) # Expand dims to broadcast
        
        x, y = generate_param_vector(mu, 0.01, length)
        y = y * high
        y_sum += y
    return x, y_sum


def plot_para_gaussian(case,  mu_test, pred_mu_test, save_as = None, legend = False):
    with torch.no_grad():
        fig = plt.figure(figsize=(6, 3)) 
        pred_alpha = 0.1
        alpha = 0.9
        ft = 18
        lw = 2
        
        color_pred = '#383838'
        color_true_1 = '#D4352D'
        color_true_2 = '#007afe'
        
        mu = mu_test[0].unsqueeze(0)
        mu_x, mu_y = params_to_vector(mu, 1000)
            
        if case in ['1', '2', '3', '4', '7', '10']:
            plt.plot(mu_x.cpu().numpy(), mu_y[0].cpu().numpy(), '-' , label = 'Training data (ground truth)', color = color_true_1, lw = lw, alpha = alpha)
            plt.plot([], [], label = 'Test data (ground truth)', color = color_true_2, lw = lw)
        else:
            plt.plot(mu_x.cpu().numpy(), mu_y[0].cpu().numpy(), '-', label = 'Ground truth, extrapolation', color = color_true_2, lw = lw, alpha = alpha)
            
        pred_mu_x, pred_mu_y = params_to_vector(pred_mu_test, 1000)
        for i in range(pred_mu_y.shape[0]):
            if i == 0:
                plt.plot(pred_mu_x.cpu().numpy(), pred_mu_y[i].cpu().numpy(), '-', label = 'Estimation', color = color_pred,  lw = 0.5, alpha = pred_alpha)
            else:
                plt.plot(pred_mu_x.cpu().numpy(), pred_mu_y[i].cpu().numpy(), '-', color = color_pred,  lw = 1, alpha = pred_alpha)
            

        plt.title('State {}'.format(case), fontsize = 25)
        plt.xlabel('$x_{Loc}$', fontsize=ft)
        plt.ylabel('$y_{L}$', fontsize=ft, rotation=0, labelpad=20)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xticks(fontsize = ft)
        plt.yticks(fontsize = ft)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_as, dpi = 300)
        plt.show()
        plt.close()