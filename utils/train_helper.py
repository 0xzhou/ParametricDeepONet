import sys, os, shutil, math
import numpy as np

class TerminalLogger(object):
    def __init__(self, stdout, save=None):
        #self.terminal = stream
        #self.log = open(save, 'a+')
        self.log = open(save, 'w')
        self.stdout = stdout

    def write(self, message):
        self.log.write(message)
        self.stdout.write(message)

    def flush(self):
        self.log.flush()
        #self.log.close()
        self.stdout.flush()
    
def save_source_files(src_dir, dst_dir):
    """
    Function to copy all .py and .json files from src_dir to dst_dir
    while maintaining the original directory structure.

    Parameters:
    src_dir (str): Path to the source directory
    dst_dir (str): Path to the destination directory
    """
    # Check if dst_dir is a subdirectory of src_dir
    if os.path.commonpath([src_dir]) == os.path.commonpath([src_dir, dst_dir]):
        raise ValueError("Save directory cannot be inside the project directory. Please specify a different directory.")

    # Walk through the source directory
    for dirpath, dirnames, filenames in os.walk(src_dir):
        if dirpath.endswith('__pycache__') or dirnames == 'record':
            continue
        # Create a corresponding directory in the destination directory
        structure = os.path.join(dst_dir, os.path.relpath(dirpath, src_dir))
        os.makedirs(structure, exist_ok=True)
        
        # Copy each python or json file to the destination directory
        for filename in filenames:
            if filename.endswith('.py') or filename.endswith('.sh') or filename.endswith('.yaml'):
                src_file = os.path.join(dirpath, filename)
                dst_file = os.path.join(structure, filename)
                shutil.copy2(src_file, dst_file)
    print("Saving source files successfully.")
    
    
def adjust_learning_rate(optimizer, init_lr, epoch, mode = 'cosine', total_epoch = 100):
    if mode == 'cosine':
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / total_epoch))
    elif mode == 'half':
        cur_lr = init_lr * 0.5 ** (epoch // 200)
        
    cur_lr = max(0.0001, cur_lr)
        
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            print("Current epoch learning rate is: ", cur_lr)
            param_group['lr'] = cur_lr
            
            
def run_experiment(trainer_class, args):
    mse_list = []
    trainer = trainer_class(args)
    
    for _ in range(args.run):
        best_test_mes = trainer.train()
        mse_list.append(best_test_mes)
    
    # print("The MSE is: {}".format(mse_list))
    # print("The average MSE is: {}".format(np.mean(mse_list)), "std: {}".format(np.std(mse_list)))



