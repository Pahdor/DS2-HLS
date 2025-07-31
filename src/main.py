from config import FLAGS

from solver.mtl.train import train_main, inference
from dse import ExhaustiveExplorer
from saver import saver
from utils import get_root_path, load, get_src_path, plot_dist, plot_models_per_graph
import torch

from os.path import join, dirname
from glob import iglob

import os
import numpy as np
import random

import config
TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
POLY_KERNEL = config.POLY_KERNEL

from data import get_data_list, MyOwnDataset

def set_seed(seed):
    # 设置环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 设置各库种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 配置PyTorch CUDA后端
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    


if __name__ == '__main__':
    
    set_seed(123)

    if not FLAGS.force_regen:
        train_dataset = MyOwnDataset(is_train=True)
        test_dataset = MyOwnDataset(is_train=False)
        print('read dataset')
    else:   
        get_data_list()
        print("regen dataset over, exit!")
        exit(0)
    
    pragma_dim = load(FLAGS.pragma_dim_path)
                
    def inf_main(dataset):
        if type(FLAGS.model_path) is None:
            saver.error('model_path must be set for running the inference.')
            raise RuntimeError()
        else:
            for ind, model_path in enumerate(FLAGS.model_path):
                inference(dataset, init_pragma_dict=pragma_dim, model_path=model_path, model_id=ind)
                inference(dataset, init_pragma_dict=pragma_dim, model_path=model_path, model_id=ind, is_train_set=True)
                if ind + 1 < len(FLAGS.model_path):
                    saver.new_sub_saver(subdir=f'run{ind+2}')
                    saver.log_info('\n\n')


    if FLAGS.subtask == 'inference':
        inf_main([train_dataset, test_dataset])
    elif FLAGS.subtask == 'dse':
        if FLAGS.dataset == 'harp':
            first_dse = True
            if FLAGS.plot_dse: graph_types = ['initial', 'extended', 'hierarchy']
            else: graph_types = [FLAGS.graph_type]

            for dataset in ['machsuite', 'poly']:
                path = join(get_root_path(), 'dse_database', dataset, 'config')
                path_graph = join(get_root_path(), 'dse_database', 'generated_graphs', dataset, 'processed')
                if dataset == 'machsuite':   
                    KERNELS = MACHSUITE_KERNEL
                elif dataset == 'poly':
                    KERNELS = POLY_KERNEL
                else:
                    raise NotImplementedError()
                
                for kernel in KERNELS:
                    if not FLAGS.all_kernels and not FLAGS.target_kernel in kernel:
                        continue
                    plot_data = {}
                    for graph_type in graph_types:
                        saver.info('#'*65)
                        saver.info(f'Now processing {graph_type} graph')
                        saver.info('*'*65)
                        saver.info(f'Starting DSE for {kernel}')
                        saver.debug(f'Starting DSE for {kernel}')
                        FLAGS.target_kernel = kernel
                        if FLAGS.explorer == 'exhaustive':
                            explorer = ExhaustiveExplorer(path, kernel, path_graph, first_dse = first_dse, run_dse = True, pragma_dim = pragma_dim)
                            if FLAGS.plot_dse: plot_data[graph_type] = explorer.plot_data
                        else:
                            raise NotImplementedError()
                        saver.info('*'*65)
                        saver.info(f'')
                        first_dse = False

                    if FLAGS.plot_dse:
                        plot_models_per_graph(saver.plotdir, kernel, graph_types, plot_data, FLAGS.target)
        else:
            raise NotImplementedError()
    elif FLAGS.subtask == 'train':
        train_main(train_dataset, test_dataset, pragma_dim)
                
    else:
        raise NotImplementedError()

    saver.close()
