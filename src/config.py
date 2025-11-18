from utils import get_user, get_host, get_root_path
import argparse
import torch
from glob import iglob
from os.path import join

decoder_arch = []
root_path = get_root_path()

parser = argparse.ArgumentParser()                                   
TASK = 'regression'                             
parser.add_argument('--task', default=TASK)

SUBTASK = 'train'
# SUBTASK = 'inference'                               
parser.add_argument('--subtask', default=SUBTASK)
parser.add_argument('--plot_dse', default=False)


#################### visualization ####################
parser.add_argument('--vis_per_kernel', default=True) ## only tsne visualization for now 


######################## data ########################

TARGETS = ['perf', 'util-BRAM', 'util-DSP', 'util-LUT', 'util-FF',
           'total-BRAM', 'total-DSP', 'total-LUT', 'total-FF']


MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil_stencil2d',
                    'nw', 'md', 'stencil-3d']

POLY_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen', 
               'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver', 
               'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm', 
               'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
               'atax-medium', 'bicg-medium', 'gesummv-medium']


parser.add_argument('--force_regen', type=bool, default=False) ## must be set to True for the first time to generate the dataset

parser.add_argument('--min_allowed_latency', type=float, default=100.0) ## if latency is less than this, prune the point (used when synthesis is not valid)
EPSILON = 1e-3
parser.add_argument('--epsilon', default=EPSILON)
NORMALIZER = 1e7
parser.add_argument('--normalizer', default=NORMALIZER)
parser.add_argument('--util_normalizer', default=1)
MAX_NUMBER = 1e10
parser.add_argument('--max_number', default=MAX_NUMBER)

norm = 'speedup-log2' # 'const' 'log2' 'speedup' 'off' 'speedup-const' 'const-log2' 'none' 'speedup-log2'
parser.add_argument('--norm_method', default=norm)
parser.add_argument('--new_speedup', default=True) # new_speedup: same reference point across all, 
                                                    # old_speedup: base is the longest latency and different per kernel

parser.add_argument('--invalid', type = bool, default=False ) # False: do not include invalid designs

parser.add_argument('--encode_log', type = bool, default=False)
v_db = 'v18' # 'v20': v20 database, 'v18': v18 database
parser.add_argument('--v_db', default=v_db) # if set to true uses the db of the new version of the tool: 2020.2

test_kernels = None
parser.add_argument('--test_kernels', default=test_kernels)
target_kernel = None
# target_kernel = 'gemm-blocked'
parser.add_argument('--target_kernel', default=target_kernel)
if target_kernel == None:
    all_kernels = True
else:
    all_kernels = False
parser.add_argument('--all_kernels', type = bool, default=all_kernels)


dataset = 'hlsyn'
parser.add_argument('--dataset', default=dataset)

benchmark = ['machsuite', 'poly']
parser.add_argument('--benchmarks', default=benchmark)

tag = 'whole-machsuite-poly'
parser.add_argument('--tag', default=tag)


###################### graph type ######################
graph_type = 'extended-pseudo-block-connected-hierarchy'
parser.add_argument('--graph_type', default=graph_type)

################## model architecture ##################
pragma_as_MLP, type_parallel, type_merge = True, '2l', '2l'
gnn_layer_after_MLP = 1
pragma_MLP_hidden_channels, merge_MLP_hidden_channels = None, None
if gnn_layer_after_MLP == 1: model_ver = 'pragma_as_MLP'
        
if type_parallel == '2l': pragma_MLP_hidden_channels = '[in_D // 2]'
elif type_parallel == '3l': pragma_MLP_hidden_channels = '[in_D // 2, in_D // 4]'
        
if type_merge == '2l': merge_MLP_hidden_channels = '[in_D // 2]'
elif type_merge == '3l': merge_MLP_hidden_channels = '[in_D // 2, in_D // 4]'
else: raise NotImplementedError()
P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = True, True, False, 0.1, 153, 335
         

################# one-hot encoder ##################
encoder_path = f'{root_path}/dataset/encoders.klepto'         
encode_edge_position = True
        
parser.add_argument('--encoder_path', default=encoder_path)


################ model architecture #################
## edge attributes
parser.add_argument('--encode_edge', type=bool, default=True)
parser.add_argument('--encode_edge_position', type=bool, default=encode_edge_position)

num_layers = 6
parser.add_argument('--num_layers', type=int, default=num_layers) 
parser.add_argument('--num_features', default=num_features) 
parser.add_argument('--edge_dim', default=edge_dim) 

multi_target = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
if SUBTASK == 'class':
    multi_target = ['perf']
parser.add_argument('--target', default=multi_target)
parser.add_argument('--MLP_common_lyr', default=0)
gnn_type = 'transformer'
parser.add_argument('--gnn_type', type=str, default=gnn_type)
parser.add_argument('--dropout', type=float, default=dropout)

jkn_mode = 'max'
parser.add_argument('--jkn_mode', type=str, default=jkn_mode)
parser.add_argument('--jkn_enable', type=bool, default=True)
node_attention = True
parser.add_argument('--node_attention', type=bool, default=node_attention)
if node_attention:
    parser.add_argument('--node_attention_MLP', type=bool, default=False)

    separate_P = True
    parser.add_argument('--separate_P', type=bool, default=separate_P)
    separate_icmp = False
    parser.add_argument('--separate_icmp', type=bool, default=separate_icmp)
    parser.add_argument('--separate_T', type=bool, default=separate_T)
    parser.add_argument('--separate_pseudo', type=bool, default=separate_pseudo)

    if separate_P:
        parser.add_argument('--P_use_all_nodes', type=bool, default=P_use_all_nodes)
        
parser.add_argument('--gae_T', default=False)
parser.add_argument('--gae_P', default=False)

# if pragma_as_MLP:
#     assert graph_type == 'extended-pseudo-block-connected-hierarchy'
parser.add_argument('--gnn_layer_after_MLP', default=gnn_layer_after_MLP) ## number of message passing layers after MLP (pragma as MLP)
parser.add_argument('--pragma_as_MLP', default=pragma_as_MLP)
pragma_as_MLP_list = ['tile', 'pipeline', 'parallel']
parser.add_argument('--pragma_as_MLP_list', default=pragma_as_MLP_list)
pragma_scope = 'block'
parser.add_argument('--pragma_scope', default=pragma_scope)
keep_pragma_attribute = False if pragma_as_MLP else True
parser.add_argument('--keep_pragma_attribute', default=keep_pragma_attribute)
pragma_order = 'parallel_and_merge'
parser.add_argument('--pragma_order', default=pragma_order)
parser.add_argument('--pragma_MLP_hidden_channels', default=pragma_MLP_hidden_channels)
parser.add_argument('--merge_MLP_hidden_channels', default=merge_MLP_hidden_channels)


model_path = None  
model_path_list = []                                            
parser.add_argument('--model_path', default=model_path) ## list of models when used in DSE, if more than 1, ensemble inference must be on

ensemble = 0
ensemble_weights = None
parser.add_argument('--ensemble', type=int, default=ensemble)
parser.add_argument('--ensemble_weights', default=ensemble_weights)
class_model_path = None
parser.add_argument('--class_model_path', default=class_model_path)

pragma_dim_path = f'{root_path}/dataset/pragma_dim.klepto'
parser.add_argument('--pragma_dim_path', default=pragma_dim_path)

boost_base_model_path = None
parser.add_argument('--boost_base_model_path', default=boost_base_model_path)
parser.add_argument('--boost_use_mtl', default=True)



################ transfer learning #################
feature_extract = False                                              ## modify 
parser.add_argument('--feature_extract', default=feature_extract) # if set to true GNN encoder (or part of it) will be fixed and only MLP will be trained
if feature_extract:
    parser.add_argument('--random_MLP', default=False) # true: initialize MLP randomly
fix_gnn_layer = None ## if none, all layers will be fixed
fix_gnn_layer = 1 ## number of gnn layers to freeze, feature_extract should be set to True
parser.add_argument('--fix_gnn_layer', default=fix_gnn_layer) # if not set to none, feature_extract should be True
FT_extra = False
parser.add_argument('--FT_extra', default=FT_extra) ## fine-tune only on the new data points


################ training details #################
parser.add_argument('--save_model', type = bool, default=True)
resample = False
val_ratio = 0.15
parser.add_argument('--resample', default=resample) ## when resample is turned on, it will divide the dataset in round-robin and train multiple times to have all the points in train/test set
parser.add_argument('--val_ratio', type=float, default=val_ratio) # ratio of database for validation set
parser.add_argument('--activation', default='elu')     
parser.add_argument('--D', type=int, default=64)
parser.add_argument('--lr', default=0.0007) ## default=0.001
scheduler, warmup, weight_decay = None, None, 0
scheduler, warmup, weight_decaty = 'cosine', 'linear', 1e-4
parser.add_argument('--weight_decay', default=weight_decay) ## default=0.0001, larger than 1e-4 didn't help original graph P+T
parser.add_argument("--scheduler", default=scheduler)
parser.add_argument("--warmup", default=warmup)

parser.add_argument('--random_seed', default=123) ## default=100
batch_size = 64
parser.add_argument('--batch_size', type=int, default=batch_size)

loss = 'MSE' # RMSE, MSE, 
parser.add_argument('--loss', type=str, default=loss) 

if model_path == None:
    if TASK == 'regression':
        epoch_num = 1500
    else:
        epoch_num = 500
else:
    epoch_num = 1500
parser.add_argument('--epoch_num', type=int, default=epoch_num)

gpu = 0
device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')
parser.add_argument('--device', default=device)


################# DSE details ##################
explorer = 'exhaustive'
parser.add_argument('--explorer', default=explorer)

model_tag = 'test'
parser.add_argument('--model_tag', default=model_tag)

parser.add_argument('--prune_util', default=True) # only DSP and BRAM
parser.add_argument('--prune_class', default=True)

parser.add_argument('--print_every_iter', type=int, default=100)

plot = True
parser.add_argument('--plot_pred_points', type=bool, default=plot)


################ 

"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

parser.add_argument('--num_clusters', default=5)

parser.add_argument('--enable_ikdl', default=False)

parser.add_argument('--enable_boost', default=False)

parser.add_argument('--enable_mtl', default=True)   # not use now

FLAGS = parser.parse_args()
