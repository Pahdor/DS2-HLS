import torch
import torch.nn as nn

from config import FLAGS

from config import FLAGS
from saver import saver
from utils import MLP, _get_y_with_target, MLP_multi_objective

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GlobalAttention, JumpingKnowledge, TransformerConv, GCNConv
from torch_geometric.nn import global_add_pool

from nn_att import MyGlobalAttention
from torch.nn import Sequential, Linear, ReLU

from collections import OrderedDict, defaultdict
from torch_geometric.nn import TransformerConv


from solver.boost.net import Net
from solver.boost.mtl_net import Net as MTLNet

class DeltaEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 初始化最后一层的权重和偏置为0，使初始输出为0
        nn.init.constant_(self.delta_net[2].weight, 1e-6)
        nn.init.zeros_(self.delta_net[2].bias)

    def forward(self, embeddings):  # [T, d]
        deltas = []
        for t in range(len(embeddings)-1):
            delta_input = torch.cat([embeddings[t], embeddings[t+1]], dim=-1)
            delta = self.delta_net(delta_input)
            deltas.append(delta)
        return torch.stack(deltas, dim=-1)  # [T-1, d]



class BoostNet(nn.Module):
    def __init__(self, in_channels, edge_dim = 0, init_pragma_dict = None, task = FLAGS.task, num_layers = FLAGS.num_layers, D = FLAGS.D, target = FLAGS.target, boost_base_model_path=FLAGS.boost_base_model_path):
        super(BoostNet, self).__init__()
        if FLAGS.boost_use_mtl:
            self.base_model = MTLNet(in_channels, edge_dim = edge_dim, init_pragma_dict = init_pragma_dict, task = task, num_layers = num_layers, D = D, target = target)
        else:
            self.base_model = Net(in_channels, edge_dim = edge_dim, init_pragma_dict = init_pragma_dict, task = task, num_layers = num_layers, D = D, target = target)
        self.base_model.load_state_dict(torch.load(boost_base_model_path))
        self.fix_base_net()
        self.target_list = FLAGS.target
        self.boost_model = nn.ModuleDict()
        
        self.delta_encoder = DeltaEncoder(64, 64)
        
        self.task = task
        
        if task == 'regression':
            self.out_dim = 1
            self.MLP_out_dim = 1
            self.loss_function = nn.MSELoss()
        else:
            self.out_dim = 2
            self.MLP_out_dim = 2
            self.loss_function = nn.CrossEntropyLoss()
            
        # if FLAGS.node_attention:
        #     dim = FLAGS.separate_T + FLAGS.separate_P + FLAGS.separate_pseudo + FLAGS.separate_icmp
        #     in_D = dim * D
        # else:
        #     in_D = D
        in_D = 64 * 3
        if D > 64:
            hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
        else:
            hidden_channels = [D // 2, D // 4, D // 8]
            
        for target in self.target_list:
                self.boost_model[target] = MLP(in_D, self.MLP_out_dim, activation_type=FLAGS.activation,
                                        hidden_channels=hidden_channels,
                                        num_hidden_lyr=len(hidden_channels))
        
        
        
    def fix_base_net(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, data, inference_mode = False):
        if FLAGS.boost_use_mtl:
            out_dict, _, _, _, _, _, _, _, _, gae_loss, diff_embed1, diff_embed2, graph_embed = self.base_model(data)
        else:
            out_dict, _, _, gae_loss, diff_embed1, diff_embed2, graph_embed = self.base_model(data)
        
        total_loss = 0
        loss_dict = {}
        
        diff_embed = self.delta_encoder([diff_embed1, diff_embed2]).squeeze()
        graph_embed = torch.concat([diff_embed, graph_embed], dim=-1)
        
        for target_name in self.target_list:
            out = self.boost_model[target_name](graph_embed)
            out_dict[target_name] = out_dict[target_name] + out
            if not inference_mode:
                y = _get_y_with_target(data, f"diff_{target_name}")
                if self.task == 'regression':
                    target = y.view((len(y), self.out_dim))
                    # print('target', target.shape)
                    if FLAGS.loss == 'RMSE':
                        loss = torch.sqrt(self.loss_function(out, target))
                        # loss = mean_squared_error(target, out, squared=False)
                    elif FLAGS.loss == 'MSE':
                        loss = self.loss_function(out, target) 
                    else:
                        raise NotImplementedError()
                    # print('loss', loss.shape)
                else:
                    target = y.view((len(y)))
                    loss = self.loss_function(out, target)
                loss_dict[f"{target_name}"] = loss
                total_loss += loss
            else:
                y = _get_y_with_target(data, f"{target_name}")
                if self.task == 'regression':
                    target = y.view((len(y), self.out_dim))
                    # print('target', target.shape)
                    if FLAGS.loss == 'RMSE':
                        loss = torch.sqrt(self.loss_function(out_dict[target_name], target))
                        # loss = mean_squared_error(target, out, squared=False)
                    elif FLAGS.loss == 'MSE':
                        loss = self.loss_function(out_dict[target_name], target) 
                    else:
                        raise NotImplementedError()
                    # print('loss', loss.shape)
                else:
                    target = y.view((len(y)))
                    loss = self.loss_function(out, target)
                loss_dict[f"{target_name}"] = loss
                total_loss += loss
            out_dict[f"diff_{target_name}"] = out
            
        return out_dict, total_loss, loss_dict, gae_loss