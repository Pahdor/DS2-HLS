
from data import get_data_list, MyOwnDataset
from torch_geometric.data import DataLoader, Data
import torch

from solver.mtl.model import Net
from utils import get_root_path, load, get_src_path, plot_dist, plot_models_per_graph
from tqdm import tqdm
from config import FLAGS

def gen_diff_dataset(is_train=True):
    dataset = MyOwnDataset(is_train=is_train)
    num_features = dataset[0].num_features
    edge_dim = dataset[0].edge_attr.shape[1]
    pragma_dim = load(FLAGS.pragma_dim_path)
    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim)
    model.load_state_dict(torch.load(FLAGS.boost_base_model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    idx = 0
    if is_train:
        save_dir = f"/home/kzw3933/Desktop/HLS-GNN/dataset/save/{FLAGS.v_db}_MLP-True-extended-pseudo-block-connected-hierarchy-regression_edge-position-True_norm_with-invalid_False-normalization_speedup-log2_tag_whole-machsuite-poly_perfutil-DSPutil-BRAMutil-LUTutil-FF/boost_train"
    else:
        save_dir = f"/home/kzw3933/Desktop/HLS-GNN/dataset/save/{FLAGS.v_db}_MLP-True-extended-pseudo-block-connected-hierarchy-regression_edge-position-True_norm_with-invalid_False-normalization_speedup-log2_tag_whole-machsuite-poly_perfutil-DSPutil-BRAMutil-LUTutil-FF/boost_test"
    {FLAGS.v_db}
    with torch.no_grad():
        for data in tqdm(dataloader):
            data = data.to(device)
            out_dict, *rest = model(data)  
            data = data.cpu()
            item = Data(
                gname=data.gname[0],
                vname=data.vname[0],
                idx=data.idx[0],
                x=data.x,
                key=data.key[0],
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                edge_type=data.edge_type,
                kernel=data.kernel[0],
                X_contextnids=data.X_contextnids,
                X_pragmanids=data.X_pragmanids,                    
                X_pragmascopenids=data.X_pragmascopenids,                    
                X_pseudonids=data.X_pseudonids,    
                X_icmpnids=data.X_icmpnids,    
                X_pragma_per_node=data.X_pragma_per_node,            
                pragmas=data.pragmas,
                kernel_name=data.kernel_name[0],
                design_name=data.design_name[0],
                perf=data.perf,
                actual_perf=data.actual_perf,
                util_BRAM=data.util_BRAM,
                util_DSP=data.util_DSP,
                util_LUT=data.util_LUT,
                util_FF=data.util_FF,
                total_BRAM=data.total_BRAM,
                total_DSP=data.total_DSP,
                total_LUT=data.total_LUT,
                total_FF=data.total_FF,
                diff_perf=data.perf - out_dict['perf'].squeeze().cpu(),
                diff_util_BRAM=data.util_BRAM - out_dict['util-BRAM'].squeeze().cpu(),
                diff_util_DSP=data.util_DSP - out_dict['util-DSP'].squeeze().cpu(),
                diff_util_LUT=data.util_LUT - out_dict['util-LUT'].squeeze().cpu(),
                diff_util_FF=data.util_FF - out_dict['util-FF'].squeeze().cpu()
            )
            torch.save(item, f"{save_dir}/data_{idx}.pt")
            idx += 1
            
    
if __name__ == '__main__':
    gen_diff_dataset(is_train=True)
    gen_diff_dataset(is_train=False)
    