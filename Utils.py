import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected


def set_seed(seed):
    """固定所有必要的随机种子以确保可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 强制 CUDA 使用确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def split_client_data(data, val_ratio=0.1, test_ratio=0.1, device='cpu'):
    # ... (保持不变)
    data = data.to(device)
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        neg_sampling_ratio=1.0
    )
    train_data, val_data, test_data = transform(data)

    val_mask = val_data.edge_label.bool()
    test_mask = test_data.edge_label.bool()
    train_data.val_pos_edge_index = val_data.edge_label_index[:, val_mask]
    train_data.val_neg_edge_index = val_data.edge_label_index[:, ~val_mask]
    train_data.test_pos_edge_index = test_data.edge_label_index[:, test_mask]
    train_data.test_neg_edge_index = test_data.edge_label_index[:, ~test_mask]

    return train_data




def draw_loss_plot(loss_record: list):
    x = []
    for i in range(1, len(loss_record)+1):
        x.append(i)
    plt.plot(x, loss_record, marker='o', linestyle='-', color='b', label='Client')

    plt.title('Loss-Round', fontsize=16)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.show()

