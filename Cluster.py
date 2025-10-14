import os
import torch
from torch_geometric.data import Data
from Parse_Anchors import read_anchors, parse_anchors
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from collections import Counter
import numpy as np

def count_filtered_anchor_clusters(results, cluster_labels, graph_idx, top_percent=0.75):
    """
    统计相似度排在前 top_percent 的锚点在聚类结果中各类别的出现次数。

    Args:
        results: [[idx1, idx2, diff], ...]，表示每对锚点的特征距离
        cluster_labels: 所有节点的聚类标签
        graph_idx: 1 或 2，表示统计 graph1 的 idx1 还是 graph2 的 idx2
        top_percent: 保留前百分之多少的相似度较高锚点（diff 越小越相似）

    Returns:
        cluster_counter: dict，{类别标签: 出现次数}
    """
    # 1. 根据差异度排序（越小越相似）
    results_sorted = sorted(results, key=lambda x: x[2])
    cutoff = int(len(results_sorted) * top_percent)
    filtered = results_sorted[:cutoff]

    # 2. 获取对应图的节点编号
    if graph_idx == 1:
        indices = [item[0] for item in filtered]
    else:
        indices = [item[1] for item in filtered]

    # 3. 统计聚类标签
    anchor_cluster_labels = [cluster_labels[i] for i in indices]
    cluster_counter = dict(Counter(anchor_cluster_labels))

    return cluster_counter


def load_pyg_data(file_path: str) -> Data:
    data = torch.load(file_path)
    if not isinstance(data, Data):
        raise ValueError(f"文件 {file_path} 中的对象不是 torch_geometric.data.Data 类型。")
    return data


def load_anchors(file_path: str, base_: int):
    anchor_list = read_anchors(file_path)
    point = base_
    anchor_list = parse_anchors(anchor_list, point)
    return anchor_list


def compute_diff(f1, f2, mode='euclidean'):
    if mode == 'euclidean':
        return torch.norm(f1 - f2, p=2).item()
    elif mode == 'cosine':
        return 1 - F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
    elif mode == 'dot':
        return torch.dot(f1, f2).item()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_anchor_feature_differences(data1, data2, anchor_pairs):
    results = []
    for pair in anchor_pairs:
        idx1, idx2 = pair[0], pair[1]
        f1, f2 = data1.x[idx1], data2.x[idx2]
        diff = compute_diff(f1, f2, 'euclidean')
        results.append([idx1, idx2, diff])
    return results


def compute_anchor_embedding_differences(z1, z2, anchor_pairs, device='cpu'):
    """
    使用 GNN 嵌入计算锚点对之间的差异。

    Args:
        z1 (torch.Tensor): 客户端 1 的所有节点 GNN 嵌入。
        z2 (torch.Tensor): 客户端 2 的所有节点 GNN 嵌入。
        anchor_pairs (list): 锚点对列表 [[idx1, idx2], ...]。
        device (str/torch.device): 计算设备。

    Returns:
        results (list): 包含 [idx1, idx2, diff] 的列表。
    """
    results = []

    # 确保嵌入在正确的设备上
    z1 = z1.to(device)
    z2 = z2.to(device)

    for pair in anchor_pairs:
        idx1, idx2 = pair[0], pair[1]

        # 使用嵌入 Z 进行计算
        emb1 = z1[idx1]
        emb2 = z2[idx2]

        # 使用 L2 范数作为欧氏距离，计算差异
        # 假设 compute_diff 使用的是 'euclidean'
        diff = compute_diff(emb1, emb2, 'euclidean')

        results.append([idx1, idx2, diff])

    return results


def kmeans_cluster_single(data: Data, n_clusters=5):
    """
    对单个图的数据进行 KMeans 聚类。

    Args:
        data: PyG Data 对象
        n_clusters: 聚类类别数

    Returns:
        cluster_labels: 每个节点对应的聚类类别标签
    """
    x = data.x.cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(x)
    labels, inertia = kmeans.labels_, kmeans.inertia_

    return labels, inertia
    # dblp 结果发现聚为10类比较好
    # wd 结果发现聚为7~8类比较好


def kmeans_cluster_new(features, n_clusters=5):
    """
    对输入的特征/嵌入进行 KMeans 聚类。

    Args:
        features: NumPy array 或 PyTorch Tensor，形状为 [num_nodes, feature_dim]
        n_clusters: 聚类类别数

    Returns:
        cluster_labels: 每个节点对应的聚类类别标签 (NumPy array)
        inertia: 聚类惯性
    """
    if isinstance(features, torch.Tensor):
        x = features.detach().cpu().numpy()
    else:
        x = features

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(x)
    labels, inertia = kmeans.labels_, kmeans.inertia_

    return labels, inertia


def gnn_embedding_kmeans_cluster(data, encoder, n_clusters=10, device='cpu'):
    """
    使用 GNN 编码器生成的嵌入进行 KMeans 聚类。

    Args:
        data: PyG Data 对象 (包含 x 和 edge_index)
        encoder: 预先训练或初始化的 GNN 模型
        n_clusters: 聚类类别数
        device: 模型和数据所在的设备

    Returns:
        cluster_labels: 每个节点对应的聚类类别标签 (NumPy array)
        inertia: 聚类惯性
    """
    encoder = encoder.to(device)
    data = data.to(device)
    encoder.eval()

    # 1. 生成 GNN 嵌入
    with torch.no_grad():
        # 注意：使用 detach() 防止梯度回传到编码器（如果它处于训练模式）
        z = encoder(data.x, data.edge_index).detach()

        # 2. 执行 KMeans 聚类
    labels, inertia = kmeans_cluster_new(z, n_clusters=n_clusters)

    # 返回标签，并保持在 CPU 上
    return labels, inertia


def spectral_cluster_new(features, n_clusters=10, random_state=42):
    """
    对输入的特征/嵌入进行谱聚类。
    谱聚类更适合处理非凸和结构化的嵌入数据，以提高聚类稳定性。

    Args:
        features: NumPy array 或 PyTorch Tensor，形状为 [num_nodes, feature_dim]
        n_clusters: 聚类类别数
        random_state: 随机种子，用于保证最终 K-Means 步骤的可复现性

    Returns:
        cluster_labels: 每个节点对应的聚类类别标签 (NumPy array)
        inertia: 聚类惯性 (由于 SpectralClustering 不直接提供，我们返回 0)
    """
    if isinstance(features, torch.Tensor):
        x = features.detach().cpu().numpy()
    else:
        x = features

    # ------------------------------------------------------------------
    # 核心：使用 SpectralClustering
    # n_init=10 提高鲁棒性；affinity='nearest_neighbors' 适合稀疏结构
    # ------------------------------------------------------------------
    # 注意：如果节点数 N 很大 (>10k-20k)，这步计算量会很大！
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels='kmeans',  # 使用 K-Means 对谱嵌入进行聚类
        random_state=random_state,
        n_init=10,  # 对 K-Means 步骤运行 10 次
        affinity='nearest_neighbors'  # 基于 K-近邻图构建相似度矩阵
    )

    # 谱聚类在 fit_predict 内部完成相似度矩阵计算、拉普拉斯分解和 K-Means
    labels = clustering.fit_predict(x)

    # 谱聚类不直接计算 Inertia，我们返回 0 或执行一次 K-Means 计算
    # 为了保持函数签名一致，我们返回 0。
    inertia = 0

    return labels, inertia


def gnn_embedding_spectral_cluster(data, encoder, n_clusters=10, device='cpu'):
    """
    使用 GNN 编码器生成的嵌入进行聚类。现在默认使用谱聚类。

    Args:
        data: PyG Data 对象 (包含 x 和 edge_index)
        encoder: 预先训练或初始化的 GNN 模型
        n_clusters: 聚类类别数
        device: 模型和数据所在的设备

    Returns:
        cluster_labels: 每个节点对应的聚类类别标签 (NumPy array)
        inertia: 聚类惯性 (返回 0)
    """
    encoder = encoder.to(device)
    data = data.to(device)
    encoder.eval()

    # 1. 生成 GNN 嵌入
    with torch.no_grad():
        z = encoder(data.x, data.edge_index).detach()

    # 2. 执行谱聚类
    # !!! 替换了原来的 kmeans_cluster_new 调用 !!!
    labels, inertia = spectral_cluster_new(z, n_clusters=n_clusters)

    return labels, inertia



def build_cluster_cooccurrence_matrix(cluster_labels1, cluster_labels2, anchor_pairs,
                                      num_clusters, top_percent):
    """
    构建聚类共现计数矩阵。

    Args:
        cluster_labels1: 图1所有节点的聚类标签（list 或 array）
        cluster_labels2: 图2所有节点的聚类标签
        anchor_pairs: 筛选后的锚点对 [[i, j, diff], ...]
        num_clusters: 聚类类别总数 k

    Returns:
        matrix: ndarray，形状为 [k, k]，matrix[i][j] 表示图1第i类与图2第j类之间的锚点对数
    """
    results_sorted = sorted(anchor_pairs, key=lambda x: x[2])
    cutoff = int(len(results_sorted) * top_percent)
    filtered = results_sorted[:cutoff]
    matrix = np.zeros((num_clusters, num_clusters), dtype=int)

    for idx1, idx2, _ in filtered:
        c1 = cluster_labels1[idx1]
        c2 = cluster_labels2[idx2]
        matrix[c1][c2] += 1

    return matrix


def extract_clear_alignments(M, min_ratio=0.3, min_count=30, mode=1):
    """
    筛选出对齐明确的类。

    Args:
        M: 共现矩阵 (numpy.ndarray)，shape = [n_class1, n_class2]
        min_ratio: 主对齐类别所占比例的阈值
        min_count: 至少有多少个锚点才视为可信
        mode: 'row' 表示从图1看向图2，'col' 表示从图2看向图1

    Returns:
        alignments: dict，如 {类id: [(对应类id, 权重), ...]}
            若 mode == 'row'，键为图1的类id，值为图2类的列表；
            若 mode == 'col'，键为图2的类id，值为图1类的列表。
    """
    M = np.array(M)
    alignments = {}

    if mode == 1:
        # 每一行表示图1的一个类
        for i, row in enumerate(M):
            total = np.sum(row)
            if total < min_count:
                continue
            ratios = row / total
            major_idxs = np.where(ratios >= min_ratio)[0]
            if len(major_idxs) > 0:
                selected_counts = row[major_idxs]
                weights = selected_counts / selected_counts.sum()
                alignments[i] = list(zip(major_idxs.tolist(), weights.tolist()))
    elif mode == 2:
        # 每一列表示图2的一个类
        for j in range(M.shape[1]):
            col = M[:, j]
            total = np.sum(col)
            if total < min_count:
                continue
            ratios = col / total
            major_idxs = np.where(ratios >= min_ratio)[0]
            if len(major_idxs) > 0:
                selected_counts = col[major_idxs]
                weights = selected_counts / selected_counts.sum()
                alignments[j] = list(zip(major_idxs.tolist(), weights.tolist()))

    return alignments



if __name__ == "__main__":
    pyg_path1 = "../Parsed_dataset/wd/wd_1.pt"
    pyg_path2 = "../Parsed_dataset/wd/wd_2.pt"
    data1, data2 = load_pyg_data(pyg_path1), load_pyg_data(pyg_path2)
    anchor_path = "../dataset/wd/anchors.txt"
    anchor_pairs = load_anchors(anchor_path, 9714)
    print(anchor_pairs)
    results = compute_anchor_feature_differences(data1, data2, anchor_pairs)


    labels1, inertia1 = kmeans_cluster_single(data1, 7)
    labels2, inertia2 = kmeans_cluster_single(data2, 7)



    matrix = build_cluster_cooccurrence_matrix(labels1, labels2, results, 7, 0.75)

    alignment1 = extract_clear_alignments(matrix, 0.25, 30, 1)
    alignment2 = extract_clear_alignments(matrix, 0.25, 30, 2)
    print(alignment1)
    print(alignment2)

