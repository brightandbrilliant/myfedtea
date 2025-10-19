import os
import torch
from collections import deque, defaultdict
from Client import Client
from Model.GraphSage import GraphSAGE
from Model.ResMLP import ResMLP
from Cluster import (
    gnn_embedding_kmeans_cluster,
    gnn_embedding_spectral_cluster,
    compute_anchor_embedding_differences,
    build_cluster_cooccurrence_matrix,
    extract_clear_alignments
)
from Parse_Anchors import read_anchors, parse_anchors
from edge_ops import (extract_augmented_pos_edges,
                      construct_augmented_neg_edges,
                      build_positive_edge_dict,
                      build_edge_type_alignment)
from Utils import (set_seed,
                   split_client_data,
                   draw_loss_plot)


def load_all_clients(pyg_data_paths, encoder_params, decoder_params, training_params, device):
    """
    加载所有客户端及其数据和模型。

    Args:
        pyg_data_paths (list): PyG Data 文件路径列表。
        encoder_params (dict): 编码器模型参数。
        decoder_params (dict): 解码器模型参数。
        training_params (dict): 训练参数 (用于初始化 Client)。
        device (torch.device): 设备。

    Returns:
        clients (list): Client 对象列表。
        raw_data_list (list): 原始 PyG Data 对象列表。
    """
    clients, raw_data_list = [], []

    for client_id, path in enumerate(pyg_data_paths):
        raw_data = torch.load(path)
        raw_data_list.append(raw_data)

        data = split_client_data(raw_data, device=device)

        encoder = GraphSAGE(**encoder_params)
        decoder = ResMLP(input_dim=encoder_params['output_dim'] * 2, **decoder_params)

        client = Client(
            client_id=client_id,
            data=data,
            encoder=encoder,
            decoder=decoder,
            device=device,
            lr=training_params['lr'],
            weight_decay=training_params['weight_decay']
        )
        clients.append(client)

    return clients, raw_data_list


def average_state_dicts(state_dicts):
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key] for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


def evaluate_all_clients(clients, use_test=False):
    metrics = []
    for i, client in enumerate(clients):
        acc, recall, precision, f1 = client.evaluate(use_test=use_test)
        metrics.append((acc, recall, precision, f1))
        print(f"Client {client.client_id}: Acc={acc:.4f}, Recall={recall:.4f}, Prec={precision:.4f}, F1={f1:.4f}")
    avg = torch.tensor(metrics).mean(dim=0).tolist()
    print(f"\n===> Average: Acc={avg[0]:.4f}, Recall={avg[1]:.4f}, Prec={avg[2]:.4f}, F1={avg[3]:.4f}")
    return avg


def aggregate_from_window(sliding_window, top_percent=0.3):
    aggregate = defaultdict(int)
    for it in sliding_window:
        for pair, count in it.items():
            aggregate[pair] += count
    sorted_items = sorted(aggregate.items(), key=lambda x: x[1], reverse=True)
    cutoff = max(1, int(len(sorted_items) * top_percent))
    return dict(sorted_items[:cutoff])


# !!!此处需要修改: 新增预训练函数 pretrain_fedavg
def pretrain_fedavg(clients, pretrain_rounds, training_params):
    """
    执行小规模的 FedAvg 预训练。

    Args:
        clients (list): Client 对象列表。
        pretrain_rounds (int): 预训练通信轮次。
        training_params (dict): 包含 'local_epochs' 的训练参数。

    Returns:
        global_encoder_state (dict): 最终全局编码器的状态字典。
        final_decoder_states (list): 最终每个客户端解码器的状态字典列表。
    """
    print("\n========= Phase 1: FedAvg Pre-training Start =========")

    # 确保所有客户端都处于训练模式
    for client in clients:
        client.encoder.train()
        client.decoder.train()

    for rnd in range(1, pretrain_rounds + 1):
        if rnd % 10 == 0 or rnd == pretrain_rounds:
            print(f"--- Pretrain Round {rnd} ---")
            # 评估（可选，可以放在评估函数中）

        for client in clients:
            # 执行本地训练
            for _ in range(training_params['local_epochs']):
                client.train()

        # 聚合编码器和解码器
        encoder_states = [client.get_encoder_state() for client in clients]
        decoder_states = [client.get_decoder_state() for client in clients]
        global_encoder_state = average_state_dicts(encoder_states)
        global_decoder_state = average_state_dicts(decoder_states)

        # 分发全局模型状态
        for client in clients:
            client.set_encoder_state(global_encoder_state)
            client.set_decoder_state(global_decoder_state)
            client.last_encoder_state = {k: v.cpu().clone() for k, v in global_encoder_state.items()}
            client.last_decoder_state = {k: v.cpu().clone() for k, v in global_decoder_state.items()}

    print("========= Phase 1: FedAvg Pre-training Finished =========")
    return


def Cluster_and_Build(clients, anchor_path, anchor_point, nClusters, device):
    cluster_labels = []

    print("==================Clustering Start==================")
    # 1. 重新进行聚类，使用预训练后的编码器和新的聚类函数
    for client in clients:
        labels, _ = gnn_embedding_kmeans_cluster(client.data, client.encoder, n_clusters=nClusters, device=device)
        cluster_labels.append(labels)

    # 2. 重新构建 edge_dicts 和对齐矩阵
    edge_dicts = [build_positive_edge_dict(clients[i].data, cluster_labels[i]) for i in range(len(clients))]

    client_pos_edges = [
        set(map(tuple, clients[k].data.edge_index.t().tolist())) for k in range(len(clients))
    ]

    anchor_raw = read_anchors(anchor_path)
    anchor_pairs = parse_anchors(anchor_raw, point=anchor_point)

    clients[0].encoder.eval()
    clients[1].encoder.eval()
    z1 = clients[0].encoder(clients[0].data.x, clients[0].data.edge_index).detach()
    z2 = clients[1].encoder(clients[1].data.x, clients[1].data.edge_index).detach()
    results = compute_anchor_embedding_differences(z1, z2, anchor_pairs, device=device)

    print("==================Alignment Start==================")
    co_matrix = build_cluster_cooccurrence_matrix(cluster_labels[0], cluster_labels[1], results, nClusters,
                                                  top_percent=0.75)
    alignment1 = extract_clear_alignments(co_matrix, min_ratio=0.25, min_count=30, mode=1)
    alignment2 = extract_clear_alignments(co_matrix, min_ratio=0.25, min_count=30, mode=2)
    edge_alignment1 = build_edge_type_alignment(alignment1, nClusters)
    edge_alignment2 = build_edge_type_alignment(alignment2, nClusters)
    edge_alignments = [edge_alignment1, edge_alignment2]

    return cluster_labels, edge_dicts, client_pos_edges, edge_alignments




if __name__ == "__main__":
    seed_ = 826
    set_seed(seed_)

    data_dir = "../Parsed_dataset/dblp"
    anchor_path = "../dataset/dblp/anchors.txt"
    pyg_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    encoder_params = {
        'input_dim': torch.load(pyg_data_files[0]).x.shape[1],
        'hidden_dim': 128,
        'output_dim': 64,
        'num_layers': 3,
        'dropout': 0.5
    }
    decoder_params = {'hidden_dim': 128, 'num_layers': 8, 'dropout': 0.3}
    training_params = {'lr': 0.001, 'weight_decay': 1e-4, 'local_epochs': 5}

    num_rounds = 700
    top_fp_fn_percent = 0.3
    enhance_interval = 30
    top_k_pos_per_type = 100
    top_k_neg_per_type = 100
    nClusters = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    anchor_point = 9086
    pretrain_rounds = 200

    print("==================Pretraining Start==================")
    # Phase 1: 预训练 FedAvg
    clients, raw_data_list = load_all_clients(
        pyg_data_files, encoder_params, decoder_params, training_params, device
    )

    pretrain_fedavg(clients, pretrain_rounds, training_params)

    # 1. 重新进行聚类，使用预训练后的编码器和新的聚类函数
    """
    cluster_labels = []

    print("==================Clustering Start==================")
    for client in clients:
        labels, _ = gnn_embedding_kmeans_cluster(client.data, client.encoder, n_clusters=nClusters, device=device)
        cluster_labels.append(labels)

    # 2. 重新构建 edge_dicts 和对齐矩阵
    edge_dicts = [build_positive_edge_dict(clients[i].data, cluster_labels[i]) for i in range(len(clients))]

    client_pos_edges = [
        set(map(tuple, clients[k].data.edge_index.t().tolist())) for k in range(len(clients))
    ]

    anchor_raw = read_anchors(anchor_path)
    anchor_pairs = parse_anchors(anchor_raw, point=anchor_point)

    clients[0].encoder.eval()
    clients[1].encoder.eval()
    z1 = clients[0].encoder(clients[0].data.x, clients[0].data.edge_index).detach()
    z2 = clients[1].encoder(clients[1].data.x, clients[1].data.edge_index).detach()
    results = compute_anchor_embedding_differences(z1, z2, anchor_pairs, device=device)

    print("==================Alignment Start==================")
    co_matrix = build_cluster_cooccurrence_matrix(cluster_labels[0], cluster_labels[1], results, nClusters,
                                                  top_percent=0.75)
    alignment1 = extract_clear_alignments(co_matrix, min_ratio=0.25, min_count=30, mode=1)
    alignment2 = extract_clear_alignments(co_matrix, min_ratio=0.25, min_count=30, mode=2)
    edge_alignment1 = build_edge_type_alignment(alignment1, nClusters)
    edge_alignment2 = build_edge_type_alignment(alignment2, nClusters)
    """


    best_f1 = -1
    best_encoder_state = None
    best_decoder_states = None

    # 初始化滑动窗口
    sliding_fn_window = [deque(maxlen=5) for _ in range(len(clients))]
    sliding_fp_window = [deque(maxlen=5) for _ in range(len(clients))]
    # sliding_loss_window = [deque(maxlen=20) for _ in range(len(clients))]
    loss_record = [[], []]
    # augment_flag = [False, False]
    rnds = [-1, -1]
    best_rnd = 0
    # last_diff = [10000, 10000]  # 设一个很大的值
    fn_fp_ignore_flags = [False, False]
    start_rnd = 300

    cluster_labels, edge_dicts, client_pos_edges, edge_alignments = Cluster_and_Build(clients, anchor_path,
                                                                      anchor_point, nClusters, device)
    edge_alignment1, edge_alignment2 = edge_alignments[0], edge_alignments[1]

    print("\n================ Federated Training Start ================")
    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        clients[0].encoder.eval()
        clients[1].encoder.eval()
        z_others = [client.encoder(client.data.x, client.data.edge_index).detach() for client in clients]

        if rnd % start_rnd == 0:
            cluster_labels, edge_dicts, client_pos_edges, edge_alignments = Cluster_and_Build(clients, anchor_path,
                                                                                              anchor_point, nClusters,
                                                                                              device)
            edge_alignment1, edge_alignment2 = edge_alignments[0], edge_alignments[1]


        for i, client in enumerate(clients):
            if rnd >= start_rnd and fn_fp_ignore_flags[i] is False:
                fn, fp = client.analyze_prediction_errors(cluster_labels[i], use_test=False,
                                                          top_percent=top_fp_fn_percent)
                sliding_fn_window[i].append(fn)
                sliding_fp_window[i].append(fp)
            elif rnd < start_rnd:
                # 在 FedAvg 阶段，确保 FN/FP 窗口是空的，或者不使用
                pass
            else:  # fn_fp_ignore_flags[i] is True
                fn_fp_ignore_flags[i] = False

            if rnd % enhance_interval == 0 and rnd >= start_rnd:
                aggregated_fn = aggregate_from_window(sliding_fn_window[i], top_percent=top_fp_fn_percent)
                aggregated_fp = aggregate_from_window(sliding_fp_window[i], top_percent=top_fp_fn_percent)

                j = 1 - i
                pos_edge_list = extract_augmented_pos_edges(
                    aggregated_fn,
                    edge_dicts[j],
                    edge_alignment1 if i == 0 else edge_alignment2,
                    client_j=clients[j],
                    top_k=top_k_pos_per_type
                )

                neg_edge_list = construct_augmented_neg_edges(
                    aggregated_fp,
                    edge_alignment1 if i == 0 else edge_alignment2,  # 客户端 i 到 j 的对齐矩阵
                    cluster_labels[j],  # 客户端 j 的聚类标签
                    client_pos_edges[j],  # 客户端 j 的正边集合 (用于排除)
                    clients[j],
                    top_k=top_k_neg_per_type
                )

                client.inject_augmented_positive_edges(pos_edge_list, z_others[j])
                client.inject_augmented_negative_edges(neg_edge_list, z_others[j])

        for i, client in enumerate(clients):
            loss_avg = 0
            for _ in range(training_params['local_epochs']):
                loss = client.train()
                loss_avg += loss

            if rnd >= start_rnd and rnd % enhance_interval == 0:
                # print("Negative Augmentation Implementing.")
                client.train_on_augmented_negatives()
                client.encoder.eval()
                client.decoder.eval()
                fn_fp_ignore_flags[i] = True
            if rnd >= start_rnd and rnd % enhance_interval == enhance_interval // 2:
                # print("Positive Augmentation Implementing.")
                client.train_on_augmented_positives()
                client.encoder.eval()
                client.decoder.eval()
                fn_fp_ignore_flags[i] = True

            loss_avg /= training_params['local_epochs']
            loss_record[i].append(loss_avg)
            # print(f'Client{i} loss: {loss_avg}')

        encoder_states = [client.get_encoder_state() for client in clients]
        decoder_states = [client.get_decoder_state() for client in clients]
        global_encoder_state = average_state_dicts(encoder_states)

        for client in clients:
            client.set_encoder_state(global_encoder_state)
            client.last_encoder_state = {k: v.cpu().clone() for k, v in global_encoder_state.items()}

        if rnd < start_rnd:
            global_decoder_state = average_state_dicts(decoder_states)
            for client in clients:
                client.set_decoder_state(global_decoder_state)
                client.last_decoder_state = {k: v.cpu().clone() for k, v in global_decoder_state.items()}
            decoder_states = [client.get_decoder_state() for client in clients]

        else:
            for i, client in enumerate(clients):
                client.last_decoder_state = {k: v.cpu().clone() for k, v in decoder_states[i].items()}

        avg_acc, avg_recall, avg_prec, avg_f1 = evaluate_all_clients(clients, use_test=True)

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_encoder_state = {k: v.clone().detach() for k, v in global_encoder_state.items()}
            best_decoder_states = []
            for state in decoder_states:
                cloned_state = {k: v.clone().detach() for k, v in state.items()}
                best_decoder_states.append(cloned_state)
            best_rnd = rnd
            print("===> New best model saved")

    print("\n================ Federated Training Finished ================")
    for i, client in enumerate(clients):
        client.set_encoder_state(best_encoder_state)
        client.set_decoder_state(best_decoder_states[i])

    print("\n================ Final Evaluation ================")
    evaluate_all_clients(clients, use_test=True)
    print(f"best rnd:{best_rnd}")
    print(f"best f1:{best_f1}")
    # draw_loss_plot(loss_record[0])
    # draw_loss_plot(loss_record[1])
